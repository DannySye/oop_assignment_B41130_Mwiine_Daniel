"""
Analytical, benchmarking, and financial costing engines for MicroGrid dispatch.
Includes solver benchmarking (iterative vs vectorized), volatility diagnostics,
cost calculations, and Monte Carlo sensitivity analysis connecting to condition number kappa(A).
"""

from __future__ import annotations
import timeit
import statistics
from typing import Tuple, Dict, Any, List
import numpy as np

from .models import MicroGrid


def benchmark_solvers(
    grid: MicroGrid,
    demands: np.ndarray,
    number_runs: int = 1000,
) -> Dict[str, Any]:
    """
    Benchmarks iterative Python loop vs single vectorized matrix RHS solve using timeit.
    """
    def run_iterative():
        return grid.solve_batch_iterative(demands)

    def run_vectorized():
        return grid.solve_batch_vectorized(demands)

    time_iterative = timeit.timeit(run_iterative, number=number_runs)
    time_vectorized = timeit.timeit(run_vectorized, number=number_runs)

    speedup = time_iterative / time_vectorized if time_vectorized > 0 else 1.0

    return {
        "number_runs": number_runs,
        "n_days": len(demands),
        "time_iterative_sec": time_iterative,
        "time_vectorized_sec": time_vectorized,
        "avg_time_iterative_ms": (time_iterative / number_runs) * 1000.0,
        "avg_time_vectorized_ms": (time_vectorized / number_runs) * 1000.0,
        "speedup_factor": speedup,
    }


def compute_dispatch_volatility(dispatch_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Calculates summary statistics (mean, variance, standard deviation)
    and coefficient of variation (CV = stdev / mean) using the statistics module.
    Determines which dispatch source exhibits higher relative volatility.
    """
    solar_vals = [float(v) for v in dispatch_matrix[:, 0]]
    battery_vals = [float(v) for v in dispatch_matrix[:, 1]]

    sol_mean = statistics.mean(solar_vals)
    sol_var = statistics.variance(solar_vals)
    sol_std = statistics.stdev(solar_vals)
    sol_cv = sol_std / sol_mean if sol_mean > 0 else 0.0

    bat_mean = statistics.mean(battery_vals)
    bat_var = statistics.variance(battery_vals)
    bat_std = statistics.stdev(battery_vals)
    bat_cv = bat_std / bat_mean if bat_mean > 0 else 0.0

    higher_vol = "Battery" if bat_cv > sol_cv else "Solar PV"

    return {
        "solar": {
            "mean": sol_mean,
            "variance": sol_var,
            "stdev": sol_std,
            "cv": sol_cv,
            "cv_percent": sol_cv * 100.0,
        },
        "battery": {
            "mean": bat_mean,
            "variance": bat_var,
            "stdev": bat_std,
            "cv": bat_cv,
            "cv_percent": bat_cv * 100.0,
        },
        "higher_relative_volatility": higher_vol,
        "cv_ratio_battery_to_solar": bat_cv / sol_cv if sol_cv > 0 else 1.0,
    }


def compute_financial_costs(
    dispatch_matrix: np.ndarray,
    unit_cost_solar: float = 150.0,
    unit_cost_battery: float = 450.0,
) -> Dict[str, Any]:
    """
    Calculates daily and aggregate monthly energy expenditure.
    Standard tariffs: Solar PV = UGX 150/kWh, Battery = UGX 450/kWh.
    """
    solar_kwh = dispatch_matrix[:, 0]
    battery_kwh = dispatch_matrix[:, 1]

    daily_solar_cost = solar_kwh * unit_cost_solar
    daily_battery_cost = battery_kwh * unit_cost_battery
    daily_total_cost = daily_solar_cost + daily_battery_cost

    total_solar_kwh = float(np.sum(solar_kwh))
    total_battery_kwh = float(np.sum(battery_kwh))
    total_energy_kwh = total_solar_kwh + total_battery_kwh

    total_solar_expenditure = float(np.sum(daily_solar_cost))
    total_battery_expenditure = float(np.sum(daily_battery_cost))
    total_monthly_expenditure = float(np.sum(daily_total_cost))

    average_levelized_cost = (
        total_monthly_expenditure / total_energy_kwh if total_energy_kwh > 0 else 0.0
    )

    return {
        "unit_cost_solar_ugx": unit_cost_solar,
        "unit_cost_battery_ugx": unit_cost_battery,
        "daily_solar_cost": daily_solar_cost.tolist(),
        "daily_battery_cost": daily_battery_cost.tolist(),
        "daily_total_cost": daily_total_cost.tolist(),
        "total_solar_kwh": total_solar_kwh,
        "total_battery_kwh": total_battery_kwh,
        "total_energy_kwh": total_energy_kwh,
        "total_solar_expenditure_ugx": total_solar_expenditure,
        "total_battery_expenditure_ugx": total_battery_expenditure,
        "total_monthly_expenditure_ugx": total_monthly_expenditure,
        "average_levelized_cost_ugx_kwh": average_levelized_cost,
    }


def monte_carlo_sensitivity_analysis(
    grid: MicroGrid,
    base_d1: float = 140.0,
    base_d2: float = 110.0,
    perturbation_pct: float = 0.05,
    n_iterations: int = 1000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Monte Carlo sensitivity analysis perturbing daily demands D1, D2 by +/- 5% over 1,000 iterations.
    Quantifies variance propagation into x and y and connects directly to condition number kappa(A).
    """
    rng = np.random.default_rng(random_seed)

    # Uniform perturbations within [-perturbation_pct, +perturbation_pct]
    d1_perturbed = base_d1 * (1.0 + rng.uniform(-perturbation_pct, perturbation_pct, size=n_iterations))
    d2_perturbed = base_d2 * (1.0 + rng.uniform(-perturbation_pct, perturbation_pct, size=n_iterations))

    demands_perturbed = np.column_stack([d1_perturbed, d2_perturbed])
    dispatch_results = grid.solve_batch_vectorized(demands_perturbed)

    x_sim = dispatch_results[:, 0]
    y_sim = dispatch_results[:, 1]

    # Input relative perturbations
    d1_rel_err = np.abs(d1_perturbed - base_d1) / base_d1
    d2_rel_err = np.abs(d2_perturbed - base_d2) / base_d2
    input_norm_rel_err = np.sqrt(d1_rel_err**2 + d2_rel_err**2)

    # Base solution
    base_x, base_y = grid.solve_day(base_d1, base_d2)
    x_rel_err = np.abs(x_sim - base_x) / base_x
    y_rel_err = np.abs(y_sim - base_y) / base_y
    output_norm_rel_err = np.sqrt(x_rel_err**2 + y_rel_err**2)

    # Empirical amplification ratio: ||delta x|| / ||x|| <= kappa(A) * ||delta b|| / ||b||
    empirical_amplification = output_norm_rel_err / np.maximum(input_norm_rel_err, 1e-9)
    max_amplification = float(np.max(empirical_amplification))
    mean_amplification = float(np.mean(empirical_amplification))

    return {
        "n_iterations": n_iterations,
        "perturbation_pct": perturbation_pct,
        "base_inputs": (base_d1, base_d2),
        "base_solution": (base_x, base_y),
        "x_mean": float(np.mean(x_sim)),
        "x_variance": float(np.var(x_sim, ddof=1)),
        "x_stdev": float(np.std(x_sim, ddof=1)),
        "y_mean": float(np.mean(y_sim)),
        "y_variance": float(np.var(y_sim, ddof=1)),
        "y_stdev": float(np.std(y_sim, ddof=1)),
        "condition_number_kappa": grid.condition_number,
        "max_empirical_error_amplification": max_amplification,
        "mean_empirical_error_amplification": mean_amplification,
        "satisfies_theoretical_bound": bool(max_amplification <= grid.condition_number * 2.0),
    }
