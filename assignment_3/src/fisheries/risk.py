"""
Risk assessment, Monte Carlo simulations, harvest regime evaluation,
and seasonal closure policy modeling for fisheries management.
"""

from __future__ import annotations
import statistics
from typing import Sequence, Dict, Any, List, Optional, Tuple
import numpy as np

from .models import FishStock, PriceModel


class RiskAssessor:
    """
    Classifies operational risk profiles based on dimensionless Coefficient of Variation (CV).
    
    Risk Tiers:
        - Low Risk: CV < 10% (0.10)
        - Moderate Risk: 10% <= CV < 20% (0.20)
        - High Risk / Critical Exposure: CV >= 20% (0.20)
    """

    def __init__(
        self,
        low_threshold: float = 0.10,
        moderate_threshold: float = 0.20,
    ) -> None:
        if low_threshold >= moderate_threshold:
            raise ValueError("low_threshold must be strictly less than moderate_threshold.")
        self.low_threshold = float(low_threshold)
        self.moderate_threshold = float(moderate_threshold)

    def classify_risk(self, cv: float) -> str:
        """Classifies risk tier given a Coefficient of Variation (decimal)."""
        if cv < self.low_threshold:
            return "Low Risk"
        elif cv < self.moderate_threshold:
            return "Moderate Risk"
        else:
            return "High Risk"

    def evaluate_revenue_series(
        self,
        weekly_revenues: Sequence[float] | np.ndarray,
    ) -> Dict[str, Any]:
        """
        Computes summary statistics and risk classification for a weekly revenue stream.
        """
        revs = [float(r) for r in weekly_revenues]
        if len(revs) < 2:
            raise ValueError("At least 2 revenue periods required.")

        mean_val = statistics.mean(revs)
        med_val = statistics.median(revs)
        var_val = statistics.variance(revs)
        std_val = statistics.stdev(revs)
        cv_val = std_val / mean_val if mean_val > 0 else 0.0

        return {
            "mean_weekly_revenue_ugx": mean_val,
            "median_weekly_revenue_ugx": med_val,
            "variance_revenue_ugx2": var_val,
            "stdev_revenue_ugx": std_val,
            "cv": cv_val,
            "cv_percent": cv_val * 100.0,
            "risk_tier": self.classify_risk(cv_val),
            "total_annual_revenue_ugx": sum(revs),
        }

    def run_monte_carlo_var(
        self,
        fish_stock: FishStock,
        price_model: PriceModel,
        harvest_rate: float = 0.10,
        weeks: int = 52,
        n_simulations: int = 1000,
        confidence_level: float = 0.05,
        seasonal_closure_weeks: Optional[Sequence[int]] = None,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Runs Monte Carlo simulation (M >= 1,000) over stochastic price realizations
        to calculate the empirical 5% Value-at-Risk (VaR_0.05) of annual export revenue.
        """
        rng = np.random.default_rng(random_seed)

        # 1. Biological harvest trajectory (tonnes -> kg: 1t = 1,000 kg)
        _, harvest_tonnes = fish_stock.simulate(
            weeks=weeks,
            harvest_rate=harvest_rate,
            seasonal_closure_weeks=seasonal_closure_weeks,
        )
        harvest_kg = harvest_tonnes * 1000.0

        annual_revenues = np.zeros(n_simulations, dtype=np.float64)

        for i in range(n_simulations):
            prices_sim = price_model.simulate_prices(weeks=weeks, rng=rng)
            annual_revenues[i] = np.sum(harvest_kg * prices_sim)

        # 5% Value-at-Risk (the revenue threshold at the bottom 5th percentile)
        var_05 = float(np.percentile(annual_revenues, confidence_level * 100.0))
        mean_annual_rev = float(np.mean(annual_revenues))
        std_annual_rev = float(np.std(annual_revenues, ddof=1))
        cv_annual = std_annual_rev / mean_annual_rev if mean_annual_rev > 0 else 0.0

        # Shortfall below mean at 5% VaR
        dollar_var = mean_annual_rev - var_05

        return {
            "n_simulations": n_simulations,
            "harvest_rate": harvest_rate,
            "total_harvest_tonnes": float(np.sum(harvest_tonnes)),
            "mean_annual_revenue_ugx": mean_annual_rev,
            "median_annual_revenue_ugx": float(np.median(annual_revenues)),
            "stdev_annual_revenue_ugx": std_annual_rev,
            "cv_annual": cv_annual,
            "var_05_revenue_ugx": var_05,
            "shortfall_at_risk_ugx": dollar_var,
            "risk_tier": self.classify_risk(cv_annual),
            "all_annual_revenues": annual_revenues,
        }


def evaluate_harvest_regimes(
    fish_stock: FishStock,
    price_model: PriceModel,
    harvest_regimes: Sequence[float] = (0.05, 0.10, 0.20, 0.30),
    n_simulations: int = 1000,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Evaluates comparative harvest rates h in {0.05, 0.10, 0.20, 0.30}.
    Tabulates terminal biomass, annual revenue, risk tier, and compares to theoretical MSY.
    """
    assessor = RiskAssessor()
    results = []

    for h in harvest_regimes:
        bio, harv = fish_stock.simulate(weeks=52, harvest_rate=h)
        terminal_biomass = float(bio[-1])
        total_harvest = float(np.sum(harv))

        # Monte Carlo revenue analysis
        mc_res = assessor.run_monte_carlo_var(
            fish_stock, price_model, harvest_rate=h, weeks=52,
            n_simulations=n_simulations, random_seed=random_seed
        )

        # MSY comparison
        msy = fish_stock.theoretical_msy
        harvest_vs_msy_ratio = total_harvest / msy if msy > 0 else 0.0

        results.append({
            "harvest_rate_h": h,
            "terminal_biomass_tonnes": terminal_biomass,
            "annual_harvest_tonnes": total_harvest,
            "theoretical_msy_tonnes": msy,
            "harvest_to_msy_ratio": harvest_vs_msy_ratio,
            "mean_revenue_ugx": mc_res["mean_annual_revenue_ugx"],
            "var_05_revenue_ugx": mc_res["var_05_revenue_ugx"],
            "cv_percent": mc_res["cv_annual"] * 100.0,
            "risk_tier": mc_res["risk_tier"],
            "biomass_trajectory": bio.tolist(),
            "revenues_distribution": mc_res["all_annual_revenues"],
        })

    return results


def simulate_seasonal_closure_policy(
    fish_stock: FishStock,
    price_model: PriceModel,
    harvest_rate: float = 0.20,
    closure_weeks_per_year: int = 8,
    years: int = 5,
    n_simulations: int = 1000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Extension Task: Formulates an 8-week annual biological seasonal closure policy
    prohibiting commercial fishing during peak spawning (weeks 18 to 25 each year).
    Simulates over a 5-year planning horizon (260 weeks).
    """
    total_weeks = years * 52
    closure_schedule: List[int] = []
    for yr in range(years):
        start_w = yr * 52 + 18
        closure_schedule.extend(range(start_w, start_w + closure_weeks_per_year))

    # 1. Unmanaged Baseline (Continuous Harvesting)
    bio_base, harv_base = fish_stock.simulate(weeks=total_weeks, harvest_rate=harvest_rate)

    # 2. Managed Policy (Seasonal 8-week Closure)
    bio_policy, harv_policy = fish_stock.simulate(
        weeks=total_weeks,
        harvest_rate=harvest_rate,
        seasonal_closure_weeks=closure_schedule,
    )

    rng = np.random.default_rng(random_seed)
    prices_5yr = price_model.simulate_prices(weeks=total_weeks, rng=rng)

    rev_base = np.sum(harv_base * 1000.0 * prices_5yr)
    rev_policy = np.sum(harv_policy * 1000.0 * prices_5yr)

    biomass_gain_tonnes = float(bio_policy[-1] - bio_base[-1])
    biomass_gain_pct = (biomass_gain_tonnes / bio_base[-1] * 100.0) if bio_base[-1] > 0 else 0.0
    net_revenue_impact_ugx = float(rev_policy - rev_base)

    return {
        "planning_years": years,
        "total_weeks": total_weeks,
        "closure_weeks_per_year": closure_weeks_per_year,
        "harvest_rate": harvest_rate,
        "baseline_terminal_biomass": float(bio_base[-1]),
        "policy_terminal_biomass": float(bio_policy[-1]),
        "biomass_recovery_gain_tonnes": biomass_gain_tonnes,
        "biomass_recovery_gain_pct": biomass_gain_pct,
        "baseline_5yr_revenue_ugx": float(rev_base),
        "policy_5yr_revenue_ugx": float(rev_policy),
        "net_revenue_impact_ugx": net_revenue_impact_ugx,
        "bio_trajectory_baseline": bio_base.tolist(),
        "bio_trajectory_policy": bio_policy.tolist(),
    }
