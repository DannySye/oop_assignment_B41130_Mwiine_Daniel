"""
Unit and integration tests for Mini-Project 2: Solar Micro-Grid Dispatch Planner.
"""

import math
import numpy as np
import pytest

from assignment_2.src.microgrid.models import MicroGrid, HybridMicroGrid
from assignment_2.src.microgrid.data import generate_synthetic_30day_demands, load_demand_csv
from assignment_2.src.microgrid.analytics import (
    benchmark_solvers,
    compute_dispatch_volatility,
    compute_financial_costs,
    monte_carlo_sensitivity_analysis,
)


@pytest.fixture
def standard_grid():
    return MicroGrid()


def test_microgrid_matrix_invariants(standard_grid):
    """Verifies det(A) = -5.0 and condition number kappa(A)."""
    assert math.isclose(standard_grid.determinant, -5.0, rel_tol=1e-5)
    assert not standard_grid.is_singular
    assert standard_grid.condition_number > 1.0


def test_microgrid_solve_day_analytical(standard_grid):
    """
    Verifies solve_day:
        3x + 2y = 130
        4x +  y = 115
    det = -5.
    x = -0.2(130) + 0.4(115) = -26 + 46 = 20.
    y = 0.8(130) - 0.6(115) = 104 - 69 = 35.
    """
    x, y = standard_grid.solve_day(130.0, 115.0)
    assert math.isclose(x, 20.0, rel_tol=1e-6)
    assert math.isclose(y, 35.0, rel_tol=1e-6)


def test_microgrid_vectorized_matches_iterative(standard_grid):
    """Verifies that vectorized batch solver produces identical results to loop solver."""
    demands = np.array([
        [130.0, 115.0],
        [140.0, 100.0],
        [150.0, 120.0],
    ])
    res_loop = standard_grid.solve_batch_iterative(demands)
    res_vec = standard_grid.solve_batch_vectorized(demands)
    assert np.allclose(res_loop, res_vec)


def test_feasibility_validation_and_nnls_fallback(standard_grid):
    """
    Boundary Case: Infeasible load causing negative dispatch triggers NNLS fallback.
    If D1 = 300, D2 = 10:
    x = -0.2(300) + 0.4(10) = -60 + 4 = -56 < 0 (Infeasible!)
    """
    result = standard_grid.solve_day_feasible(300.0, 10.0)
    assert not result["is_feasible"]
    assert result["fallback_triggered"]
    # Non-negative dispatch
    assert result["solar_x"] >= 0.0
    assert result["battery_y"] >= 0.0


def test_singular_matrix_raises_error():
    """Boundary Case: Singular matrix raises LinAlgError."""
    singular_A = np.array([[2.0, 4.0], [1.0, 2.0]])  # Row 1 is 2 * Row 2
    grid_sing = MicroGrid(singular_A)
    assert grid_sing.is_singular
    with pytest.raises(np.linalg.LinAlgError):
        grid_sing.solve_day(100.0, 50.0)


def test_synthetic_csv_generation_and_loading(tmp_path):
    """Verifies synthetic CSV generation and round-trip parsing."""
    csv_file = tmp_path / "test_demands.csv"
    demands = generate_synthetic_30day_demands(output_path=csv_file, random_seed=42)
    assert demands.shape == (30, 2)

    loaded_demands = load_demand_csv(csv_file)
    assert loaded_demands.shape == (30, 2)
    assert np.allclose(demands, loaded_demands, atol=0.01)


def test_financial_costing(standard_grid):
    """Verifies levelized energy cost computation."""
    dispatch = np.array([[10.0, 20.0]])  # 10 kWh solar, 20 kWh battery
    # Cost = 10 * 150 + 20 * 450 = 1500 + 9000 = 10500 UGX
    costs = compute_financial_costs(dispatch, unit_cost_solar=150.0, unit_cost_battery=450.0)
    assert costs["total_solar_expenditure_ugx"] == 1500.0
    assert costs["total_battery_expenditure_ugx"] == 9000.0
    assert costs["total_monthly_expenditure_ugx"] == 10500.0


def test_hybrid_microgrid_3x3_and_singularity():
    """Extension Case: Tests 3x3 hybrid system and singularity detection."""
    hybrid = HybridMicroGrid()
    assert hybrid.matrix_rank == 3
    sol = hybrid.solve_day_hybrid(150.0, 120.0, 100.0)
    assert len(sol) == 3

    # Create linearly dependent 3rd row: Row 3 = Row 1 + Row 2
    singular_A3 = np.array([
        [3.0, 2.0, 1.0],
        [4.0, 1.0, 2.0],
        [7.0, 3.0, 3.0],  # Exactly sum of row 1 and 2
    ])
    hybrid_sing = HybridMicroGrid(singular_A3)
    assert hybrid_sing.matrix_rank < 3
    with pytest.raises(np.linalg.LinAlgError):
        hybrid_sing.solve_day_hybrid(150.0, 120.0, 270.0)


def test_monte_carlo_sensitivity(standard_grid):
    """Extension Case: Verifies Monte Carlo sensitivity analysis runs deterministically."""
    mc_res = monte_carlo_sensitivity_analysis(
        standard_grid,
        base_d1=140.0,
        base_d2=110.0,
        perturbation_pct=0.05,
        n_iterations=500,
        random_seed=42,
    )
    assert mc_res["n_iterations"] == 500
    assert mc_res["x_variance"] > 0
    assert mc_res["y_variance"] > 0
    assert mc_res["max_empirical_error_amplification"] > 0
