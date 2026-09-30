"""
Unit and integration tests for Mini-Project 3: Lake Victoria Fish Stock & Export Risk Model.
"""

import math
import numpy as np
import pytest

from assignment_3.src.fisheries.models import FishStock, PriceModel
from assignment_3.src.fisheries.risk import (
    RiskAssessor,
    evaluate_harvest_regimes,
    simulate_seasonal_closure_policy,
)


@pytest.fixture
def default_stock():
    return FishStock(intrinsic_growth_rate=0.40, carrying_capacity=10000.0, initial_biomass=4000.0)


@pytest.fixture
def default_price_model():
    return PriceModel(initial_price=12000.0, min_price=9000.0, max_price=16000.0, volatility=500.0)


def test_fibonacci_first_15_terms():
    """Task 1: Verifies first 15 terms of Fibonacci sequence."""
    fib = [1, 1]
    for _ in range(13):
        fib.append(fib[-1] + fib[-2])
    assert len(fib) == 15
    assert fib[:5] == [1, 1, 2, 3, 5]
    assert fib[-1] == 610  # 15th Fibonacci number


def test_fish_stock_properties(default_stock):
    """Verifies MSY, B_MSY, and h_MSY calculations."""
    # MSY = (r * K) / 4 = (0.4 * 10,000) / 4 = 1,000 tonnes
    assert math.isclose(default_stock.theoretical_msy, 1000.0)
    assert math.isclose(default_stock.biomass_at_msy, 5000.0)
    assert math.isclose(default_stock.harvest_rate_at_msy, 0.20)


def test_fish_stock_simulation_bounded(default_stock):
    """Verifies logistic simulation stays bounded by carrying capacity K."""
    bio, harv = default_stock.simulate(weeks=52, harvest_rate=0.10)
    assert len(bio) == 53
    assert len(harv) == 52
    assert np.all(bio <= default_stock.K * 1.05)
    assert np.all(bio >= 0.0)


def test_fish_stock_boundary_validation():
    """Boundary Case: Invalid negative biomass or harvest rate > 1.0 raises ValueError."""
    with pytest.raises(ValueError):
        FishStock(initial_biomass=-100.0)
    stock = FishStock()
    with pytest.raises(ValueError):
        stock.simulate(harvest_rate=1.5)


def test_price_model_bounded_walk(default_price_model):
    """Verifies that price simulations strictly respect the reflecting [9,000, 16,000] boundaries."""
    rng = np.random.default_rng(42)
    prices = default_price_model.simulate_prices(weeks=104, rng=rng)
    assert np.all(prices >= default_price_model.min_price)
    assert np.all(prices <= default_price_model.max_price)


def test_risk_assessor_classification():
    """Verifies risk tier classification based on CV thresholds."""
    assessor = RiskAssessor(low_threshold=0.10, moderate_threshold=0.20)
    assert assessor.classify_risk(0.05) == "Low Risk"
    assert assessor.classify_risk(0.15) == "Moderate Risk"
    assert assessor.classify_risk(0.25) == "High Risk"


def test_monte_carlo_var_calculation(default_stock, default_price_model):
    """Verifies Monte Carlo VaR_0.05 calculation with M = 500."""
    assessor = RiskAssessor()
    mc_res = assessor.run_monte_carlo_var(
        default_stock,
        default_price_model,
        harvest_rate=0.10,
        weeks=52,
        n_simulations=500,
        random_seed=42,
    )
    assert mc_res["n_simulations"] == 500
    assert mc_res["var_05_revenue_ugx"] < mc_res["mean_annual_revenue_ugx"]
    assert mc_res["shortfall_at_risk_ugx"] > 0


def test_evaluate_harvest_regimes(default_stock, default_price_model):
    """Verifies comparative evaluation across all 4 harvest regimes."""
    regimes = [0.05, 0.10, 0.20, 0.30]
    results = evaluate_harvest_regimes(default_stock, default_price_model, regimes, n_simulations=200, random_seed=42)
    assert len(results) == 4
    # Terminal biomass should decrease as harvest rate increases
    terminal_biomasses = [r["terminal_biomass_tonnes"] for r in results]
    assert terminal_biomasses[0] > terminal_biomasses[1] > terminal_biomasses[2] > terminal_biomasses[3]


def test_seasonal_closure_extension(default_stock, default_price_model):
    """Extension Case: Verifies 8-week seasonal closure results in positive biological recovery."""
    res = simulate_seasonal_closure_policy(
        default_stock,
        default_price_model,
        harvest_rate=0.20,
        closure_weeks_per_year=8,
        years=2,
        n_simulations=200,
        random_seed=42,
    )
    assert res["biomass_recovery_gain_tonnes"] > 0
    assert res["policy_terminal_biomass"] > res["baseline_terminal_biomass"]
