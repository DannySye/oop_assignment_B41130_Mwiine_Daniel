"""
Unit and integration tests for Mini-Project 5: Taxi Route Revenue, Pricing & Fleet Planner.
"""

import math
import numpy as np
import pytest

from assignment_5.src.transport.models import Route, solve_ntinda_market_equilibrium
from assignment_5.src.transport.forecasters import (
    MovingAverageForecaster,
    SimpleExponentialSmoothingForecaster,
    LinearTrendForecaster,
    SeasonalNaiveForecaster,
    walk_forward_backtest,
    grid_search_optimal_ses_alpha,
)
from assignment_5.src.transport.fleet import (
    calculate_fleet_deployment,
    generate_60day_seasonal_demand,
)


@pytest.fixture
def ntinda_route():
    return Route("Kampala-Ntinda", [35, 40, 42, 50, 55, 60, 48, 52, 47, 45], fare_ugx=2000.0)


@pytest.fixture
def entebbe_route():
    return Route("Kampala-Entebbe", [60, 58, 65, 70, 72, 80, 75, 68, 66, 64], fare_ugx=5000.0)


def test_route_initialization_and_turnover(ntinda_route):
    """Verifies Route creation, daily turnover, cumulative revenue, and statistics."""
    assert ntinda_route.route_name == "Kampala-Ntinda"
    assert len(ntinda_route) == 10
    assert ntinda_route.daily_turnover()[0] == 35 * 2000.0  # 70,000 UGX
    assert ntinda_route.cumulative_revenue() == sum([35, 40, 42, 50, 55, 60, 48, 52, 47, 45]) * 2000.0

    stats = ntinda_route.passenger_statistics()
    assert stats["mean"] == float(np.mean([35, 40, 42, 50, 55, 60, 48, 52, 47, 45]))
    assert stats["variance"] > 0


def test_route_boundary_validation():
    """Boundary Case: Negative fare or empty passenger array raises ValueError."""
    with pytest.raises(ValueError):
        Route("BadFare", [50, 60], fare_ugx=-1000.0)
    with pytest.raises(ValueError):
        Route("EmptyPassengers", [], fare_ugx=2000.0)


def test_market_equilibrium_solution():
    """
    Verifies microeconomic market equilibrium:
        Qd = 120 - 0.02P
        Qs = 10 + 0.03P
    Equilibrium: 0.05P = 110 -> P* = 2,200 UGX, Q* = 76.
    """
    eq = solve_ntinda_market_equilibrium()
    assert math.isclose(eq["equilibrium_fare_p_star"], 2200.0, rel_tol=1e-5)
    assert math.isclose(eq["equilibrium_volume_q_star"], 76.0, rel_tol=1e-5)
    assert eq["is_below_equilibrium"] is True
    assert eq["capacity_shortage"] == 10.0


def test_forecasters_fit_and_predict(ntinda_route):
    """Verifies SMA, SES, and Linear Trend forecasters."""
    history = ntinda_route.passengers[:5]

    sma = MovingAverageForecaster(window=3)
    sma.fit(history)
    pred_sma = sma.predict(steps=2)
    # Expected: mean of [42, 50, 55] = 49.0
    assert math.isclose(pred_sma[0], 49.0, rel_tol=1e-5)

    ses = SimpleExponentialSmoothingForecaster(alpha=0.3)
    ses.fit(history)
    pred_ses = ses.predict(steps=2)
    assert len(pred_ses) == 2

    lin = LinearTrendForecaster()
    lin.fit(history)
    pred_lin = lin.predict(steps=2)
    assert len(pred_lin) == 2


def test_walk_forward_backtest(ntinda_route):
    """Verifies walk-forward validation over days 4 through 10."""
    models = [
        MovingAverageForecaster(window=3),
        SimpleExponentialSmoothingForecaster(alpha=0.3),
        LinearTrendForecaster(),
    ]
    res = walk_forward_backtest(ntinda_route.passengers, models, start_origin=3)
    assert len(res["test_days"]) == 7  # Days 4 through 10
    assert res["best_model_name"] in [m.name for m in models]
    assert res["best_model_mae"] > 0.0


def test_grid_search_ses_alpha(ntinda_route):
    """Verifies grid search for optimal alpha*."""
    best_a, best_mae, curve = grid_search_optimal_ses_alpha(
        ntinda_route.passengers,
        alpha_candidates=[0.1, 0.3, 0.5, 0.7, 0.9],
        start_origin=3,
    )
    assert 0.0 < best_a < 1.0
    assert best_mae > 0.0


def test_fleet_deployment_optimization():
    """
    Verifies fleet sizing with 8 trips/day, 14 seats, and 15% safety buffer.
    Daily capacity per vehicle: 8 * 14 = 112 passengers.
    For 200 passengers:
    Base = 200 / 112 = 1.7857 vehicles
    Buffered (+15%) = 1.7857 * 1.15 = 2.0536 vehicles
    Ceil = 3 vehicles.
    """
    deploy = calculate_fleet_deployment(
        projected_passengers=200.0,
        trips_per_day=8,
        vehicle_capacity=14,
        safety_margin_pct=0.15,
    )
    assert deploy["daily_capacity_per_vehicle"] == 112
    assert deploy["deployed_fleet_size"] == 3
    assert deploy["effective_total_capacity"] == 336
    assert deploy["spare_passenger_capacity"] == 136


def test_seasonal_naive_vs_sma_extension():
    """
    Extension Case: Verifies Seasonal-Naive outperforms standard SMA
    on a synthesized 60-day seasonal dataset.
    """
    series = generate_60day_seasonal_demand(base_demand=60.0, random_seed=42)
    assert len(series) == 60

    models = [
        MovingAverageForecaster(window=3, name="3-Day SMA"),
        SeasonalNaiveForecaster(seasonality=7, name="Seasonal-Naive (m=7)"),
    ]
    # Backtest over days 14 through 60
    res = walk_forward_backtest(series, models, start_origin=14)
    mae_sma = res["mae_by_model"]["3-Day SMA"]
    mae_snaive = res["mae_by_model"]["Seasonal-Naive (m=7)"]

    # Seasonal-Naive must outperform 3-Day SMA on weekly seasonal series
    assert mae_snaive < mae_sma
