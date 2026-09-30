"""
Unit and integration test suite for Mini-Project 1: UBOS District Population Forecaster.
Covers domain models, statistical analysis, growth dynamics, forecasting models,
backtesting metrics, bootstrap resampling, and edge/boundary cases.
"""

import math
import numpy as np
import pytest

from assignment_1.src.population.models import DistrictPopulation
from assignment_1.src.population.stats import (
    compute_standard_statistics,
    compute_numpy_statistics,
    compare_statistics_and_numpy,
)
from assignment_1.src.population.growth import (
    compute_yoy_growth,
    compute_cagr,
    analyze_district_growth,
    rank_districts_by_growth,
)
from assignment_1.src.population.forecasters import (
    LinearTrendForecaster,
    ExponentialCAGRForecaster,
    FibonacciRatioForecaster,
)
from assignment_1.src.population.evaluation import (
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    evaluate_forecast,
    backtest_district,
)
from assignment_1.src.population.planning import ClassroomPlanner
from assignment_1.src.population.bootstrap import bootstrap_prediction_intervals


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def sample_years():
    return [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]


@pytest.fixture
def kampala_values():
    return [1200, 1250, 1300, 1350, 1420, 1500, 1580, 1650, 1720, 1800]


@pytest.fixture
def kampala_district(sample_years, kampala_values):
    return DistrictPopulation("Kampala", sample_years, kampala_values)


# ---------------------------------------------------------
# Domain Model Tests & Boundary Cases
# ---------------------------------------------------------

def test_district_population_initialization(kampala_district, sample_years, kampala_values):
    """Verifies standard initialization, attributes, length, and representations."""
    assert kampala_district.district_name == "Kampala"
    assert len(kampala_district) == 10
    assert np.array_equal(kampala_district.years, np.array(sample_years))
    assert np.array_equal(kampala_district.values, np.array(kampala_values, dtype=np.float64))
    assert "Kampala" in repr(kampala_district)
    assert "2015-2024" in repr(kampala_district)


def test_district_population_immutability(kampala_district):
    """Verifies that underlying arrays are protected against in-place mutation."""
    with pytest.raises(ValueError):
        kampala_district.values[0] = 9999.0  # writeable flag is False


def test_district_population_mismatched_dimensions(sample_years):
    """Boundary Case: Mismatched array lengths must raise ValueError."""
    bad_values = [100.0, 200.0]
    with pytest.raises(ValueError, match="Dimension mismatch"):
        DistrictPopulation("TestDistrict", sample_years, bad_values)


def test_district_population_negative_or_zero_values(sample_years):
    """Boundary Case: Non-positive population values must raise ValueError."""
    values_with_zero = [100, 110, 120, 0, 140, 150, 160, 170, 180, 190]
    values_with_neg = [100, 110, 120, -5, 140, 150, 160, 170, 180, 190]

    with pytest.raises(ValueError, match="strictly positive"):
        DistrictPopulation("ZeroDistrict", sample_years, values_with_zero)
    with pytest.raises(ValueError, match="strictly positive"):
        DistrictPopulation("NegDistrict", sample_years, values_with_neg)


def test_district_population_non_increasing_years():
    """Boundary Case: Years not in strictly ascending order must raise ValueError."""
    unordered_years = [2015, 2017, 2016, 2018]
    duplicate_years = [2015, 2016, 2016, 2018]
    values = [100, 110, 120, 130]

    with pytest.raises(ValueError, match="strictly ascending"):
        DistrictPopulation("Unordered", unordered_years, values)
    with pytest.raises(ValueError, match="strictly ascending"):
        DistrictPopulation("Duplicates", duplicate_years, values)


def test_district_population_too_few_observations():
    """Boundary Case: Empty or single observation collection must raise ValueError."""
    with pytest.raises(ValueError, match="At least 2 observations"):
        DistrictPopulation("TooFew", [2020], [500.0])
    with pytest.raises(ValueError, match="At least 2 observations"):
        DistrictPopulation("Empty", [], [])


def test_district_population_indexing_and_slicing(kampala_district):
    """Verifies item retrieval and subset slicing."""
    yr, val = kampala_district[0]
    assert yr == 2015
    assert val == 1200.0

    subset = kampala_district[0:5]
    assert isinstance(subset, DistrictPopulation)
    assert len(subset) == 5
    assert subset.years[-1] == 2019

    # Slicing fewer than 2 elements should raise ValueError
    with pytest.raises(ValueError, match="at least 2 observations"):
        _ = kampala_district[0:1]


def test_district_population_get_subset(kampala_district):
    """Verifies get_subset temporal filtering."""
    sub = kampala_district.get_subset(2018, 2021)
    assert len(sub) == 4
    assert sub.years[0] == 2018
    assert sub.years[-1] == 2021


# ---------------------------------------------------------
# Statistical Analysis & ddof Tests
# ---------------------------------------------------------

def test_statistics_vs_numpy_ddof_divergence(kampala_values):
    """
    Verifies that Python standard statistics.variance (ddof=1) differs from np.var (ddof=0),
    and strictly equals np.var (ddof=1) due to Bessel's correction.
    """
    comparison = compare_statistics_and_numpy(kampala_values)
    n = len(kampala_values)

    var_stats = comparison["standard_statistics"]["variance"]
    var_np0 = comparison["numpy_ddof_0"]["variance"]
    var_np1 = comparison["numpy_ddof_1"]["variance"]

    # Must diverge with default numpy (ddof=0)
    assert var_stats != var_np0
    # Must match with ddof=1
    assert math.isclose(var_stats, var_np1, rel_tol=1e-9)

    # Ratio must equal N / (N - 1)
    expected_ratio = n / (n - 1)
    assert math.isclose(var_stats / var_np0, expected_ratio, rel_tol=1e-9)
    assert comparison["matches_with_ddof1"] is True


# ---------------------------------------------------------
# Growth Dynamics Tests
# ---------------------------------------------------------

def test_compute_yoy_growth(kampala_values, sample_years):
    """Verifies YoY calculation."""
    yrs, yoy = compute_yoy_growth(kampala_values, sample_years)
    assert len(yoy) == len(kampala_values) - 1
    # First YoY: (1250 - 1200) / 1200 * 100 = 4.1667%
    expected_first = (1250.0 - 1200.0) / 1200.0 * 100.0
    assert math.isclose(yoy[0], expected_first, rel_tol=1e-5)


def test_compute_cagr_formula():
    """Verifies analytical accuracy of CAGR calculation."""
    start_val = 100.0
    end_val = 200.0
    periods = 10
    expected_cagr = (200.0 / 100.0) ** (1.0 / 10.0) - 1.0

    calc_cagr = compute_cagr(start_val, end_val, periods)
    assert math.isclose(calc_cagr, expected_cagr, rel_tol=1e-9)


def test_cagr_boundary_cases():
    """Boundary Case: Non-positive values or zero periods must raise ValueError."""
    with pytest.raises(ValueError):
        compute_cagr(-10.0, 100.0, 5)
    with pytest.raises(ValueError):
        compute_cagr(100.0, 0.0, 5)
    with pytest.raises(ValueError):
        compute_cagr(100.0, 200.0, 0)


def test_rank_districts_by_growth(kampala_district):
    """Verifies ranking logic across multiple districts."""
    wakiso = DistrictPopulation("Wakiso", [2015, 2024], [950, 1670])
    rankings = rank_districts_by_growth([kampala_district, wakiso])
    # Wakiso CAGR (~6.47%) is higher than Kampala (~4.61%)
    assert rankings[0]["district_name"] == "Wakiso"
    assert rankings[1]["district_name"] == "Kampala"


# ---------------------------------------------------------
# Forecasting Architecture Tests
# ---------------------------------------------------------

def test_unfitted_model_raises_runtime_error():
    """Boundary Case: Calling predict or residuals on an unfitted model raises RuntimeError."""
    linear = LinearTrendForecaster()
    with pytest.raises(RuntimeError, match="must be fitted"):
        linear.predict(5)
    with pytest.raises(RuntimeError, match="must be fitted"):
        linear.residuals()


def test_linear_trend_forecaster(sample_years, kampala_values):
    """Verifies LinearTrendForecaster fits degree-1 polynomial accurately."""
    train_years = sample_years[:7]
    train_vals = kampala_values[:7]

    model = LinearTrendForecaster()
    model.fit(train_years, train_vals)

    assert model.is_fitted
    assert model.slope is not None and model.slope > 0
    assert model.r_squared is not None and model.r_squared > 0.95

    # Predict 3 steps ahead
    preds = model.predict(3)
    assert len(preds) == 3
    # Check monotonic increase
    assert preds[0] < preds[1] < preds[2]


def test_exponential_cagr_forecaster(sample_years, kampala_values):
    """Verifies ExponentialCAGRForecaster compounding fit."""
    train_years = sample_years[:7]
    train_vals = kampala_values[:7]

    model = ExponentialCAGRForecaster()
    model.fit(train_years, train_vals)

    assert model.is_fitted
    assert model.cagr is not None and model.cagr > 0
    preds = model.predict([2022, 2023, 2024])
    assert len(preds) == 3
    assert preds[0] < preds[1] < preds[2]


def test_fibonacci_ratio_forecaster():
    """Verifies Fibonacci ratio generator and prediction modes."""
    # Test ratio convergence
    r1 = FibonacciRatioForecaster.get_fibonacci_ratio(1)  # 1/1 = 1.0
    r2 = FibonacciRatioForecaster.get_fibonacci_ratio(2)  # 2/1 = 2.0
    r3 = FibonacciRatioForecaster.get_fibonacci_ratio(3)  # 3/2 = 1.5
    assert r1 == 1.0
    assert r2 == 2.0
    assert r3 == 1.5

    # Direct mode test
    model_direct = FibonacciRatioForecaster(mode="direct")
    model_direct.fit([2020, 2021], [100.0, 110.0])
    preds_direct = model_direct.predict(2)
    assert len(preds_direct) == 2
    # Step 1 with k=3 ratio=1.5 => 110 * 1.5 = 165
    assert math.isclose(preds_direct[0], 165.0, rel_tol=1e-5)

    # Calibrated mode test
    model_cal = FibonacciRatioForecaster(mode="calibrated")
    model_cal.fit([2020, 2021], [100.0, 110.0])
    preds_cal = model_cal.predict(2)
    assert len(preds_cal) == 2
    assert preds_cal[0] < preds_direct[0]


# ---------------------------------------------------------
# Backtesting & Evaluation Tests
# ---------------------------------------------------------

def test_evaluation_metrics():
    """Verifies calculation of MAE, RMSE, and MAPE."""
    actual = [100.0, 200.0]
    predicted = [110.0, 190.0]

    # |100 - 110| = 10, |200 - 190| = 10 => MAE = 10.0
    assert calculate_mae(actual, predicted) == 10.0
    # sqrt((10^2 + 10^2) / 2) = 10.0 => RMSE = 10.0
    assert calculate_rmse(actual, predicted) == 10.0
    # (|10/100| + |10/200|) / 2 * 100 = (0.10 + 0.05) / 2 * 100 = 7.5%
    assert math.isclose(calculate_mape(actual, predicted), 7.5, rel_tol=1e-9)


def test_mape_zero_division():
    """Boundary Case: MAPE with zero actual must raise ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError):
        calculate_mape([0.0, 100.0], [10.0, 105.0])


def test_backtest_district_pipeline(kampala_district):
    """Verifies full backtesting workflow on train 2015-2021 and test 2022-2024."""
    models = [
        LinearTrendForecaster(),
        ExponentialCAGRForecaster(),
        FibonacciRatioForecaster(mode="calibrated"),
    ]
    bt = backtest_district(kampala_district, models, train_end_year=2021)

    assert bt["district_name"] == "Kampala"
    assert len(bt["train_years"]) == 7
    assert len(bt["test_years"]) == 3
    assert len(bt["models_evaluation"]) == 3
    assert bt["best_model_name"] in [m.name for m in models]
    assert bt["best_model_mae"] >= 0.0


# ---------------------------------------------------------
# Classroom Infrastructure Planning Tests
# ---------------------------------------------------------

def test_classroom_planner_calculations():
    """Verifies standard classroom capacity calculation with ceiling rounding."""
    planner = ClassroomPlanner(school_age_ratio=0.18, classroom_capacity=53)

    # 100k population increase => 100,000 * 0.18 = 18,000 pupils
    # 18,000 / 53 = 339.6226 => ceil is 340 classrooms
    res = planner.calculate_requirements(
        district_name="TestDistrict",
        base_population_k=1000.0,
        projected_population_k=1100.0,
    )
    assert res["net_population_growth"] == 100000.0
    assert res["additional_pupils"] == 18000.0
    assert res["net_classrooms_needed"] == 340


def test_classroom_planner_negative_growth():
    """Boundary Case: If population does not increase, required classrooms should be 0."""
    planner = ClassroomPlanner()
    res = planner.calculate_requirements("Declining", 500.0, 480.0)
    assert res["net_classrooms_needed"] == 0


# ---------------------------------------------------------
# Bootstrap Prediction Intervals Tests (Extension)
# ---------------------------------------------------------

def test_bootstrap_prediction_intervals(kampala_district):
    """Verifies residual bootstrap produces valid 95% prediction intervals with >= 1,000 iterations."""
    linear = LinearTrendForecaster()
    linear.fit(kampala_district.years, kampala_district.values)

    intervals = bootstrap_prediction_intervals(
        linear,
        horizon=5,
        n_bootstraps=1000,
        confidence_level=0.95,
        random_seed=42,
    )

    assert len(intervals["target_years"]) == 5
    assert len(intervals["lower_bounds"]) == 5
    assert len(intervals["upper_bounds"]) == 5

    # Check bounds order: lower <= median <= upper
    for low, med, high in zip(intervals["lower_bounds"], intervals["median_forecasts"], intervals["upper_bounds"]):
        assert low <= med <= high
