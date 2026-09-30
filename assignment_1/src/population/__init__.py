"""
UBOS Population Forecasting and Infrastructure Planning Package.
Advent 2026 OOP Assignment.
"""

from .models import DistrictPopulation
from .forecasters import (
    Forecaster,
    LinearTrendForecaster,
    ExponentialCAGRForecaster,
    FibonacciRatioForecaster,
)
from .stats import (
    compute_standard_statistics,
    compute_numpy_statistics,
    compare_statistics_and_numpy,
)
from .growth import (
    compute_yoy_growth,
    compute_cagr,
    analyze_district_growth,
    rank_districts_by_growth,
)
from .evaluation import (
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    evaluate_forecast,
    backtest_district,
)
from .bootstrap import bootstrap_prediction_intervals
from .planning import ClassroomPlanner

__all__ = [
    "DistrictPopulation",
    "Forecaster",
    "LinearTrendForecaster",
    "ExponentialCAGRForecaster",
    "FibonacciRatioForecaster",
    "compute_standard_statistics",
    "compute_numpy_statistics",
    "compare_statistics_and_numpy",
    "compute_yoy_growth",
    "compute_cagr",
    "analyze_district_growth",
    "rank_districts_by_growth",
    "calculate_mae",
    "calculate_rmse",
    "calculate_mape",
    "evaluate_forecast",
    "backtest_district",
    "bootstrap_prediction_intervals",
    "ClassroomPlanner",
]
