"""
Model evaluation and backtesting framework.
Calculates MAE, RMSE, and MAPE metrics to assess out-of-sample forecast accuracy.
"""

from __future__ import annotations
from typing import Sequence, Dict, List, Any
import numpy as np

from .models import DistrictPopulation
from .forecasters import Forecaster


def calculate_mae(actual: Sequence[float] | np.ndarray, predicted: Sequence[float] | np.ndarray) -> float:
    """
    Computes Mean Absolute Error (MAE).
    MAE = (1 / m) * sum(|y_i - y_hat_i|)
    """
    y = np.asarray(actual, dtype=np.float64)
    y_hat = np.asarray(predicted, dtype=np.float64)
    if len(y) != len(y_hat):
        raise ValueError("Actual and predicted sequences must have identical lengths.")
    if len(y) == 0:
        raise ValueError("Cannot calculate MAE on empty arrays.")
    return float(np.mean(np.abs(y - y_hat)))


def calculate_rmse(actual: Sequence[float] | np.ndarray, predicted: Sequence[float] | np.ndarray) -> float:
    """
    Computes Root Mean Squared Error (RMSE).
    RMSE = sqrt((1 / m) * sum((y_i - y_hat_i)^2))
    """
    y = np.asarray(actual, dtype=np.float64)
    y_hat = np.asarray(predicted, dtype=np.float64)
    if len(y) != len(y_hat):
        raise ValueError("Actual and predicted sequences must have identical lengths.")
    if len(y) == 0:
        raise ValueError("Cannot calculate RMSE on empty arrays.")
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def calculate_mape(actual: Sequence[float] | np.ndarray, predicted: Sequence[float] | np.ndarray) -> float:
    """
    Computes Mean Absolute Percentage Error (MAPE) as a percentage.
    MAPE = (100% / m) * sum(|(y_i - y_hat_i) / y_i|)
    """
    y = np.asarray(actual, dtype=np.float64)
    y_hat = np.asarray(predicted, dtype=np.float64)
    if len(y) != len(y_hat):
        raise ValueError("Actual and predicted sequences must have identical lengths.")
    if len(y) == 0:
        raise ValueError("Cannot calculate MAPE on empty arrays.")
    if np.any(y == 0):
        raise ZeroDivisionError("Cannot calculate MAPE with zero actual values.")
    return float(np.mean(np.abs((y - y_hat) / y)) * 100.0)


def evaluate_forecast(actual: Sequence[float] | np.ndarray, predicted: Sequence[float] | np.ndarray) -> Dict[str, float]:
    """Computes MAE, RMSE, and MAPE in a unified dictionary."""
    return {
        "mae": calculate_mae(actual, predicted),
        "rmse": calculate_rmse(actual, predicted),
        "mape": calculate_mape(actual, predicted),
    }


def backtest_district(
    district: DistrictPopulation,
    models: Sequence[Forecaster],
    train_end_year: int = 2021,
) -> Dict[str, Any]:
    """
    Performs out-of-sample backtesting on a district population time series.
    Splits data into train (years <= train_end_year) and test (years > train_end_year).

    Args:
        district: DistrictPopulation dataset.
        models: Collection of Forecaster instances to evaluate.
        train_end_year: Threshold year defining the training partition boundary.

    Returns:
        Structured dictionary with test metrics, forecast trajectories, and top model selection.
    """
    train_mask = district.years <= train_end_year
    test_mask = district.years > train_end_year

    if np.sum(train_mask) < 2:
        raise ValueError(f"Training set has fewer than 2 observations before or in year {train_end_year}.")
    if np.sum(test_mask) < 1:
        raise ValueError(f"Testing set has no observations after year {train_end_year}.")

    train_years = district.years[train_mask]
    train_values = district.values[train_mask]
    test_years = district.years[test_mask]
    test_values = district.values[test_mask]

    results: List[Dict[str, Any]] = []

    for model in models:
        # Clone or fit model on training split
        model.fit(train_years, train_values)
        predictions = model.predict(test_years)
        fitted_train = model.predict(train_years)

        metrics = evaluate_forecast(test_values, predictions)
        results.append({
            "model_name": model.name,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mape": metrics["mape"],
            "test_predictions": predictions.tolist(),
            "fitted_train": fitted_train.tolist(),
            "model_instance": model,
        })

    # Sort models by MAE ascending
    results.sort(key=lambda r: (r["mae"], r["rmse"]))
    best_model_info = results[0]

    return {
        "district_name": district.district_name,
        "train_years": train_years.tolist(),
        "train_values": train_values.tolist(),
        "test_years": test_years.tolist(),
        "test_values": test_values.tolist(),
        "models_evaluation": results,
        "best_model_name": best_model_info["model_name"],
        "best_model_mae": best_model_info["mae"],
        "best_model_rmse": best_model_info["rmse"],
        "best_model_mape": best_model_info["mape"],
        "best_model_instance": best_model_info["model_instance"],
    }
