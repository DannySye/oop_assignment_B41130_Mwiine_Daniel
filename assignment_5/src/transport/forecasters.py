"""
Forecasting hierarchy and walk-forward validation for commuter transit demand.
Implements Moving Average (SMA), Simple Exponential Smoothing (SES), Linear Trend,
and Seasonal-Naive predictors.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, List, Dict, Any, Optional, Tuple
import numpy as np


class TransportForecaster(ABC):
    """Abstract base class for commuter time series forecasting."""

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__
        self._history: Optional[np.ndarray] = None

    @abstractmethod
    def fit(self, history: Sequence[float | int] | np.ndarray) -> "TransportForecaster":
        """Ingests historical demand time series."""
        pass

    @abstractmethod
    def predict(self, steps: int = 1) -> np.ndarray:
        """Projects demand forward by `steps` periods."""
        pass

    def __repr__(self) -> str:
        return f"{self.name}()"


class MovingAverageForecaster(TransportForecaster):
    """3-Day Simple Moving Average (SMA) Forecaster."""

    def __init__(self, window: int = 3, name: Optional[str] = None) -> None:
        super().__init__(name=name or f"{window}-Day Moving Average")
        if window < 1:
            raise ValueError("window must be >= 1.")
        self.window = int(window)

    def fit(self, history: Sequence[float | int] | np.ndarray) -> "MovingAverageForecaster":
        arr = np.asarray(history, dtype=np.float64)
        if len(arr) < self.window:
            raise ValueError(f"History length ({len(arr)}) must be >= window ({self.window}).")
        self._history = np.copy(arr)
        return self

    def predict(self, steps: int = 1) -> np.ndarray:
        if self._history is None:
            raise RuntimeError("Model must be fitted before predict().")
        # Moving average of the last `window` values
        recent_window = self._history[-self.window:]
        forecast_val = float(np.mean(recent_window))
        return np.full(steps, forecast_val, dtype=np.float64)


class SimpleExponentialSmoothingForecaster(TransportForecaster):
    """
    Simple Exponential Smoothing (SES) Forecaster.
    Recurrence:
        y_hat_{t+1} = alpha * y_t + (1 - alpha) * y_hat_t
    """

    def __init__(self, alpha: float = 0.3, name: Optional[str] = None) -> None:
        super().__init__(name=name or f"SES (alpha={alpha:.2f})")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}.")
        self.alpha = float(alpha)
        self.level: Optional[float] = None

    def fit(self, history: Sequence[float | int] | np.ndarray) -> "SimpleExponentialSmoothingForecaster":
        arr = np.asarray(history, dtype=np.float64)
        if len(arr) < 2:
            raise ValueError("At least 2 observations required for SES.")
        self._history = np.copy(arr)

        # Initialize level at first observation
        smoothed_level = arr[0]
        for t in range(1, len(arr)):
            smoothed_level = self.alpha * arr[t] + (1.0 - self.alpha) * smoothed_level

        self.level = float(smoothed_level)
        return self

    def predict(self, steps: int = 1) -> np.ndarray:
        if self.level is None:
            raise RuntimeError("Model must be fitted before predict().")
        return np.full(steps, self.level, dtype=np.float64)


class LinearTrendForecaster(TransportForecaster):
    """Linear trend extrapolation via first-degree polynomial regression."""

    def __init__(self, name: str = "Linear Trend Extrapolation") -> None:
        super().__init__(name=name)
        self.slope: Optional[float] = None
        self.intercept: Optional[float] = None

    def fit(self, history: Sequence[float | int] | np.ndarray) -> "LinearTrendForecaster":
        arr = np.asarray(history, dtype=np.float64)
        if len(arr) < 2:
            raise ValueError("At least 2 points required for linear trend.")
        self._history = np.copy(arr)
        t = np.arange(1, len(arr) + 1, dtype=np.float64)
        coeffs = np.polyfit(t, arr, deg=1)
        self.slope = float(coeffs[0])
        self.intercept = float(coeffs[1])
        return self

    def predict(self, steps: int = 1) -> np.ndarray:
        if self.slope is None or self.intercept is None or self._history is None:
            raise RuntimeError("Model must be fitted before predict().")
        n = len(self._history)
        target_t = np.arange(n + 1, n + steps + 1, dtype=np.float64)
        return self.slope * target_t + self.intercept


class SeasonalNaiveForecaster(TransportForecaster):
    """Seasonal-Naive benchmark predicting y_{t+h} = y_{t+h - m}."""

    def __init__(self, seasonality: int = 7, name: Optional[str] = None) -> None:
        super().__init__(name=name or f"Seasonal-Naive (m={seasonality})")
        self.seasonality = int(seasonality)

    def fit(self, history: Sequence[float | int] | np.ndarray) -> "SeasonalNaiveForecaster":
        arr = np.asarray(history, dtype=np.float64)
        if len(arr) < self.seasonality:
            raise ValueError(f"History must contain at least {self.seasonality} points for seasonal naive.")
        self._history = np.copy(arr)
        return self

    def predict(self, steps: int = 1) -> np.ndarray:
        if self._history is None:
            raise RuntimeError("Model must be fitted before predict().")
        m = self.seasonality
        preds = []
        for s in range(1, steps + 1):
            lookback_idx = -(m - ((s - 1) % m))
            preds.append(float(self._history[lookback_idx]))
        return np.array(preds, dtype=np.float64)


def walk_forward_backtest(
    passengers: Sequence[float] | np.ndarray,
    models: Sequence[TransportForecaster],
    start_origin: int = 3,  # 0-indexed: index 3 is day 4
) -> Dict[str, Any]:
    """
    Performs rolling-origin walk-forward backtesting over days 4 through 10.
    At each origin t, model is trained on history[:t] and evaluates 1-step ahead prediction on history[t].
    Computes Mean Absolute Error (MAE).
    """
    y = np.asarray(passengers, dtype=np.float64)
    n = len(y)

    model_errors: Dict[str, List[float]] = {m.name: [] for m in models}
    model_predictions: Dict[str, List[float]] = {m.name: [] for m in models}
    test_days = list(range(start_origin + 1, n + 1))

    for t in range(start_origin, n):
        history_t = y[:t]
        actual_val = y[t]

        for m in models:
            m.fit(history_t)
            pred_val = float(m.predict(steps=1)[0])
            model_predictions[m.name].append(pred_val)
            model_errors[m.name].append(abs(actual_val - pred_val))

    mae_results = {name: float(np.mean(errs)) for name, errs in model_errors.items()}

    # Identify best model
    best_model_name = min(mae_results, key=mae_results.get)

    return {
        "test_days": test_days,
        "actual_values": y[start_origin:].tolist(),
        "mae_by_model": mae_results,
        "best_model_name": best_model_name,
        "best_model_mae": mae_results[best_model_name],
        "predictions_by_model": model_predictions,
    }


def grid_search_optimal_ses_alpha(
    passengers: Sequence[float] | np.ndarray,
    alpha_candidates: Optional[Sequence[float]] = None,
    start_origin: int = 3,
) -> Tuple[float, float, Dict[float, float]]:
    """
    Grid searches for optimal smoothing factor alpha* in (0, 1) minimizing rolling-origin MAE.
    """
    if alpha_candidates is None:
        alpha_candidates = np.linspace(0.05, 0.95, 19)

    mae_curve: Dict[float, float] = {}
    for a in alpha_candidates:
        model = SimpleExponentialSmoothingForecaster(alpha=float(a))
        res = walk_forward_backtest(passengers, [model], start_origin=start_origin)
        mae_curve[float(a)] = res["best_model_mae"]

    best_alpha = min(mae_curve, key=mae_curve.get)
    best_mae = mae_curve[best_alpha]

    return best_alpha, best_mae, mae_curve
