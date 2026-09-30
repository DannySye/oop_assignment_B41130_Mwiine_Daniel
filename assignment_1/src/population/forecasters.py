"""
Forecasting architecture conforming to Object-Oriented Programming (OOP) principles.
Defines abstract base class `Forecaster` and three concrete subclasses:
- LinearTrendForecaster: Degree-1 polynomial regression.
- ExponentialCAGRForecaster: Compounding exponential growth extrapolation.
- FibonacciRatioForecaster: Successive Fibonacci ratio scaling.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, Union, Optional
import numpy as np


class Forecaster(ABC):
    """
    Abstract Base Class for time series population forecasting.
    Enforces a strict fit-predict lifecycle.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        """
        Initializes the Forecaster base.

        Args:
            name: Optional descriptive label for the forecaster.
        """
        self._name = name or self.__class__.__name__
        self._train_years: Optional[np.ndarray] = None
        self._train_values: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    @property
    def name(self) -> str:
        """Human-readable identifier of the forecasting model."""
        return self._name

    @property
    def is_fitted(self) -> bool:
        """Boolean flag indicating whether the model has been fitted."""
        return self._is_fitted

    @property
    def train_years(self) -> np.ndarray:
        """Returns the training years array."""
        self._check_is_fitted()
        return self._train_years  # type: ignore

    @property
    def train_values(self) -> np.ndarray:
        """Returns the training target values array."""
        self._check_is_fitted()
        return self._train_values  # type: ignore

    def _check_is_fitted(self) -> None:
        """Raises RuntimeError if fit() has not been called."""
        if not self._is_fitted or self._train_years is None or self._train_values is None:
            raise RuntimeError(f"Model '{self._name}' must be fitted before calling this method.")

    def _validate_fit_inputs(
        self,
        years: Sequence[int] | np.ndarray,
        values: Sequence[float | int] | np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validates training inputs.

        Returns:
            Tuple of validated (years_arr, values_arr).
        """
        y_arr = np.asarray(years, dtype=np.int64)
        v_arr = np.asarray(values, dtype=np.float64)

        if y_arr.ndim != 1 or v_arr.ndim != 1:
            raise ValueError("Training years and values must be 1-dimensional arrays.")
        if len(y_arr) < 2:
            raise ValueError(f"At least 2 training observations required, received {len(y_arr)}.")
        if len(y_arr) != len(v_arr):
            raise ValueError(f"Length mismatch: years ({len(y_arr)}) vs values ({len(v_arr)}).")
        if np.any(v_arr <= 0):
            raise ValueError("Population figures must be strictly positive (> 0).")
        if np.any(np.diff(y_arr) <= 0):
            raise ValueError("Training years must be strictly increasing.")

        return y_arr, v_arr

    def _resolve_target_years(
        self,
        horizon: Union[int, Sequence[int], np.ndarray],
    ) -> np.ndarray:
        """
        Resolves horizon into an array of integer target calendar years.

        Args:
            horizon: Number of future periods (int >= 1) OR an explicit array of target years.

        Returns:
            1D array of target calendar years.
        """
        self._check_is_fitted()
        last_train_year = int(self._train_years[-1])  # type: ignore

        if isinstance(horizon, (int, np.integer)):
            if horizon < 1:
                raise ValueError(f"Horizon step count must be at least 1, received {horizon}.")
            return np.arange(last_train_year + 1, last_train_year + horizon + 1, dtype=np.int64)
        else:
            target_years = np.asarray(horizon, dtype=np.int64)
            if target_years.ndim != 1 or len(target_years) == 0:
                raise ValueError("Target horizon years must be a non-empty 1D sequence.")
            return target_years

    @abstractmethod
    def fit(
        self,
        years: Sequence[int] | np.ndarray,
        values: Sequence[float | int] | np.ndarray,
    ) -> "Forecaster":
        """
        Fits the forecasting model on historical data.

        Args:
            years: Calendar years of training observations.
            values: Historical population values.

        Returns:
            self: The fitted forecaster instance.
        """
        pass

    @abstractmethod
    def predict(
        self,
        horizon: Union[int, Sequence[int], np.ndarray],
    ) -> np.ndarray:
        """
        Generates out-of-sample population predictions.

        Args:
            horizon: Positive integer step count OR explicit sequence of target years.

        Returns:
            1D NumPy array of forecasted population values.
        """
        pass

    def predict_in_sample(self) -> np.ndarray:
        """
        Generates in-sample predictions for the training years.

        Returns:
            1D array of fitted values.
        """
        self._check_is_fitted()
        return self.predict(self._train_years)  # type: ignore

    def residuals(self) -> np.ndarray:
        """
        Computes training residuals (actual - fitted).

        Returns:
            1D array of residuals.
        """
        self._check_is_fitted()
        fitted = self.predict_in_sample()
        return self._train_values - fitted  # type: ignore

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return f"{self.__class__.__name__}(name='{self._name}', status='{status}')"


class LinearTrendForecaster(Forecaster):
    """
    Forecaster using first-degree polynomial regression (Ordinary Least Squares):
        y(t) = slope * t + intercept
    """

    def __init__(self, name: str = "Linear Trend Forecaster") -> None:
        super().__init__(name=name)
        self.slope: Optional[float] = None
        self.intercept: Optional[float] = None
        self.r_squared: Optional[float] = None

    def fit(
        self,
        years: Sequence[int] | np.ndarray,
        values: Sequence[float | int] | np.ndarray,
    ) -> "LinearTrendForecaster":
        y_arr, v_arr = self._validate_fit_inputs(years, values)
        self._train_years = np.copy(y_arr)
        self._train_values = np.copy(v_arr)

        # Fit degree 1 polynomial
        coeffs = np.polyfit(y_arr, v_arr, deg=1)
        self.slope = float(coeffs[0])
        self.intercept = float(coeffs[1])

        # Compute coefficient of determination (R^2)
        fitted = self.slope * y_arr + self.intercept
        ss_res = np.sum((v_arr - fitted) ** 2)
        ss_tot = np.sum((v_arr - np.mean(v_arr)) ** 2)
        self.r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: Union[int, Sequence[int], np.ndarray],
    ) -> np.ndarray:
        target_years = self._resolve_target_years(horizon)
        if self.slope is None or self.intercept is None:
            raise RuntimeError("Model parameters missing.")
        predictions = self.slope * target_years + self.intercept
        return predictions


class ExponentialCAGRForecaster(Forecaster):
    """
    Forecaster modeling exponential compounding growth:
        P(t) = exp(alpha + beta * (t - t0))
    Fitted via log-linear ordinary least squares, guaranteeing optimal geometric fit
    across the entire training window.
    """

    def __init__(self, name: str = "Exponential / CAGR Forecaster") -> None:
        super().__init__(name=name)
        self.base_year: Optional[int] = None
        self.alpha: Optional[float] = None
        self.beta: Optional[float] = None
        self.cagr: Optional[float] = None

    def fit(
        self,
        years: Sequence[int] | np.ndarray,
        values: Sequence[float | int] | np.ndarray,
    ) -> "ExponentialCAGRForecaster":
        y_arr, v_arr = self._validate_fit_inputs(years, values)
        self._train_years = np.copy(y_arr)
        self._train_values = np.copy(v_arr)
        self.base_year = int(y_arr[0])

        # Centered time variable to avoid floating precision issues with large calendar years
        t_shifted = y_arr - self.base_year
        log_v = np.log(v_arr)

        # Log-linear fit: ln(y) = beta * t_shifted + alpha
        coeffs = np.polyfit(t_shifted, log_v, deg=1)
        self.beta = float(coeffs[0])
        self.alpha = float(coeffs[1])
        self.cagr = float(np.exp(self.beta) - 1.0)

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: Union[int, Sequence[int], np.ndarray],
    ) -> np.ndarray:
        target_years = self._resolve_target_years(horizon)
        if self.alpha is None or self.beta is None or self.base_year is None:
            raise RuntimeError("Model parameters missing.")

        t_shifted = target_years - self.base_year
        predictions = np.exp(self.alpha + self.beta * t_shifted)
        return predictions


class FibonacciRatioForecaster(Forecaster):
    """
    Forecaster scaling previous periods by successive Fibonacci ratios (F_{k+1} / F_k).

    As k -> inf, the ratio converges to the Golden Ratio:
        phi = (1 + sqrt(5)) / 2 approx 1.6180339887...

    Supported modes:
    - 'direct': Directly multiplies previous population by (F_{k+1} / F_k).
                This reflects the literal wording of naive growth models.
    - 'calibrated': Scales the empirical historical growth increment by the Fibonacci ratio
                    normalized against phi, maintaining realistic annual demographic scales.
    """

    def __init__(
        self,
        mode: str = "direct",
        name: str = "Fibonacci-Ratio Forecaster",
    ) -> None:
        super().__init__(name=name)
        if mode not in ("direct", "calibrated"):
            raise ValueError(f"mode must be 'direct' or 'calibrated', received '{mode}'.")
        self.mode = mode
        self._baseline_growth_rate: Optional[float] = None
        self._fib_cache = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]

    @staticmethod
    def get_fibonacci_ratio(k: int) -> float:
        """
        Returns F_{k+1} / F_k for index k >= 1.
        Uses recurrence for arbitrary k.
        """
        if k < 1:
            raise ValueError("Fibonacci index k must be >= 1.")
        # Compute Fibonacci numbers up to k + 1
        a, b = 1, 1
        for _ in range(k - 1):
            a, b = b, a + b
        return float(b / a)

    def fit(
        self,
        years: Sequence[int] | np.ndarray,
        values: Sequence[float | int] | np.ndarray,
    ) -> "FibonacciRatioForecaster":
        y_arr, v_arr = self._validate_fit_inputs(years, values)
        self._train_years = np.copy(y_arr)
        self._train_values = np.copy(v_arr)

        # Calculate average historical growth rate from training series for calibrated mode
        n_periods = len(y_arr) - 1
        self._baseline_growth_rate = float((v_arr[-1] / v_arr[0]) ** (1.0 / n_periods) - 1.0)

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: Union[int, Sequence[int], np.ndarray],
    ) -> np.ndarray:
        target_years = self._resolve_target_years(horizon)
        self._check_is_fitted()

        last_train_year = int(self._train_years[-1])  # type: ignore
        last_train_val = float(self._train_values[-1])  # type: ignore
        phi = (1.0 + np.sqrt(5.0)) / 2.0

        predictions = []
        current_val = last_train_val

        for i, yr in enumerate(target_years):
            step = yr - last_train_year
            if step <= 0:
                # In-sample evaluation: back-project from nearest observation
                # In-sample interpolation using training baseline
                idx = np.where(self._train_years == yr)[0]
                if len(idx) > 0:
                    predictions.append(float(self._train_values[idx[0]]))
                else:
                    predictions.append(current_val)
                continue

            k = max(1, step + 2)  # Starting at F_4 / F_3 = 1.5 for step 1
            ratio = self.get_fibonacci_ratio(k)

            if self.mode == "direct":
                # Literal interpretation: P_{t+1} = P_t * (F_{k+1} / F_k)
                current_val = current_val * ratio
            else:
                # Calibrated mode: P_{t+1} = P_t * (1 + g_base * (ratio / phi))
                g_eff = (self._baseline_growth_rate or 0.04) * (ratio / phi)
                current_val = current_val * (1.0 + g_eff)

            predictions.append(float(current_val))

        return np.asarray(predictions, dtype=np.float64)
