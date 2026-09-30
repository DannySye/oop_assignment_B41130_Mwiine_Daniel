"""
Residual bootstrap resampling engine for constructing empirical prediction intervals.
Implements non-parametric residual bootstrapping (>= 1,000 iterations) to quantify
both parameter estimation variance and future innovation uncertainty.
"""

from __future__ import annotations
import copy
from typing import Sequence, Union, Optional, Dict, Any
import numpy as np

from .forecasters import Forecaster, LinearTrendForecaster, ExponentialCAGRForecaster, FibonacciRatioForecaster


def bootstrap_prediction_intervals(
    model: Forecaster,
    horizon: Union[int, Sequence[int], np.ndarray],
    n_bootstraps: int = 2000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Constructs non-parametric empirical prediction intervals via residual bootstrap resampling.

    Algorithm:
    1. Extract training residuals: e_t = y_t - y_hat_t.
    2. Mean-center residuals: e_tilde_t = e_t - mean(e_t).
    3. For b = 1 .. B (>= 1,000 iterations):
       a. Sample residuals with replacement: e*_t.
       b. Construct synthetic training data: y*_t = y_hat_t + e*_t.
       c. Refit an identical model instance on (years, y*_t).
       d. Generate forecast trajectory y_hat*_future.
       e. Add future innovation shock sampled from centered residuals.
    4. Compute empirical lower (e.g. 2.5%) and upper (e.g. 97.5%) quantiles.

    Args:
        model: Fitted Forecaster instance.
        horizon: Integer step count or sequence of future calendar years.
        n_bootstraps: Number of bootstrap iterations (default: 2,000 >= 1,000).
        confidence_level: Desired coverage probability (default: 0.95).
        random_seed: Random number generator seed for deterministic reproducibility.

    Returns:
        Dictionary containing target years, point forecasts, lower bounds, upper bounds,
        and all simulated bootstrap trajectories.
    """
    if not model.is_fitted:
        raise RuntimeError("Model must be fitted before running bootstrap prediction intervals.")

    if n_bootstraps < 100:
        raise ValueError(f"n_bootstraps should be at least 100, received {n_bootstraps}.")

    if not (0.5 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be between 0.5 and 1.0, received {confidence_level}.")

    rng = np.random.default_rng(random_seed)

    train_years = model.train_years
    train_values = model.train_values
    n_train = len(train_years)

    # 1. Base point predictions
    base_point_forecasts = model.predict(horizon)
    target_years = model._resolve_target_years(horizon)
    n_future = len(target_years)

    # 2. Residual calculation and centering
    in_sample_fitted = model.predict_in_sample()
    raw_residuals = train_values - in_sample_fitted
    centered_residuals = raw_residuals - np.mean(raw_residuals)

    # Instantiate fresh model clone helper
    def create_model_clone() -> Forecaster:
        if isinstance(model, LinearTrendForecaster):
            return LinearTrendForecaster(name=model.name)
        elif isinstance(model, ExponentialCAGRForecaster):
            return ExponentialCAGRForecaster(name=model.name)
        elif isinstance(model, FibonacciRatioForecaster):
            return FibonacciRatioForecaster(mode=model.mode, name=model.name)
        else:
            return copy.deepcopy(model)

    bootstrap_matrix = np.zeros((n_bootstraps, n_future), dtype=np.float64)

    for b in range(n_bootstraps):
        # Sample training residuals with replacement
        boot_train_res = rng.choice(centered_residuals, size=n_train, replace=True)
        synthetic_train_values = in_sample_fitted + boot_train_res

        # Ensure positive values
        synthetic_train_values = np.maximum(synthetic_train_values, 1.0)

        # Refit on synthetic series
        boot_model = create_model_clone()
        boot_model.fit(train_years, synthetic_train_values)

        # Future projection with future innovation shock
        future_trend = boot_model.predict(target_years)
        future_innovations = rng.choice(centered_residuals, size=n_future, replace=True)
        simulated_future_path = future_trend + future_innovations

        # Clamp at strictly positive
        bootstrap_matrix[b, :] = np.maximum(simulated_future_path, 1.0)

    # Compute percentiles
    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    lower_bounds = np.percentile(bootstrap_matrix, lower_pct, axis=0)
    upper_bounds = np.percentile(bootstrap_matrix, upper_pct, axis=0)
    median_forecasts = np.percentile(bootstrap_matrix, 50.0, axis=0)

    return {
        "model_name": model.name,
        "target_years": target_years.tolist(),
        "point_forecasts": base_point_forecasts.tolist(),
        "lower_bounds": lower_bounds.tolist(),
        "upper_bounds": upper_bounds.tolist(),
        "median_forecasts": median_forecasts.tolist(),
        "n_bootstraps": n_bootstraps,
        "confidence_level": confidence_level,
        "raw_residual_std": float(np.std(raw_residuals, ddof=1)),
        "bootstrap_sample_trajectories": bootstrap_matrix[:50, :].tolist(),  # First 50 for visualization sample
    }
