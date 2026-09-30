"""
Statistical analysis tools for district population metrics.
Contrasts Python's standard `statistics` module with NumPy's vectorized calculations,
focusing on degrees of freedom (ddof) and Bessel's correction.
"""

from __future__ import annotations
import statistics
from typing import Dict, Any, Sequence
import numpy as np


def compute_standard_statistics(values: Sequence[float | int]) -> Dict[str, float]:
    """
    Computes summary statistics using Python's standard library `statistics` module.

    Note: `statistics.variance` and `statistics.stdev` compute sample statistics
    with N - 1 degrees of freedom (Bessel's correction).

    Args:
        values: Sequence of numeric population values.

    Returns:
        Dict containing mean, median, variance, and standard deviation.
    """
    val_list = [float(v) for v in values]
    if len(val_list) < 2:
        raise ValueError("At least 2 data points required to calculate variance and standard deviation.")

    return {
        "mean": float(statistics.mean(val_list)),
        "median": float(statistics.median(val_list)),
        "variance": float(statistics.variance(val_list)),
        "stdev": float(statistics.stdev(val_list)),
    }


def compute_numpy_statistics(values: Sequence[float | int] | np.ndarray, ddof: int = 0) -> Dict[str, float]:
    """
    Computes summary statistics using NumPy.

    Args:
        values: Sequence or array of numeric population values.
        ddof: Delta Degrees of Freedom. The divisor used in calculations is N - ddof.
              Default is 0 (population variance), setting ddof=1 yields sample variance.

    Returns:
        Dict containing mean, median, variance, and standard deviation.
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2 and ddof > 0:
        raise ValueError("At least 2 data points required when ddof > 0.")

    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "variance": float(np.var(arr, ddof=ddof)),
        "stdev": float(np.std(arr, ddof=ddof)),
        "ddof": int(ddof),
    }


def compare_statistics_and_numpy(values: Sequence[float | int] | np.ndarray) -> Dict[str, Any]:
    """
    Compares outputs between `statistics` module and `numpy` across ddof=0 and ddof=1.
    Provides detailed theoretical explanation for the divergence.

    Args:
        values: Sequence or array of numeric observations.

    Returns:
        Structured dictionary comparing the statistical figures and explaining ddof divergence.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)

    stats_res = compute_standard_statistics(arr)
    np_ddof0 = compute_numpy_statistics(arr, ddof=0)
    np_ddof1 = compute_numpy_statistics(arr, ddof=1)

    var_stats = stats_res["variance"]
    var_np_0 = np_ddof0["variance"]
    var_np_1 = np_ddof1["variance"]

    theoretical_ratio = n / (n - 1)
    empirical_ratio = var_stats / var_np_0

    explanation = (
        f"DIVERGENCE ANALYSIS (N = {n}):\n"
        f"1. `statistics.variance` computes the unbiased SAMPLE variance by dividing the sum of "
        f"squared deviations by (N - 1) = {n - 1} (Bessel's correction).\n"
        f"2. `np.var(..., ddof=0)` (NumPy default) computes the POPULATION / maximum likelihood variance "
        f"by dividing by N = {n}.\n"
        f"3. The mathematical ratio between sample variance and default numpy variance is N / (N - 1) "
        f"= {n} / {n - 1} = {theoretical_ratio:.4f}. Empirically, {var_stats:.4f} / {var_np_0:.4f} = {empirical_ratio:.4f}.\n"
        f"4. When `np.var(..., ddof=1)` is supplied, NumPy divides by N - 1 = {n - 1}, yielding "
        f"{var_np_1:.4f}, which matches `statistics.variance` ({var_stats:.4f}) identically."
    )

    return {
        "n_samples": n,
        "standard_statistics": stats_res,
        "numpy_ddof_0": np_ddof0,
        "numpy_ddof_1": np_ddof1,
        "variance_difference_default": float(var_stats - var_np_0),
        "variance_difference_ddof1": float(abs(var_stats - var_np_1)),
        "theoretical_ratio": theoretical_ratio,
        "empirical_ratio": empirical_ratio,
        "matches_with_ddof1": bool(np.isclose(var_stats, var_np_1)),
        "explanation": explanation,
    }
