"""
Demographic growth dynamics analysis: Year-on-Year (YoY) growth and Compound Annual Growth Rate (CAGR).
"""

from __future__ import annotations
from typing import Sequence, Tuple, Optional, Dict, List, Any
import numpy as np

from .models import DistrictPopulation


def compute_yoy_growth(
    values: Sequence[float | int] | np.ndarray,
    years: Optional[Sequence[int] | np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Year-on-Year (YoY) percentage growth rates across consecutive periods.

    Formula:
        YoY_t = ((P_t - P_{t-1}) / P_{t-1}) * 100%

    Args:
        values: Sequence of population figures.
        years: Optional sequence of corresponding years.

    Returns:
        Tuple of (target_years, yoy_percentages). If years is None, target_years is indices 1..N-1.
    """
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) < 2:
        raise ValueError("At least 2 points required to calculate YoY growth.")

    if np.any(vals <= 0):
        raise ValueError("Population values must be strictly positive to calculate percentage growth.")

    yoy = ((vals[1:] - vals[:-1]) / vals[:-1]) * 100.0

    if years is not None:
        yrs = np.asarray(years, dtype=np.int64)
        if len(yrs) != len(vals):
            raise ValueError("years and values arrays must have matching dimensions.")
        target_years = yrs[1:]
    else:
        target_years = np.arange(1, len(vals))

    return target_years, yoy


def compute_cagr(start_value: float, end_value: float, periods: int) -> float:
    """
    Calculates the Compound Annual Growth Rate (CAGR).

    Formula:
        CAGR = (P_end / P_start) ** (1 / n) - 1

    Args:
        start_value: Initial population at beginning of period.
        end_value: Terminal population at end of period.
        periods: Number of compounding annual intervals (end_year - start_year).

    Returns:
        CAGR as a decimal (e.g., 0.054 for 5.4%).

    Raises:
        ValueError: If start_value or end_value <= 0, or periods <= 0.
    """
    if start_value <= 0 or end_value <= 0:
        raise ValueError("Population figures for CAGR must be strictly positive.")
    if periods <= 0:
        raise ValueError("Compounding periods n must be a strictly positive integer.")

    return float((end_value / start_value) ** (1.0 / periods) - 1.0)


def analyze_district_growth(district: DistrictPopulation) -> Dict[str, Any]:
    """
    Comprehensive growth profile for a DistrictPopulation object.

    Args:
        district: DistrictPopulation instance.

    Returns:
        Dictionary containing YoY time series, summary YoY statistics, and overall CAGR.
    """
    target_years, yoy_rates = compute_yoy_growth(district.values, district.years)
    n_years = len(district.years) - 1
    cagr_rate = compute_cagr(district.values[0], district.values[-1], n_years)

    return {
        "district_name": district.district_name,
        "start_year": int(district.years[0]),
        "end_year": int(district.years[-1]),
        "periods": n_years,
        "start_population_k": float(district.values[0]),
        "end_population_k": float(district.values[-1]),
        "cagr": float(cagr_rate),
        "cagr_percent": float(cagr_rate * 100.0),
        "yoy_years": target_years.tolist(),
        "yoy_rates_percent": yoy_rates.tolist(),
        "mean_yoy_percent": float(np.mean(yoy_rates)),
        "min_yoy_percent": float(np.min(yoy_rates)),
        "max_yoy_percent": float(np.max(yoy_rates)),
    }


def rank_districts_by_growth(districts: Sequence[DistrictPopulation]) -> List[Dict[str, Any]]:
    """
    Ranks a collection of districts by CAGR in descending order.

    Args:
        districts: Sequence of DistrictPopulation objects.

    Returns:
        List of growth summary dictionaries sorted from highest to lowest CAGR.
    """
    summaries = [analyze_district_growth(d) for d in districts]
    summaries.sort(key=lambda s: s["cagr"], reverse=True)
    return summaries
