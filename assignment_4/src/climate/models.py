"""
Domain models for regional rainfall profiles and crop agronomic suitability rules.
Calibrated for Ugandan agro-ecological zones (UNMA / NARO / FAO guidelines).
"""

from __future__ import annotations
from typing import Sequence, List, Dict, Any, Tuple
import numpy as np


MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]


class Region:
    """
    Encapsulates 12-month precipitation averages for a geographic region.
    
    Attributes:
        name (str): Name of the region or district.
        monthly_rainfall (np.ndarray): 12-element 1D array of monthly rainfall (mm).
    """

    def __init__(self, name: str, monthly_rainfall: Sequence[float | int] | np.ndarray) -> None:
        if not isinstance(name, str) or not name.strip():
            raise TypeError("Region name must be a non-empty string.")

        arr = np.asarray(monthly_rainfall, dtype=np.float64)
        if arr.ndim != 1 or len(arr) != 12:
            raise ValueError(f"monthly_rainfall must be a 1D sequence of exactly 12 months, got shape {arr.shape}.")
        if np.any(arr < 0):
            raise ValueError("Precipitation values cannot be negative.")

        self.name = name.strip()
        self._rainfall = np.copy(arr)
        self._rainfall.flags.writeable = False

    @property
    def monthly_rainfall(self) -> np.ndarray:
        """Returns 12-month rainfall array."""
        return self._rainfall

    def annual_cumulative(self) -> float:
        """Returns total annual cumulative rainfall in mm."""
        return float(np.sum(self._rainfall))

    def monthly_mean(self) -> float:
        """Returns average monthly rainfall in mm."""
        return float(np.mean(self._rainfall))

    def wettest_month(self) -> Tuple[str, float]:
        """Returns (month_name, rainfall_mm) for the month with peak precipitation."""
        idx = int(np.argmax(self._rainfall))
        return MONTH_NAMES[idx], float(self._rainfall[idx])

    def driest_month(self) -> Tuple[str, float]:
        """Returns (month_name, rainfall_mm) for the month with minimum precipitation."""
        idx = int(np.argmin(self._rainfall))
        return MONTH_NAMES[idx], float(self._rainfall[idx])

    def coefficient_of_variation(self) -> float:
        """
        Precipitation Coefficient of Variation: CV = sigma / mu.
        Reflects seasonal rainfall dispersion throughout the calendar year.
        """
        mu = self.monthly_mean()
        if mu == 0:
            return 0.0
        sigma = float(np.std(self._rainfall, ddof=1))
        return float(sigma / mu)

    def __len__(self) -> int:
        return 12

    def __getitem__(self, idx: int) -> float:
        return float(self._rainfall[idx])

    def __repr__(self) -> str:
        return (
            f"Region(name='{self.name}', annual_total={self.annual_cumulative():.1f}mm, "
            f"mean={self.monthly_mean():.1f}mm, CV={self.coefficient_of_variation():.2f})"
        )


class CropRule:
    """
    Models monthly precipitation tolerance rules for agronomic suitability.
    Cites agronomic standards from FAO EcoCrop and NARO Uganda.
    """

    def __init__(
        self,
        crop_name: str,
        min_threshold_mm: float,
        max_threshold_mm: float,
        reference_citation: str = "FAO / NARO Uganda",
    ) -> None:
        if min_threshold_mm < 0 or max_threshold_mm <= min_threshold_mm:
            raise ValueError("Thresholds must satisfy 0 <= min_threshold_mm < max_threshold_mm.")

        self.crop_name = crop_name.strip()
        self.min_threshold = float(min_threshold_mm)
        self.max_threshold = float(max_threshold_mm)
        self.citation = reference_citation

    def classify_month(self, rainfall_mm: float) -> str:
        """
        Classifies rainfall into an agronomic state:
        - 'Drought stress': rainfall < min_threshold
        - 'Optimal': min_threshold <= rainfall <= max_threshold
        - 'Waterlogging risk': rainfall > max_threshold
        """
        if rainfall_mm < self.min_threshold:
            return "Drought stress"
        elif rainfall_mm > self.max_threshold:
            return "Waterlogging risk"
        else:
            return "Optimal"

    def classify_region(self, region: Region) -> List[str]:
        """Classifies each of the 12 calendar months for a Region."""
        return [self.classify_month(r) for r in region.monthly_rainfall]

    def __repr__(self) -> str:
        return (
            f"CropRule(crop='{self.crop_name}', range=[{self.min_threshold:.0f}, {self.max_threshold:.0f}]mm, "
            f"citation='{self.citation}')"
        )
