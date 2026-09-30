"""
Domain model representing historical district population time series.
Designed for the Uganda Bureau of Statistics (UBOS) population forecasting project.
"""

from __future__ import annotations
from typing import Sequence, Iterator, Tuple, Union, Any
import numpy as np


class DistrictPopulation:
    """
    Represents historical population data for a Ugandan district.
    
    Attributes:
        district_name (str): Name of the administrative district.
        years (np.ndarray): 1D array of historical calendar years (integer dtype).
        values (np.ndarray): 1D array of population estimates in thousands (float64 dtype).
    """

    def __init__(
        self,
        district_name: str,
        years: Sequence[int] | np.ndarray,
        values: Sequence[float | int] | np.ndarray,
    ) -> None:
        """
        Initializes and validates a DistrictPopulation instance.

        Args:
            district_name: Name of the district (e.g. 'Kampala', 'Wakiso').
            years: Sequence or NumPy array of chronological years.
            values: Sequence or NumPy array of population figures (in thousands).

        Raises:
            TypeError: If input types are invalid.
            ValueError: If lengths do not match, values are non-positive, or years are not strictly increasing.
        """
        if not isinstance(district_name, str) or not district_name.strip():
            raise TypeError("district_name must be a non-empty string.")

        # Convert years to 1D integer numpy array
        try:
            years_arr = np.asarray(years, dtype=np.int64)
        except Exception as exc:
            raise TypeError(f"Could not convert years to integer array: {exc}") from exc

        # Convert values to 1D float64 numpy array
        try:
            values_arr = np.asarray(values, dtype=np.float64)
        except Exception as exc:
            raise TypeError(f"Could not convert values to float64 array: {exc}") from exc

        if years_arr.ndim != 1 or values_arr.ndim != 1:
            raise ValueError(f"years and values must be 1-dimensional, got {years_arr.ndim}D and {values_arr.ndim}D.")

        if len(years_arr) < 2:
            raise ValueError(f"At least 2 observations are required to model population dynamics, got {len(years_arr)}.")

        if len(years_arr) != len(values_arr):
            raise ValueError(
                f"Dimension mismatch: years has length {len(years_arr)}, but values has length {len(values_arr)}."
            )

        if np.any(years_arr <= 0):
            raise ValueError("All calendar years must be strictly positive integers.")

        if np.any(values_arr <= 0):
            raise ValueError("All population values must be strictly positive (> 0).")

        # Check strictly increasing order for years
        diffs = np.diff(years_arr)
        if np.any(diffs <= 0):
            raise ValueError("Years must be sorted in strictly ascending chronological order without duplicates.")

        self._district_name = district_name.strip()
        self._years = np.copy(years_arr)
        self._values = np.copy(values_arr)

        # Make arrays read-only to guarantee immutability
        self._years.flags.writeable = False
        self._values.flags.writeable = False

    @property
    def district_name(self) -> str:
        """Returns the district name."""
        return self._district_name

    @property
    def years(self) -> np.ndarray:
        """Returns an immutable array of years."""
        return self._years

    @property
    def values(self) -> np.ndarray:
        """Returns an immutable array of population estimates (thousands)."""
        return self._values

    def __len__(self) -> int:
        """Returns the number of recorded annual observations."""
        return len(self._years)

    def __repr__(self) -> str:
        """Detailed string representation of DistrictPopulation."""
        start_year, end_year = self._years[0], self._years[-1]
        start_val, end_val = self._values[0], self._values[-1]
        return (
            f"DistrictPopulation(district_name='{self._district_name}', "
            f"period={start_year}-{end_year}, n_obs={len(self._years)}, "
            f"range=[{start_val:.1f}k, {end_val:.1f}k])"
        )

    def __getitem__(self, item: Union[int, slice]) -> Union[Tuple[int, float], "DistrictPopulation"]:
        """
        Allows indexing and slicing.
        Integer index returns (year, population).
        Slice returns a new DistrictPopulation subset.
        """
        if isinstance(item, int):
            return int(self._years[item]), float(self._values[item])
        elif isinstance(item, slice):
            sliced_years = self._years[item]
            sliced_values = self._values[item]
            if len(sliced_years) < 2:
                raise ValueError("Slice subset must contain at least 2 observations.")
            return DistrictPopulation(self._district_name, sliced_years, sliced_values)
        else:
            raise TypeError(f"Invalid index type: {type(item).__name__}")

    def __iter__(self) -> Iterator[Tuple[int, float]]:
        """Yields (year, population) tuples across the time series."""
        for yr, val in zip(self._years, self._values):
            yield int(yr), float(val)

    def __eq__(self, other: Any) -> bool:
        """Checks equality against another DistrictPopulation instance."""
        if not isinstance(other, DistrictPopulation):
            return False
        return (
            self._district_name == other._district_name
            and np.array_equal(self._years, other._years)
            and np.allclose(self._values, other._values)
        )

    def get_subset(self, start_year: int, end_year: int) -> "DistrictPopulation":
        """
        Extracts a temporal sub-series between start_year and end_year inclusive.

        Args:
            start_year: Beginning calendar year.
            end_year: Ending calendar year.

        Returns:
            DistrictPopulation: A new instance containing the filtered years and values.
        """
        mask = (self._years >= start_year) & (self._years <= end_year)
        sub_years = self._years[mask]
        sub_values = self._values[mask]

        if len(sub_years) < 2:
            raise ValueError(
                f"Subset between {start_year} and {end_year} contains fewer than 2 data points ({len(sub_years)} found)."
            )

        return DistrictPopulation(self._district_name, sub_years, sub_values)

    def to_dict(self) -> dict:
        """Converts the object to a dictionary representation."""
        return {
            "district_name": self._district_name,
            "years": self._years.tolist(),
            "values": self._values.tolist(),
            "n_obs": len(self),
            "start_year": int(self._years[0]),
            "end_year": int(self._years[-1]),
        }
