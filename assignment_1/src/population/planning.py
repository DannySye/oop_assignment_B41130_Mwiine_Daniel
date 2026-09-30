"""
Infrastructure planning module for primary school classroom construction.
Translates demographic population projections into actionable physical infrastructure requirements.
"""

from __future__ import annotations
import math
from typing import Dict, Any, Sequence, Tuple
import numpy as np


class ClassroomPlanner:
    """
    Computes required primary school classroom construction based on population growth.

    Standard parameters:
        - Primary school age proportion: 18% (0.18) of total population (Ministry of Education & Sports / UBOS).
        - Standard classroom capacity: 53 pupils per classroom (National Primary School Guidelines).
    """

    def __init__(
        self,
        school_age_ratio: float = 0.18,
        classroom_capacity: int = 53,
    ) -> None:
        """
        Initializes the planner with demographic and infrastructure standards.

        Args:
            school_age_ratio: Fraction of population in primary school age cohort (default 0.18).
            classroom_capacity: Standard seating capacity per classroom (default 53).
        """
        if not (0.0 < school_age_ratio < 1.0):
            raise ValueError(f"school_age_ratio must be between 0 and 1, got {school_age_ratio}.")
        if classroom_capacity <= 0:
            raise ValueError(f"classroom_capacity must be a strictly positive integer, got {classroom_capacity}.")

        self.school_age_ratio = float(school_age_ratio)
        self.classroom_capacity = int(classroom_capacity)

    def calculate_requirements(
        self,
        district_name: str,
        base_population_k: float,
        projected_population_k: float,
        base_year: int = 2024,
        target_year: int = 2029,
    ) -> Dict[str, Any]:
        """
        Calculates net additional school-age children and classrooms required.

        Args:
            district_name: Name of district.
            base_population_k: Baseline population in thousands (e.g. at 2024).
            projected_population_k: Forecasted population in thousands (e.g. at 2029).
            base_year: Starting year of planning horizon.
            target_year: Ending year of planning horizon.

        Returns:
            Dictionary containing net population increase, additional pupils, and required classrooms.
        """
        if base_population_k <= 0 or projected_population_k <= 0:
            raise ValueError("Population figures must be strictly positive.")

        pop_delta_k = projected_population_k - base_population_k
        net_population_increase = pop_delta_k * 1000.0
        additional_pupils = net_population_increase * self.school_age_ratio

        # Ceiling rounding: cannot build a fraction of a classroom
        if additional_pupils > 0:
            net_classrooms_needed = int(math.ceil(additional_pupils / self.classroom_capacity))
        else:
            net_classrooms_needed = 0

        return {
            "district_name": district_name,
            "base_year": base_year,
            "target_year": target_year,
            "base_population_k": float(base_population_k),
            "projected_population_k": float(projected_population_k),
            "net_population_growth": float(net_population_increase),
            "school_age_ratio": self.school_age_ratio,
            "additional_pupils": float(additional_pupils),
            "pupils_per_classroom": self.classroom_capacity,
            "net_classrooms_needed": net_classrooms_needed,
        }

    def plan_multiple_districts(
        self,
        district_data: Sequence[Tuple[str, float, float]],
        base_year: int = 2024,
        target_year: int = 2029,
    ) -> Sequence[Dict[str, Any]]:
        """
        Generates planning schedules across a list of (district_name, base_pop_k, proj_pop_k).
        """
        return [
            self.calculate_requirements(
                district_name=name,
                base_population_k=b_pop,
                projected_population_k=p_pop,
                base_year=base_year,
                target_year=target_year,
            )
            for name, b_pop, p_pop in district_data
        ]
