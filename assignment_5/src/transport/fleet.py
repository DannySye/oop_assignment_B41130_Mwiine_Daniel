"""
Fleet deployment sizing and seasonal demand synthesis for urban matatu fleets.
"""

from __future__ import annotations
import math
from typing import Dict, Any, Tuple
import numpy as np


def calculate_fleet_deployment(
    projected_passengers: float,
    trips_per_day: int = 8,
    vehicle_capacity: int = 14,
    safety_margin_pct: float = 0.15,
) -> Dict[str, Any]:
    """
    Computes required fleet size to satisfy projected passenger demand.

    Specifications:
        - 14-seater commuter vehicle.
        - 8 one-way trips per vehicle per operating day.
        - Daily vehicle capacity: 8 * 14 = 112 passengers/vehicle/day.
        - Operational safety margin buffer: +15%.
        - Integer ceiling rounding: vehicles cannot be deployed fractionally.
    """
    if projected_passengers < 0:
        raise ValueError("projected_passengers cannot be negative.")
    if trips_per_day <= 0 or vehicle_capacity <= 0:
        raise ValueError("Trips and capacity must be positive integers.")

    daily_capacity_per_vehicle = trips_per_day * vehicle_capacity  # 112
    base_vehicles = projected_passengers / daily_capacity_per_vehicle
    buffered_vehicles = base_vehicles * (1.0 + safety_margin_pct)
    required_vehicles = int(math.ceil(buffered_vehicles))

    effective_total_capacity = required_vehicles * daily_capacity_per_vehicle
    spare_capacity = effective_total_capacity - projected_passengers

    return {
        "projected_passengers": projected_passengers,
        "trips_per_day": trips_per_day,
        "vehicle_capacity": vehicle_capacity,
        "daily_capacity_per_vehicle": daily_capacity_per_vehicle,
        "base_vehicles_exact": base_vehicles,
        "safety_margin_pct": safety_margin_pct,
        "buffered_vehicles_exact": buffered_vehicles,
        "deployed_fleet_size": required_vehicles,
        "effective_total_capacity": effective_total_capacity,
        "spare_passenger_capacity": spare_capacity,
    }


def generate_60day_seasonal_demand(
    base_demand: float = 60.0,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Extension Task: Synthesizes a 60-day commuter demand dataset featuring
    authentic day-of-week seasonality (e.g. peak volumes on Fridays, suppressed on Sundays).

    Day-of-week multipliers (Mon=0 .. Sun=6):
        Mon: 1.05, Tue: 1.00, Wed: 1.02, Thu: 1.08, Fri: 1.30 (PEAK), Sat: 0.85, Sun: 0.65 (DIP)
    """
    rng = np.random.default_rng(random_seed)
    dow_multipliers = np.array([1.05, 1.00, 1.02, 1.08, 1.30, 0.85, 0.65])

    days = 60
    demand = np.zeros(days, dtype=np.float64)

    for d in range(days):
        dow = d % 7
        mult = dow_multipliers[dow]
        noise = rng.normal(0.0, 3.0)
        demand[d] = max(10.0, base_demand * mult + noise)

    return demand
