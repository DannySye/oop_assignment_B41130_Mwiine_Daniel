"""
Domain models for Matatu commuter transport routes and microeconomic market equilibrium.
Designed for the Kampala Minibus Taxi Association.
"""

from __future__ import annotations
import statistics
from typing import Sequence, Dict, Any, Tuple
import numpy as np
import scipy.linalg


class Route:
    """
    Represents a radial commuter taxi corridor in the Kampala Metropolitan Area.
    
    Attributes:
        route_name (str): Origin-destination corridor (e.g. 'Kampala-Ntinda').
        passengers (np.ndarray): 1D array of daily passenger tallies (float64).
        fare_ugx (float): Fixed tariff per passenger in UGX.
    """

    def __init__(
        self,
        route_name: str,
        passengers: Sequence[float | int] | np.ndarray,
        fare_ugx: float,
    ) -> None:
        if not isinstance(route_name, str) or not route_name.strip():
            raise TypeError("route_name must be a non-empty string.")
        if fare_ugx <= 0:
            raise ValueError(f"fare_ugx must be strictly positive (> 0), got {fare_ugx}.")

        arr = np.asarray(passengers, dtype=np.float64)
        if arr.ndim != 1 or len(arr) == 0:
            raise ValueError("passengers must be a non-empty 1D array.")
        if np.any(arr < 0):
            raise ValueError("Passenger counts cannot be negative.")

        self.route_name = route_name.strip()
        self.fare_ugx = float(fare_ugx)
        self._passengers = np.copy(arr)
        self._passengers.flags.writeable = False

    @property
    def passengers(self) -> np.ndarray:
        return self._passengers

    def daily_turnover(self) -> np.ndarray:
        """Computes daily revenue in UGX: Turnover(t) = Passengers(t) * Fare."""
        return self._passengers * self.fare_ugx

    def cumulative_revenue(self) -> float:
        """Returns total revenue earned across the recorded operating window."""
        return float(np.sum(self.daily_turnover()))

    def passenger_statistics(self) -> Dict[str, float]:
        """
        Computes mean, variance, and standard deviation of daily passenger volume
        using Python's standard `statistics` module.
        """
        pass_list = [float(p) for p in self._passengers]
        if len(pass_list) < 2:
            raise ValueError("At least 2 days of observations required for variance.")

        return {
            "mean": float(statistics.mean(pass_list)),
            "variance": float(statistics.variance(pass_list)),
            "stdev": float(statistics.stdev(pass_list)),
        }

    def revenue_statistics(self) -> Dict[str, float]:
        """Computes mean, variance, and standard deviation of daily turnover (UGX)."""
        rev_list = [float(r) for r in self.daily_turnover()]
        return {
            "mean_revenue_ugx": float(statistics.mean(rev_list)),
            "variance_revenue_ugx2": float(statistics.variance(rev_list)),
            "stdev_revenue_ugx": float(statistics.stdev(rev_list)),
        }

    def __len__(self) -> int:
        return len(self._passengers)

    def __repr__(self) -> str:
        stats = self.passenger_statistics()
        return (
            f"Route('{self.route_name}', fare=UGX {self.fare_ugx:,.0f}, "
            f"days={len(self)}, mean_passengers={stats['mean']:.1f})"
        )


def solve_ntinda_market_equilibrium() -> Dict[str, Any]:
    """
    Solves the microeconomic market equilibrium for the Kampala-Ntinda corridor.

    Functions:
        Demand: Qd = 120 - 0.02 * P
        Supply: Qs =  10 + 0.03 * P

    Equilibrium Linear System (Q - Q_func = 0):
        [ 1.0   0.02 ] [ Q ]   [ 120.0 ]
        [ 1.0  -0.03 ] [ P ] = [  10.0 ]

    Solved via scipy.linalg.solve.
    """
    A = np.array([
        [1.0, 0.02],
        [1.0, -0.03],
    ], dtype=np.float64)
    b = np.array([120.0, 10.0], dtype=np.float64)

    # Solve system for [Q*, P*]
    sol = scipy.linalg.solve(A, b)
    q_star = float(sol[0])
    p_star = float(sol[1])

    prevailing_fare = 2000.0
    qd_prevailing = 120.0 - 0.02 * prevailing_fare  # 80
    qs_prevailing = 10.0 + 0.03 * prevailing_fare   # 70
    capacity_gap = qd_prevailing - qs_prevailing    # +10 (Shortage)

    is_below_equilibrium = prevailing_fare < p_star

    analysis = (
        f"MICROECONOMIC MARKET EQUILIBRIUM ANALYSIS (NTINDA CORRIDOR):\n"
        f"1. Theoretical Equilibrium: P* = UGX {p_star:,.0f}, Q* = {q_star:.1f} passengers/trip-hour.\n"
        f"2. Prevailing Fare: UGX {prevailing_fare:,.0f} is set BELOW the theoretical equilibrium (P* = UGX {p_star:,.0f}).\n"
        f"3. Market Implication: At UGX 2,000, passenger demand is Qd = {qd_prevailing:.0f} while operator supply is Qs = {qs_prevailing:.0f}.\n"
        f"   This produces an acute capacity shortage of {capacity_gap:.0f} passengers/trip-hour (+14.3% excess demand), "
        f"   manifesting as peak-hour taxi stage congestion, long commuter passenger queues, and informal fare gouging during rainfall."
    )

    return {
        "coefficient_matrix": A.tolist(),
        "rhs_vector": b.tolist(),
        "equilibrium_fare_p_star": p_star,
        "equilibrium_volume_q_star": q_star,
        "prevailing_fare_ugx": prevailing_fare,
        "qd_at_prevailing": qd_prevailing,
        "qs_at_prevailing": qs_prevailing,
        "capacity_shortage": capacity_gap,
        "is_below_equilibrium": is_below_equilibrium,
        "economic_interpretation": analysis,
    }
