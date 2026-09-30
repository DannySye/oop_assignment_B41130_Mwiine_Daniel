"""
Fish population biological dynamics and stochastic pricing models for Lake Victoria fisheries.
Implements logistic population dynamics with proportional harvest, carrying capacity constraints,
and bounded random walk price modeling.
"""

from __future__ import annotations
from typing import Tuple, Optional, Sequence, Dict, Any, List
import numpy as np


class FishStock:
    """
    Biological population model of Nile Perch / Tilapia stock in Lake Victoria.
    Implements discrete-time logistic growth subject to proportional harvesting:
        N(t+1) = N(t) + r * N(t) * (1 - N(t) / K) - h * N(t)
    
    Attributes:
        r (float): Intrinsic growth rate (default 0.40).
        K (float): Environmental carrying capacity in tonnes (default 10,000).
        N0 (float): Initial stock biomass in tonnes (default 4,000).
    """

    def __init__(
        self,
        intrinsic_growth_rate: float = 0.40,
        carrying_capacity: float = 10000.0,
        initial_biomass: float = 4000.0,
    ) -> None:
        if intrinsic_growth_rate <= 0:
            raise ValueError("Intrinsic growth rate r must be strictly positive.")
        if carrying_capacity <= 0:
            raise ValueError("Carrying capacity K must be strictly positive.")
        if initial_biomass <= 0:
            raise ValueError("Initial biomass N0 must be strictly positive.")

        self.r = float(intrinsic_growth_rate)
        self.K = float(carrying_capacity)
        self.N0 = float(initial_biomass)

    @property
    def theoretical_msy(self) -> float:
        """
        Theoretical Maximum Sustainable Yield (MSY):
            MSY = (r * K) / 4
        """
        return (self.r * self.K) / 4.0

    @property
    def biomass_at_msy(self) -> float:
        """Biomass yielding MSY: B_MSY = K / 2."""
        return self.K / 2.0

    @property
    def harvest_rate_at_msy(self) -> float:
        """Proportional harvest rate achieving MSY: h_MSY = r / 2."""
        return self.r / 2.0

    def simulate(
        self,
        weeks: int = 52,
        harvest_rate: float = 0.10,
        seasonal_closure_weeks: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates discrete-time logistic biomass trajectory over specified weeks.

        Args:
            weeks: Simulation duration in weeks (default 52).
            harvest_rate: Proportional extraction fraction h in [0, 1].
            seasonal_closure_weeks: Optional sequence of week indices (1-indexed) where harvest is 0.

        Returns:
            Tuple of (biomass_array, harvest_array) in tonnes.
        """
        if weeks <= 0:
            raise ValueError("Simulation weeks must be at least 1.")
        if not (0.0 <= harvest_rate <= 1.0):
            raise ValueError(f"Harvest rate h must be in [0, 1], got {harvest_rate}.")

        closure_set = set(seasonal_closure_weeks or [])

        biomass = np.zeros(weeks + 1, dtype=np.float64)
        harvest = np.zeros(weeks, dtype=np.float64)
        biomass[0] = self.N0

        # We normalize growth rate to weekly step: dt = 1/52 (annualized) or step rate
        # Prompt: N(t+1) = N(t) + r * N(t) * (1 - N(t) / K) - h * N(t)
        # Using the discrete equation given in the specification
        for t in range(weeks):
            current_N = biomass[t]
            current_week = t + 1
            h_eff = 0.0 if current_week in closure_set else harvest_rate

            harvest_t = h_eff * current_N
            growth_t = self.r * current_N * (1.0 - current_N / self.K)
            next_N = current_N + growth_t - harvest_t

            # Biological extinction floor at 0
            biomass[t + 1] = max(0.0, next_N)
            harvest[t] = harvest_t

        return biomass, harvest

    def __repr__(self) -> str:
        return f"FishStock(r={self.r:.2f}, K={self.K:,.0f}t, N0={self.N0:,.0f}t, MSY={self.theoretical_msy:,.1f}t)"


class PriceModel:
    """
    Simulates stochastic market prices (UGX / kg) for export fish via a bounded random walk.
    
    Default parameters:
        - Initial price: UGX 12,000 / kg
        - Lower boundary: UGX 9,000 / kg (reflecting/clamped)
        - Upper boundary: UGX 16,000 / kg (reflecting/clamped)
        - Volatility step: sigma = UGX 500 / week
    """

    def __init__(
        self,
        initial_price: float = 12000.0,
        min_price: float = 9000.0,
        max_price: float = 16000.0,
        volatility: float = 500.0,
    ) -> None:
        if min_price >= max_price:
            raise ValueError("min_price must be strictly less than max_price.")
        if not (min_price <= initial_price <= max_price):
            raise ValueError("initial_price must lie within [min_price, max_price].")

        self.initial_price = float(initial_price)
        self.min_price = float(min_price)
        self.max_price = float(max_price)
        self.volatility = float(volatility)

    def simulate_prices(
        self,
        weeks: int = 52,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Simulates weekly price trajectory (UGX / kg) with reflecting boundary conditions.

        Args:
            weeks: Number of weeks to simulate.
            rng: Optional NumPy random Generator.

        Returns:
            1D array of weekly prices of length `weeks`.
        """
        if rng is None:
            rng = np.random.default_rng()

        shocks = rng.normal(0.0, self.volatility, size=weeks)
        prices = np.zeros(weeks, dtype=np.float64)
        current_p = self.initial_price

        for t in range(weeks):
            proposed_p = current_p + shocks[t]

            # Reflecting boundary conditions
            if proposed_p < self.min_price:
                current_p = self.min_price + (self.min_price - proposed_p)
                current_p = min(current_p, self.max_price)
            elif proposed_p > self.max_price:
                current_p = self.max_price - (proposed_p - self.max_price)
                current_p = max(current_p, self.min_price)
            else:
                current_p = proposed_p

            prices[t] = current_p

        return prices

    def __repr__(self) -> str:
        return f"PriceModel(p0={self.initial_price:,.0f}, range=[{self.min_price:,.0f}, {self.max_price:,.0f}])"
