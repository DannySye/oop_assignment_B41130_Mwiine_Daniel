"""
Domain models for Solar-Battery and Hybrid Micro-Grid dispatch systems.
Implements MicroGrid (2x2) and HybridMicroGrid (3x3) with condition number diagnostics,
matrix solvers, and NNLS physical feasibility fallback protocols.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Sequence, Optional
import numpy as np
import scipy.linalg
from scipy.optimize import nnls


class MicroGrid:
    """
    Solar-Battery Micro-Grid operational dispatch solver.
    
    System Equations:
        3x + 2y = D1  (Daytime load, kWh)
        4x +  y = D2  (Critical-equipment load, kWh)
    where:
        x = Solar PV generation (kWh)
        y = Battery discharge (kWh)
    """

    DEFAULT_A = np.array([[3.0, 2.0], [4.0, 1.0]], dtype=np.float64)

    def __init__(self, coefficient_matrix: Optional[np.ndarray] = None) -> None:
        """
        Initializes MicroGrid with system coefficient matrix A.
        """
        if coefficient_matrix is None:
            self._A = np.copy(self.DEFAULT_A)
        else:
            arr = np.asarray(coefficient_matrix, dtype=np.float64)
            if arr.shape != (2, 2):
                raise ValueError(f"Coefficient matrix A must have shape (2, 2), got {arr.shape}.")
            self._A = np.copy(arr)

        # Precompute matrix invariants
        self._det = float(np.linalg.det(self._A))
        self._cond = float(np.linalg.cond(self._A))
        self._is_singular = bool(np.isclose(self._det, 0.0))

    @property
    def matrix_A(self) -> np.ndarray:
        """Returns read-only copy of coefficient matrix A."""
        return np.copy(self._A)

    @property
    def determinant(self) -> float:
        """Determinant det(A)."""
        return self._det

    @property
    def condition_number(self) -> float:
        """Condition number kappa(A) = ||A|| * ||A^-1||."""
        return self._cond

    @property
    def is_singular(self) -> bool:
        """Whether matrix A is singular."""
        return self._is_singular

    def solve_day(self, d1: float, d2: float) -> Tuple[float, float]:
        """
        Solves daily dispatch system for loads d1 and d2 using scipy.linalg.solve.

        Args:
            d1: Daytime load in kWh (> 0).
            d2: Critical-equipment load in kWh (> 0).

        Returns:
            Tuple of (solar_x, battery_y) in kWh.
        """
        if self._is_singular:
            raise np.linalg.LinAlgError("System matrix A is singular and cannot be inverted.")

        d_vec = np.array([float(d1), float(d2)], dtype=np.float64)
        solution = scipy.linalg.solve(self._A, d_vec)
        return float(solution[0]), float(solution[1])

    def solve_day_feasible(self, d1: float, d2: float) -> Dict[str, Any]:
        """
        Solves dispatch with physical non-negativity constraint audit.
        If x < 0 or y < 0, triggers Non-Negative Least Squares (NNLS) fallback.
        """
        x_raw, y_raw = self.solve_day(d1, d2)
        is_feasible = (x_raw >= 0.0) and (y_raw >= 0.0)

        if is_feasible:
            return {
                "solar_x": x_raw,
                "battery_y": y_raw,
                "is_feasible": True,
                "fallback_triggered": False,
                "d1_actual": d1,
                "d2_actual": d2,
                "residual_norm": 0.0,
            }
        else:
            # Fallback: Constrained Non-Negative Least Squares (NNLS)
            d_vec = np.array([float(d1), float(d2)], dtype=np.float64)
            nnls_sol, residual_norm = nnls(self._A, d_vec)
            return {
                "solar_x": float(nnls_sol[0]),
                "battery_y": float(nnls_sol[1]),
                "is_feasible": False,
                "fallback_triggered": True,
                "raw_unconstrained": (x_raw, y_raw),
                "d1_actual": float(np.dot(self._A[0], nnls_sol)),
                "d2_actual": float(np.dot(self._A[1], nnls_sol)),
                "residual_norm": float(residual_norm),
            }

    def solve_batch_iterative(self, demands: np.ndarray) -> np.ndarray:
        """
        Solves a sequence of days iteratively using a for loop.
        Demands shape: (N, 2) or (2, N). Returns dispatch array of shape (N, 2).
        """
        d_arr = np.asarray(demands, dtype=np.float64)
        if d_arr.ndim != 2:
            raise ValueError("demands must be a 2D array.")
        if d_arr.shape[1] == 2:
            n_days = d_arr.shape[0]
            results = np.zeros((n_days, 2), dtype=np.float64)
            for i in range(n_days):
                results[i, 0], results[i, 1] = self.solve_day(d_arr[i, 0], d_arr[i, 1])
            return results
        elif d_arr.shape[0] == 2:
            n_days = d_arr.shape[1]
            results = np.zeros((n_days, 2), dtype=np.float64)
            for i in range(n_days):
                results[i, 0], results[i, 1] = self.solve_day(d_arr[0, i], d_arr[1, i])
            return results
        else:
            raise ValueError(f"Demands array must have dimension 2 along one axis, got shape {d_arr.shape}.")

    def solve_batch_vectorized(self, demands: np.ndarray) -> np.ndarray:
        """
        Solves a sequence of days simultaneously using a single vectorized matrix RHS:
            A * X = D (shape: 2 x N) -> X = A^-1 * D
        Returns dispatch array of shape (N, 2).
        """
        d_arr = np.asarray(demands, dtype=np.float64)
        if d_arr.shape[1] == 2 and d_arr.shape[0] != 2:
            rhs = d_arr.T  # Shape: (2, N)
        elif d_arr.shape[0] == 2:
            rhs = d_arr
        else:
            rhs = d_arr.T

        # Solve single matrix equation A * X = RHS
        x_mat = scipy.linalg.solve(self._A, rhs)  # Shape: (2, N)
        return x_mat.T  # Return (N, 2)

    def __repr__(self) -> str:
        return f"MicroGrid(det={self._det:.2f}, cond={self._cond:.2f}, singular={self._is_singular})"


class HybridMicroGrid(MicroGrid):
    """
    Subclass integrating a backup diesel generator (z) into a 3x3 linear dispatch system.

    System Equations:
        3x + 2y + 1z = D1 (Daytime load)
        4x +  y + 2z = D2 (Critical equipment load)
        1x + 2y + 3z = D3 (Night-shift / thermal recovery load)
    """

    DEFAULT_HYBRID_A = np.array([
        [3.0, 2.0, 1.0],
        [4.0, 1.0, 2.0],
        [1.0, 2.0, 3.0],
    ], dtype=np.float64)

    def __init__(self, coefficient_matrix: Optional[np.ndarray] = None) -> None:
        """
        Initializes HybridMicroGrid with a 3x3 matrix.
        """
        if coefficient_matrix is None:
            self._A3 = np.copy(self.DEFAULT_HYBRID_A)
        else:
            arr = np.asarray(coefficient_matrix, dtype=np.float64)
            if arr.shape != (3, 3):
                raise ValueError(f"HybridMicroGrid requires 3x3 matrix, got {arr.shape}.")
            self._A3 = np.copy(arr)

        super().__init__(coefficient_matrix=self._A3[:2, :2])
        self._det3 = float(np.linalg.det(self._A3))
        self._cond3 = float(np.linalg.cond(self._A3))
        self._rank3 = int(np.linalg.matrix_rank(self._A3))
        self._is_singular3 = bool(self._rank3 < 3 or np.isclose(self._det3, 0.0))

    @property
    def matrix_3x3(self) -> np.ndarray:
        return np.copy(self._A3)

    @property
    def determinant_3x3(self) -> float:
        return self._det3

    @property
    def condition_number_3x3(self) -> float:
        return self._cond3

    @property
    def matrix_rank(self) -> int:
        return self._rank3

    def solve_day_hybrid(self, d1: float, d2: float, d3: float) -> Tuple[float, float, float]:
        """
        Solves 3x3 dispatch system for loads (d1, d2, d3).
        Returns (solar_x, battery_y, diesel_z) in kWh.
        """
        if self._is_singular3:
            raise np.linalg.LinAlgError(
                f"3x3 Hybrid system matrix is singular (rank={self._rank3} < 3, det={self._det3:.4e}). "
                "The third constraint is linearly dependent, creating infinite or non-existent solutions."
            )
        d_vec = np.array([float(d1), float(d2), float(d3)], dtype=np.float64)
        sol = scipy.linalg.solve(self._A3, d_vec)
        return float(sol[0]), float(sol[1]), float(sol[2])

    def __repr__(self) -> str:
        return f"HybridMicroGrid(det3={self._det3:.2f}, cond3={self._cond3:.2f}, rank={self._rank3})"
