"""LPPLS (Log-Periodic Power Law Singularity) model.

Implements Sornette's equation for phase transition detection:
    ln(E[p(t)]) = A + B(tc - t)^m + C(tc - t)^m * cos(ω * ln(tc - t) + φ)

References:
    Sornette, D. (2003). "Why Stock Markets Crash"
    Johansen, A., Sornette, D. (1999). "Critical Crashes"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LPPLSParams:
    """Fitted LPPLS parameters."""

    tc: float  # critical time (date of phase transition)
    m: float  # power law exponent (0.1 < m < 0.9)
    omega: float  # angular log-frequency (6 < ω < 13 for finance)
    A: float  # log-price at tc
    B: float  # amplitude (B < 0 for bubble)
    C1: float  # cosine amplitude
    C2: float  # sine amplitude

    @property
    def C(self) -> float:
        """Combined log-periodic amplitude."""
        return np.sqrt(self.C1**2 + self.C2**2)

    @property
    def phi(self) -> float:
        """Phase angle."""
        return np.arctan2(self.C2, self.C1)

    @property
    def is_bubble(self) -> bool:
        """B < 0 indicates a bubble (super-exponential growth)."""
        return self.B < 0

    @property
    def damping(self) -> float:
        """Power-law vs log-periodic strength: |B|/|C|, C = sqrt(C1²+C2²).

        Used consistently in optimizer post-filters and DS-LPPLS (see ds_filters).
        """
        c = self.C
        if c < 1e-10:
            return float("inf")
        return abs(self.B) / c


class LPPLS:
    """Log-Periodic Power Law Singularity model.

    Detects phase transitions (bubbles, crashes) in time series data
    by fitting log-periodic oscillations near a critical time tc.
    """

    def __init__(self, params: LPPLSParams | None = None) -> None:
        self.params = params

    @staticmethod
    def func(
        t: NDArray, tc: float, m: float, omega: float, A: float, B: float, C1: float, C2: float
    ) -> NDArray:
        """LPPLS function value at time t.

        Uses linearized form with C1, C2 instead of C, phi
        to enable OLS for linear parameters.
        """
        dt = tc - t
        # WHY: clip to avoid log(0) or negative values
        dt = np.clip(dt, 1e-10, None)

        dt_m = np.power(dt, m)
        log_dt = np.log(dt)

        f = A + B * dt_m + C1 * dt_m * np.cos(omega * log_dt) + C2 * dt_m * np.sin(omega * log_dt)
        return f

    def predict(self, t: NDArray) -> NDArray:
        """Predict log-price at times t using fitted parameters."""
        if self.params is None:
            raise ValueError("Model not fitted. Call fit() or set params first.")
        p = self.params
        return self.func(t, p.tc, p.m, p.omega, p.A, p.B, p.C1, p.C2)

    def residuals(self, t: NDArray, log_price: NDArray) -> NDArray:
        """Compute residuals (observed - predicted)."""
        return log_price - self.predict(t)

    def r_squared(self, t: NDArray, log_price: NDArray) -> float:
        """Coefficient of determination R²."""
        ss_res = np.sum(self.residuals(t, log_price) ** 2)
        ss_tot = np.sum((log_price - np.mean(log_price)) ** 2)
        if ss_tot < 1e-10:
            return 0.0
        return 1.0 - ss_res / ss_tot

    def rmse(self, t: NDArray, log_price: NDArray) -> float:
        """Root mean squared error."""
        res = self.residuals(t, log_price)
        return float(np.sqrt(np.mean(res**2)))

    @staticmethod
    def solve_linear(
        t: NDArray, log_price: NDArray, tc: float, m: float, omega: float
    ) -> tuple[float, float, float, float]:
        """Solve for linear parameters (A, B, C1, C2) via OLS.

        Given fixed nonlinear params (tc, m, omega), the LPPLS equation is
        linear in (A, B, C1, C2). OLS is exact and fast.
        """
        dt = tc - t
        dt = np.clip(dt, 1e-10, None)

        dt_m = np.power(dt, m)
        log_dt = np.log(dt)

        # Design matrix: [1, dt^m, dt^m*cos(ω*ln(dt)), dt^m*sin(ω*ln(dt))]
        X = np.column_stack(
            [
                np.ones_like(t),
                dt_m,
                dt_m * np.cos(omega * log_dt),
                dt_m * np.sin(omega * log_dt),
            ]
        )

        # OLS: β = (X'X)^{-1} X'y
        try:
            beta, _, _, _ = np.linalg.lstsq(X, log_price, rcond=None)
            return float(beta[0]), float(beta[1]), float(beta[2]), float(beta[3])
        except np.linalg.LinAlgError:
            return 0.0, 0.0, 0.0, 0.0

    @staticmethod
    def sse_for_nonlinear(
        t: NDArray, log_price: NDArray, tc: float, m: float, omega: float
    ) -> float:
        """Sum of squared errors for given nonlinear params.

        Solves linear params via OLS, returns SSE.
        Used as objective function for grid search / optimizer.
        """
        A, B, C1, C2 = LPPLS.solve_linear(t, log_price, tc, m, omega)
        predicted = LPPLS.func(t, tc, m, omega, A, B, C1, C2)
        return float(np.sum((log_price - predicted) ** 2))
