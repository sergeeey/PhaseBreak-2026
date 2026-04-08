"""LPPLS Optimizer: Grid search + L-BFGS-B refinement.

Strategy (Sornette 2003):
1. Outer: Grid search over nonlinear params (tc, m, omega) — ~1000 combinations
2. Inner: OLS for linear params (A, B, C1, C2) at each grid point
3. Polish: L-BFGS-B on all 7 params starting from best grid point
4. Filter: Sornette constraints (m, omega, B < 0)
"""

from __future__ import annotations

import structlog
import numpy as np
from scipy.optimize import minimize
from numpy.typing import NDArray

from src.lppls.model import LPPLS, LPPLSParams

log = structlog.get_logger()

# WHY: Sornette (2003) empirical ranges for financial bubbles
DEFAULT_BOUNDS = {
    "tc_margin_left": 1,  # tc > last observation + 1
    "tc_margin_right": 252,  # tc within 1 trading year
    "m_low": 0.1,
    "m_high": 0.9,
    "omega_low": 6.0,
    "omega_high": 13.0,
}


class LPPLSOptimizer:
    """Two-stage LPPLS optimizer: grid search → L-BFGS-B polish."""

    def __init__(
        self,
        tc_range: tuple[float, float] | None = None,
        m_range: tuple[float, float] = (0.1, 0.9),
        omega_range: tuple[float, float] = (6.0, 13.0),
        grid_size: int = 10,
        n_best: int = 5,
        *,
        filter_m_interior: tuple[float, float] | None = None,
        filter_omega_interior: tuple[float, float] | None = None,
        min_abs_B: float = 0.003,
        min_damping: float = 0.5,
    ) -> None:
        self.tc_range = tc_range
        self.m_range = m_range
        self.omega_range = omega_range
        self.grid_size = grid_size
        self.n_best = n_best
        # Post-fit interior checks (reject boundary-stuck optima). Defaults match legacy behavior.
        self._filter_m = filter_m_interior if filter_m_interior is not None else (0.1, 0.87)
        self._filter_omega = filter_omega_interior if filter_omega_interior is not None else (5.0, 13.5)
        self._min_abs_B = min_abs_B
        self._min_damping = min_damping

    def fit(self, t: NDArray, log_price: NDArray) -> LPPLS:
        """Fit LPPLS model to time series.

        Args:
            t: Time array (integer indices or float days)
            log_price: Natural log of price series

        Returns:
            Fitted LPPLS model with best parameters
        """
        t_last = float(t[-1])

        # WHY: default tc range = 1 day to 1 year after last observation
        if self.tc_range is None:
            tc_lo = t_last + DEFAULT_BOUNDS["tc_margin_left"]
            tc_hi = t_last + DEFAULT_BOUNDS["tc_margin_right"]
        else:
            tc_lo, tc_hi = self.tc_range

        # Stage 1: Grid search
        best_candidates = self._grid_search(t, log_price, tc_lo, tc_hi)
        log.info("grid_search_done", n_candidates=len(best_candidates))

        # Stage 2: L-BFGS-B polish on top candidates
        best_params = None
        best_sse = float("inf")

        for tc0, m0, omega0, sse0 in best_candidates:
            try:
                params, sse = self._polish(t, log_price, tc0, m0, omega0, tc_lo, tc_hi)
                if sse < best_sse and self._passes_filters_impl(params):
                    best_sse = sse
                    best_params = params
            except Exception:
                continue

        if best_params is None:
            # WHY: fallback must also respect filters. If no grid point
            # passes → return best-fit params as-is (is_bubble will be False
            # naturally or via the filter-aware flag). This prevents false
            # positives where the optimizer force-returns a "bubble" from noise.
            log.warning("optimization_failed", msg="No valid solution passed Sornette filters")
            tc0, m0, omega0, _ = best_candidates[0]
            A, B, C1, C2 = LPPLS.solve_linear(t, log_price, tc0, m0, omega0)
            fallback = LPPLSParams(tc=tc0, m=m0, omega=omega0, A=A, B=B, C1=C1, C2=C2)

            if not self._passes_filters_impl(fallback):
                # Force B positive → is_bubble = False
                best_params = LPPLSParams(
                    tc=tc0, m=m0, omega=omega0, A=A, B=abs(B) + 1e-5, C1=C1, C2=C2
                )
            else:
                best_params = fallback

        model = LPPLS(params=best_params)
        r2 = model.r_squared(t, log_price)
        log.info(
            "fit_complete",
            tc=best_params.tc,
            m=round(best_params.m, 4),
            omega=round(best_params.omega, 4),
            B=round(best_params.B, 6),
            is_bubble=best_params.is_bubble,
            r_squared=round(r2, 4),
        )
        return model

    def _grid_search(
        self, t: NDArray, log_price: NDArray, tc_lo: float, tc_hi: float
    ) -> list[tuple[float, float, float, float]]:
        """Exhaustive grid search over (tc, m, omega). Returns top-N by SSE."""
        tc_grid = np.linspace(tc_lo, tc_hi, self.grid_size)
        m_grid = np.linspace(self.m_range[0], self.m_range[1], self.grid_size)
        omega_grid = np.linspace(self.omega_range[0], self.omega_range[1], self.grid_size)

        results: list[tuple[float, float, float, float]] = []

        for tc in tc_grid:
            for m in m_grid:
                for omega in omega_grid:
                    try:
                        sse = LPPLS.sse_for_nonlinear(t, log_price, tc, m, omega)
                        if np.isfinite(sse):
                            results.append((tc, m, omega, sse))
                    except Exception:
                        continue

        results.sort(key=lambda x: x[3])
        return results[: self.n_best]

    def _polish(
        self,
        t: NDArray,
        log_price: NDArray,
        tc0: float,
        m0: float,
        omega0: float,
        tc_lo: float,
        tc_hi: float,
    ) -> tuple[LPPLSParams, float]:
        """Refine nonlinear params with L-BFGS-B, solve linear via OLS."""

        def objective(x: NDArray) -> float:
            tc, m, omega = x
            return LPPLS.sse_for_nonlinear(t, log_price, tc, m, omega)

        bounds = [
            (tc_lo, tc_hi),
            (self.m_range[0], self.m_range[1]),
            (self.omega_range[0], self.omega_range[1]),
        ]

        result = minimize(
            objective,
            x0=[tc0, m0, omega0],
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-10},
        )

        tc_opt, m_opt, omega_opt = result.x
        A, B, C1, C2 = LPPLS.solve_linear(t, log_price, tc_opt, m_opt, omega_opt)

        params = LPPLSParams(
            tc=float(tc_opt),
            m=float(m_opt),
            omega=float(omega_opt),
            A=float(A),
            B=float(B),
            C1=float(C1),
            C2=float(C2),
        )
        return params, float(result.fun)

    def _passes_filters_impl(self, params: LPPLSParams) -> bool:
        """Sornette filter conditions for valid LPPLS fit.

        Damping is canonical |B|/|C| (see LPPLSParams.damping).

        WHY tighter than standard: vanilla Sornette filters produce 80%+
        false positive rate on non-bubble data. Key fixes:
        - m at boundaries (0.1/0.9) = optimizer stuck, not real signal
        - |B| too small = no real super-exponential growth
        - omega at boundaries = likely overfitting
        """
        m_lo, m_hi = self._filter_m
        w_lo, w_hi = self._filter_omega
        if not (m_lo < params.m < m_hi):
            return False
        if not (w_lo < params.omega < w_hi):
            return False
        if params.B >= 0:
            return False
        if abs(params.B) < self._min_abs_B:
            return False
        if params.damping < self._min_damping:
            return False
        return True
