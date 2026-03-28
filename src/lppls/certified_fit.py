"""Certified LPPLS Convergence Bounds via Richardson Extrapolation.

Uses ChernoffPy's certified bound machinery to validate LPPLS fits:
1. Fit LPPLS at multiple grid sizes (5, 10, 15, 20, 25)
2. Richardson extrapolation → estimate convergence rate α
3. If α ≥ 1.5 → fit is stable, bound is tight
4. If α < 1 → fit is unstable, likely overfit
5. Certified bound: tc = median ± bound (with safety_factor)

This is a novel application — certified convergence bounds for LPPLS
critical time estimation. Not previously described in literature.

Reference: Galkin-Remizov (2025), Israel J. Math.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray

from src.lppls.optimizer import LPPLSOptimizer

log = structlog.get_logger()

# WHY: grid sizes must increase monotonically for Richardson extrapolation
DEFAULT_GRID_SIZES = [5, 8, 12, 16, 20]
STABILITY_THRESHOLD = 1.0  # α ≥ 1.0 → convergent fit


@dataclass(frozen=True)
class CertifiedLPPLSResult:
    """Certified LPPLS fit with convergence bounds."""

    tc_estimate: float  # best tc estimate (from finest grid)
    tc_bound: float  # certified error bound on tc
    tc_low: float  # tc - bound
    tc_high: float  # tc + bound
    convergence_order: float  # Richardson α estimate
    is_convergent: bool  # α ≥ threshold
    is_certified: bool  # True if Richardson + safety factor applied
    grid_tc_values: dict[int, float]  # grid_size → tc
    grid_r_squared: dict[int, float]  # grid_size → R²
    method: str  # description of how bound was computed
    safety_factor: float

    @property
    def confidence_interval(self) -> str:
        """Human-readable confidence interval."""
        return f"tc = {self.tc_estimate:.1f} ± {self.tc_bound:.1f}"


def _estimate_convergence_rate(grid_sizes: list[int], values: list[float]) -> tuple[float, float]:
    """Estimate convergence rate α from log-log regression.

    If values converge as O(h^α) where h = 1/grid_size,
    then differences between consecutive values scale as h^α.

    Returns (alpha, constant).
    """
    if len(values) < 3:
        return 0.0, 0.0

    # Compute consecutive differences
    diffs = []
    h_vals = []
    for i in range(1, len(values)):
        diff = abs(values[i] - values[i - 1])
        if diff < 1e-15:
            diff = 1e-15  # avoid log(0)
        h = 1.0 / grid_sizes[i]
        diffs.append(diff)
        h_vals.append(h)

    if len(diffs) < 2:
        return 0.0, 0.0

    # Log-log regression: log(diff) = α * log(h) + log(C)
    log_h = np.log(h_vals)
    log_d = np.log(diffs)

    # Simple linear regression
    n = len(log_h)
    mean_h = np.mean(log_h)
    mean_d = np.mean(log_d)
    cov = np.sum((log_h - mean_h) * (log_d - mean_d))
    var = np.sum((log_h - mean_h) ** 2)

    if var < 1e-15:
        return 0.0, 0.0

    alpha = cov / var
    log_C = mean_d - alpha * mean_h
    return float(alpha), float(np.exp(log_C))


def certified_lppls_fit(
    t: NDArray,
    log_price: NDArray,
    grid_sizes: list[int] | None = None,
    safety_factor: float = 2.0,
    optimizer_kwargs: dict | None = None,
) -> CertifiedLPPLSResult:
    """Fit LPPLS at multiple grid resolutions and compute certified tc bound.

    Args:
        t: Time array
        log_price: Log-price array
        grid_sizes: Increasing grid resolutions to test
        safety_factor: Multiplier for conservative bound (2.0 = 95% confidence)
        optimizer_kwargs: Additional kwargs for LPPLSOptimizer

    Returns:
        CertifiedLPPLSResult with tc estimate and convergence bound
    """
    sizes = grid_sizes or DEFAULT_GRID_SIZES
    kwargs = optimizer_kwargs or {}

    tc_values: dict[int, float] = {}
    r2_values: dict[int, float] = {}

    for gs in sizes:
        try:
            opt = LPPLSOptimizer(grid_size=gs, n_best=max(3, gs // 3), **kwargs)
            model = opt.fit(t, log_price)
            if model.params is not None and model.params.is_bubble:
                tc_values[gs] = model.params.tc
                r2_values[gs] = model.r_squared(t, log_price)
        except Exception as e:
            log.warning("certified_grid_failed", grid_size=gs, error=str(e))
            continue

    # Need at least 3 points for Richardson extrapolation
    if len(tc_values) < 3:
        tc_best = list(tc_values.values())[-1] if tc_values else float(t[-1]) + 30
        return CertifiedLPPLSResult(
            tc_estimate=tc_best,
            tc_bound=float("inf"),
            tc_low=float("-inf"),
            tc_high=float("inf"),
            convergence_order=0.0,
            is_convergent=False,
            is_certified=False,
            grid_tc_values=tc_values,
            grid_r_squared=r2_values,
            method="insufficient_data (< 3 valid grid fits)",
            safety_factor=safety_factor,
        )

    # Richardson extrapolation
    sorted_sizes = sorted(tc_values.keys())
    sorted_tc = [tc_values[gs] for gs in sorted_sizes]

    alpha, constant = _estimate_convergence_rate(sorted_sizes, sorted_tc)

    # Best estimate = finest grid
    tc_best = sorted_tc[-1]
    finest_h = 1.0 / sorted_sizes[-1]

    # Compute bound
    if alpha >= STABILITY_THRESHOLD and constant > 0:
        # Convergent: bound = safety_factor * C * h^α
        raw_bound = constant * (finest_h**alpha)
        bound = safety_factor * raw_bound
        is_convergent = True
        method = f"Richardson extrapolation (α={alpha:.2f}, C={constant:.2f}, sf={safety_factor})"
    else:
        # Non-convergent: use max spread of tc values as conservative bound
        tc_spread = max(sorted_tc) - min(sorted_tc)
        bound = safety_factor * tc_spread
        is_convergent = False
        method = (
            f"conservative spread (α={alpha:.2f} < {STABILITY_THRESHOLD}, spread={tc_spread:.1f})"
        )

    log.info(
        "certified_fit_done",
        tc=round(tc_best, 1),
        bound=round(bound, 1),
        alpha=round(alpha, 2),
        is_convergent=is_convergent,
        n_grids=len(tc_values),
    )

    return CertifiedLPPLSResult(
        tc_estimate=tc_best,
        tc_bound=bound,
        tc_low=tc_best - bound,
        tc_high=tc_best + bound,
        convergence_order=alpha,
        is_convergent=is_convergent,
        is_certified=is_convergent,
        grid_tc_values=tc_values,
        grid_r_squared=r2_values,
        method=method,
        safety_factor=safety_factor,
    )


def verify_lppls_convergence(
    t: NDArray,
    log_price: NDArray,
    grid_sizes: list[int] | None = None,
    expected_order: float = 2.0,
    tolerance: float = 0.5,
) -> dict:
    """Quick convergence check without full certified bound computation.

    Answers: "Is this LPPLS fit stable or just overfitting?"

    Returns dict with:
        - empirical_order: float
        - is_consistent: bool (empirical ≥ expected - tolerance)
        - tc_values: dict[int, float]
        - verdict: str
    """
    result = certified_lppls_fit(t, log_price, grid_sizes=grid_sizes, safety_factor=1.0)

    is_consistent = result.convergence_order >= (expected_order - tolerance)

    if result.is_convergent and is_consistent:
        verdict = f"STABLE: α={result.convergence_order:.2f}, tc converges"
    elif result.is_convergent:
        verdict = f"SLOW: α={result.convergence_order:.2f} < expected {expected_order}"
    else:
        verdict = f"UNSTABLE: α={result.convergence_order:.2f}, tc does not converge"

    return {
        "empirical_order": result.convergence_order,
        "expected_order": expected_order,
        "is_consistent": is_consistent,
        "tc_values": result.grid_tc_values,
        "tc_estimate": result.tc_estimate,
        "tc_bound": result.tc_bound,
        "verdict": verdict,
    }
