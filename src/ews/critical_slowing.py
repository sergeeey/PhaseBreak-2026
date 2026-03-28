"""Critical Slowing Down — model-independent early warning signals.

Before a critical transition, systems exhibit:
1. Rising variance (fluctuations grow)
2. Rising lag-1 autocorrelation (recovery slows)
3. Rising skewness (asymmetry increases)

These are universal physical precursors (Scheffer 2009, Dakos 2012),
independent of LPPLS parametric assumptions.

References:
    Scheffer et al. (2009) "Early-warning signals for critical transitions" Nature
    Dakos et al. (2012) "Methods for detecting early warnings" PLoS ONE
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import structlog
from numpy.typing import NDArray
from scipy.stats import kendalltau

log = structlog.get_logger()


@dataclass
class EWSResult:
    """Early Warning Signal analysis result."""

    # Rolling features
    rolling_variance: NDArray
    rolling_ac1: NDArray  # lag-1 autocorrelation
    rolling_skewness: NDArray
    recovery_rate: NDArray  # 1 / AR(1) coefficient

    # Trend statistics (Kendall tau)
    kendall_tau_variance: float  # >0 = variance increasing → approaching transition
    kendall_tau_ac1: float  # >0 = autocorrelation increasing → slowing down
    kendall_tau_skewness: float
    kendall_p_variance: float
    kendall_p_ac1: float

    # Composite
    ews_score: float  # 0-1, weighted composite
    is_slowing: bool  # True if both variance and AC1 trends are positive and strong

    # Metadata
    window_size: int
    n_points: int


def _rolling_stat(series: NDArray, window: int, func) -> NDArray:
    """Compute rolling statistic with given window."""
    n = len(series)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        chunk = series[i - window + 1 : i + 1]
        valid = chunk[np.isfinite(chunk)]
        if len(valid) >= window // 2:
            result[i] = func(valid)
    return result


def _rolling_variance(series: NDArray, window: int) -> NDArray:
    return _rolling_stat(series, window, np.var)


def _rolling_ac1(series: NDArray, window: int) -> NDArray:
    """Rolling lag-1 autocorrelation."""

    def ac1(x):
        if len(x) < 3:
            return 0.0
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])

    return _rolling_stat(series, window, ac1)


def _rolling_skewness(series: NDArray, window: int) -> NDArray:
    def skew(x):
        m = np.mean(x)
        s = np.std(x)
        if s < 1e-15:
            return 0.0
        return float(np.mean(((x - m) / s) ** 3))

    return _rolling_stat(series, window, skew)


def _detrend(series: NDArray, method: str = "linear") -> NDArray:
    """Remove trend before EWS analysis (Dakos 2012 recommendation)."""
    if method == "linear":
        t = np.arange(len(series), dtype=float)
        mask = np.isfinite(series)
        if mask.sum() < 3:
            return series.copy()
        coeffs = np.polyfit(t[mask], series[mask], deg=1)
        trend = np.polyval(coeffs, t)
        return series - trend
    elif method == "none":
        return series.copy()
    else:
        raise ValueError(f"Unknown detrend method: {method}")


def compute_ews(
    series: NDArray,
    window: int = 50,
    detrend: str = "linear",
) -> EWSResult:
    """Compute Early Warning Signals for critical slowing down.

    Args:
        series: Time series (e.g., log-price, spectral index)
        window: Rolling window size
        detrend: Detrending method ("linear" or "none")

    Returns:
        EWSResult with rolling features and trend statistics
    """
    # Detrend
    residuals = _detrend(series, method=detrend)

    # Rolling features
    r_var = _rolling_variance(residuals, window)
    r_ac1 = _rolling_ac1(residuals, window)
    r_skew = _rolling_skewness(residuals, window)

    # Recovery rate = 1 / AC1 (high AC1 → slow recovery)
    r_recovery = np.where(np.abs(r_ac1) > 0.01, 1.0 / np.abs(r_ac1), np.nan)

    # Kendall tau trends (on non-NaN portions)
    def _kendall(arr: NDArray) -> tuple[float, float]:
        valid_mask = np.isfinite(arr)
        if valid_mask.sum() < 5:
            return 0.0, 1.0
        idx = np.arange(len(arr))[valid_mask]
        vals = arr[valid_mask]
        tau, p = kendalltau(idx, vals)
        return float(tau), float(p)

    tau_var, p_var = _kendall(r_var)
    tau_ac1, p_ac1 = _kendall(r_ac1)
    tau_skew, _ = _kendall(r_skew)

    # Composite EWS score (0-1)
    # Normalize taus from [-1, 1] to [0, 1]
    norm_tau_var = (tau_var + 1) / 2
    norm_tau_ac1 = (tau_ac1 + 1) / 2
    norm_tau_skew = (tau_skew + 1) / 2
    ews_score = 0.5 * norm_tau_var + 0.3 * norm_tau_ac1 + 0.2 * norm_tau_skew
    ews_score = float(np.clip(ews_score, 0, 1))

    # Decision: is system slowing down?
    is_slowing = tau_var > 0.3 and tau_ac1 > 0.3

    log.info(
        "ews_computed",
        tau_var=round(tau_var, 3),
        tau_ac1=round(tau_ac1, 3),
        ews_score=round(ews_score, 3),
        is_slowing=is_slowing,
    )

    return EWSResult(
        rolling_variance=r_var,
        rolling_ac1=r_ac1,
        rolling_skewness=r_skew,
        recovery_rate=r_recovery,
        kendall_tau_variance=tau_var,
        kendall_tau_ac1=tau_ac1,
        kendall_tau_skewness=tau_skew,
        kendall_p_variance=p_var,
        kendall_p_ac1=p_ac1,
        ews_score=ews_score,
        is_slowing=is_slowing,
        window_size=window,
        n_points=len(series),
    )
