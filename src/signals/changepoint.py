"""Changepoint detection as independent signal layer.

Simple offline changepoint detector based on CUSUM and
variance-ratio statistics. Used as diagnostic signal
alongside LPPLS, not as replacement.

Statuses:
  LPPLS_ONLY:    LPPLS detects, changepoint does not
  CP_ONLY:       Changepoint detects, LPPLS does not
  LPPLS_AND_CP:  Both agree — strongest signal
  NO_SIGNAL:     Neither detects anything
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray

log = structlog.get_logger()


@dataclass
class ChangepointResult:
    """Changepoint detection output."""

    strength: float  # 0-1, how strong the changepoint signal is
    location: int | None  # index of most likely changepoint
    cusum_max: float  # max CUSUM statistic
    variance_ratio: float  # max variance ratio across split points
    is_changepoint: bool  # strength > threshold


def _cusum(series: NDArray) -> NDArray:
    """Cumulative sum of deviations from mean."""
    mean = np.mean(series)
    return np.cumsum(series - mean)


def _variance_ratio(series: NDArray, min_segment: int = 10) -> tuple[float, int]:
    """Find split point maximizing variance difference between segments.

    Returns (max_ratio, best_split_index).
    """
    n = len(series)
    if n < 2 * min_segment:
        return 0.0, n // 2

    best_ratio = 0.0
    best_idx = n // 2

    for i in range(min_segment, n - min_segment):
        left_var = np.var(series[:i])
        right_var = np.var(series[i:])
        if left_var < 1e-15:
            ratio = right_var * 1e10 if right_var > 1e-15 else 0.0
        else:
            ratio = right_var / left_var
        # Take max of ratio and its inverse (detect both increase and decrease)
        ratio = max(ratio, 1.0 / ratio if ratio > 1e-15 else 0.0)
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i

    return float(best_ratio), best_idx


def detect_changepoint_strength(series: NDArray, threshold: float = 0.6) -> float:
    """Compute changepoint strength score (0-1).

    Combines CUSUM max deviation and variance ratio into single score.
    """
    if len(series) < 20:
        return 0.0

    log_series = np.log(np.clip(series, 1e-10, None))
    returns = np.diff(log_series)

    # CUSUM on returns
    cs = _cusum(returns)
    cusum_range = np.max(cs) - np.min(cs)
    cusum_norm = cusum_range / (np.std(returns) * np.sqrt(len(returns)) + 1e-10)
    cusum_score = min(1.0, cusum_norm / 3.0)  # normalize: 3σ → score=1

    # Variance ratio
    vr, _ = _variance_ratio(returns)
    vr_score = min(1.0, (vr - 1) / 4.0)  # ratio=5 → score=1

    strength = 0.6 * cusum_score + 0.4 * vr_score
    return float(np.clip(strength, 0, 1))


def detect_recent_changepoint(
    series: NDArray,
    threshold: float = 0.5,
) -> ChangepointResult:
    """Detect if there's a structural changepoint in the series.

    Returns ChangepointResult with strength, location, and diagnostics.
    """
    if len(series) < 20:
        return ChangepointResult(
            strength=0.0,
            location=None,
            cusum_max=0.0,
            variance_ratio=0.0,
            is_changepoint=False,
        )

    log_series = np.log(np.clip(series, 1e-10, None))
    returns = np.diff(log_series)

    # CUSUM
    cs = _cusum(returns)
    cusum_max = float(np.max(np.abs(cs)))

    # Variance ratio
    vr, vr_idx = _variance_ratio(returns)

    # Combined strength
    strength = detect_changepoint_strength(series, threshold)

    # Location: where CUSUM peaks
    cp_location = int(np.argmax(np.abs(cs))) + 1  # +1 for original series index

    log.info(
        "changepoint_detected",
        strength=round(strength, 3),
        location=cp_location,
        cusum_max=round(cusum_max, 3),
        vr=round(vr, 2),
        is_cp=strength > threshold,
    )

    return ChangepointResult(
        strength=strength,
        location=cp_location,
        cusum_max=cusum_max,
        variance_ratio=vr,
        is_changepoint=strength > threshold,
    )
