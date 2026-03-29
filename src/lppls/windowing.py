"""Adaptive multi-scale window selection for LPPLS fitting.

Instead of fixed windows [60, 90, 120, 180], selects windows
based on data frequency and volatility regime.

Rules:
  daily + high-vol → shorter windows [30, 45, 60, 90]
  daily + normal   → standard [60, 90, 120, 180]
  quarterly        → [8, 12, 16, 20] quarters
"""

from __future__ import annotations

import numpy as np
import structlog
from numpy.typing import NDArray

log = structlog.get_logger()

WINDOW_PRESETS: dict[str, list[int]] = {
    "daily_highvol": [30, 45, 60, 90],
    "daily_normal": [60, 90, 120, 180],
    "daily_long": [90, 120, 180, 252],
    "quarterly": [8, 12, 16, 20],
    "monthly": [12, 18, 24, 36],
}


def infer_volatility_regime(series: NDArray, threshold: float = 1.5) -> str:
    """Classify series volatility as 'high' or 'normal'.

    Compares recent volatility (last 20%) to historical.
    If recent > threshold × historical → 'high'.
    """
    if len(series) < 20:
        return "normal"

    log_returns = np.diff(np.log(np.clip(series, 1e-10, None)))
    if len(log_returns) < 10:
        return "normal"

    split = max(5, len(log_returns) // 5)
    recent_vol = np.std(log_returns[-split:])
    historical_vol = np.std(log_returns[:-split])

    if historical_vol < 1e-10:
        return "normal"

    ratio = recent_vol / historical_vol
    regime = "high" if ratio > threshold else "normal"

    log.debug("volatility_regime", ratio=round(ratio, 2), regime=regime)
    return regime


def select_adaptive_windows(
    series: NDArray,
    frequency: str = "daily",
    volatility_regime: str | None = None,
) -> list[int]:
    """Select LPPLS fitting windows adapted to data characteristics.

    Args:
        series: Price/index time series
        frequency: "daily", "monthly", or "quarterly"
        volatility_regime: Override auto-detection ("high" or "normal")

    Returns:
        List of window sizes (in data points, not calendar time)
    """
    if frequency == "quarterly":
        windows = WINDOW_PRESETS["quarterly"]
    elif frequency == "monthly":
        windows = WINDOW_PRESETS["monthly"]
    else:
        # Daily: check volatility
        vol = volatility_regime or infer_volatility_regime(series)
        if vol == "high":
            windows = WINDOW_PRESETS["daily_highvol"]
        else:
            windows = WINDOW_PRESETS["daily_normal"]

    # Filter: no window larger than 80% of data
    max_window = int(len(series) * 0.8)
    windows = [w for w in windows if w <= max_window]

    if not windows:
        # Fallback: use 40%, 60%, 80% of data length
        n = len(series)
        windows = [max(5, int(n * f)) for f in [0.4, 0.6, 0.8]]

    log.info("adaptive_windows", frequency=frequency, windows=windows, n_data=len(series))
    return windows
