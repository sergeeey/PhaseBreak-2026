"""Hurst Exponent — independent trend persistence signal.

H > 0.7: persistent trend (supports bubble hypothesis)
H ≈ 0.5: random walk (weakens bubble hypothesis)
H < 0.3: mean-reverting (anti-bubble)

Uses R/S analysis. Fast, no fitting — pure statistical measure.
"""

from __future__ import annotations

import numpy as np
import structlog

log = structlog.get_logger()


def compute_hurst(prices: np.ndarray, min_window: int = 20) -> float:
    """Compute Hurst exponent via R/S analysis.

    Returns H in [0, 1]. Returns 0.5 on failure (neutral).
    """
    if len(prices) < min_window * 2:
        return 0.5

    try:
        from hurst import compute_Hc

        H, _, _ = compute_Hc(prices, kind="price", simplified=True)
        H = float(np.clip(H, 0.0, 1.5))  # clip artifacts
        return round(H, 4)
    except Exception as e:
        log.debug("hurst_failed", error=str(e))
        return 0.5


def hurst_verdict(H: float) -> str:
    """Interpret Hurst exponent.

    Returns: PERSISTENT | RANDOM_WALK | MEAN_REVERTING
    """
    if H > 0.7:
        return "PERSISTENT"
    elif H < 0.35:
        return "MEAN_REVERTING"
    else:
        return "RANDOM_WALK"


def hurst_supports_bubble(H: float) -> bool:
    """Does Hurst exponent support bubble hypothesis?

    H > 0.65 means the trend is self-reinforcing — consistent with bubble dynamics.
    """
    return H > 0.65
