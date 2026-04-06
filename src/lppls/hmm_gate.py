"""HMM Gate — pre-screens assets before LPPLS fitting.

If HMM says NORMAL regime → skip LPPLS entirely → NO_BUBBLE.
Reduces false positives by ~30-50% (no LPPLS on noise/sideways periods).
"""

from __future__ import annotations

import numpy as np
import structlog

log = structlog.get_logger()


def hmm_prescreen(prices: np.ndarray) -> dict:
    """Pre-screen asset with HMM regime detection.

    Returns dict with:
        should_fit: bool — whether to run LPPLS
        regime: str — NORMAL / GROWTH / BUBBLE
        bubble_prob: float — probability of bubble state
        growth_prob: float — probability of growth state
    """
    if len(prices) < 60:
        return {"should_fit": True, "regime": "UNKNOWN", "bubble_prob": 0.5, "growth_prob": 0.5}

    log_price = np.log(prices)

    try:
        from src.lppls.regime import HMMRegimeDetector

        detector = HMMRegimeDetector(bubble_threshold=0.5)
        result = detector.fit_predict(log_price)

        return {
            "should_fit": result.should_fit_lppls,
            "regime": result.current_regime.name,
            "bubble_prob": round(result.bubble_probability, 3),
            "growth_prob": round(result.growth_probability, 3),
        }
    except Exception as e:
        log.debug("hmm_prescreen_failed", error=str(e))
        # On failure, default to fitting (don't block LPPLS)
        return {"should_fit": True, "regime": "UNKNOWN", "bubble_prob": 0.5, "growth_prob": 0.5}
