"""Soft filter scoring for LPPLS fits.

Converts hard pass/fail Sornette filters into continuous quality scores.
Works in two modes: strict_hard_filters (backward compat) and soft_scoring.
"""

from __future__ import annotations

import numpy as np
from src.lppls.model import LPPLSParams


def score_m(m: float) -> float:
    """Score for power law exponent m. Peak at 0.4-0.6, drops near boundaries."""
    if m <= 0.1 or m >= 0.9:
        return 0.0
    center = 0.5
    return float(np.exp(-(((m - center) / 0.25) ** 2)))


def score_B(B: float) -> float:
    """Score for amplitude B. Must be negative, stronger = better."""
    if B >= 0:
        return 0.0
    absB = abs(B)
    if absB < 0.003:
        return 0.1
    return float(min(1.0, absB / 0.5))


def score_omega(omega: float) -> float:
    """Score for log-frequency. Peak in Sornette range 6-13."""
    if omega < 4 or omega > 25:
        return 0.0
    if 6 <= omega <= 13:
        return 1.0
    if omega < 6:
        return float((omega - 4) / 2)
    return float(max(0, 1 - (omega - 13) / 12))


def score_damping(damping: float) -> float:
    """Score for damping ratio |B|/|C|. Higher = better."""
    if damping < 0.5:
        return float(damping / 0.5)
    return min(1.0, 0.5 + damping / 10)


def score_r_squared(r2: float) -> float:
    """Score for fit quality R²."""
    if r2 < 0:
        return 0.0
    if r2 > 0.95:
        return 1.0
    return float(max(0, (r2 - 0.3) / 0.65))


def compute_quality_score(
    params: LPPLSParams | None,
    r_squared: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute weighted quality score for an LPPLS fit.

    Returns 0-1 score. Higher = more likely real phase transition.
    """
    if params is None:
        return 0.0

    w = weights or {
        "m": 0.20,
        "B": 0.25,
        "omega": 0.15,
        "damping": 0.15,
        "r2": 0.25,
    }

    scores = {
        "m": score_m(params.m),
        "B": score_B(params.B),
        "omega": score_omega(params.omega),
        "damping": score_damping(params.damping),
        "r2": score_r_squared(r_squared),
    }

    total = sum(w[k] * scores[k] for k in w)
    return float(np.clip(total, 0, 1))


def get_score_breakdown(params: LPPLSParams, r_squared: float) -> dict[str, float]:
    """Get individual scores for diagnostics."""
    return {
        "score_m": score_m(params.m),
        "score_B": score_B(params.B),
        "score_omega": score_omega(params.omega),
        "score_damping": score_damping(params.damping),
        "score_r2": score_r_squared(r_squared),
        "quality_total": compute_quality_score(params, r_squared),
    }
