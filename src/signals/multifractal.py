"""Multifractal Detrended Fluctuation Analysis (MFDFA) for phase transition detection.

Computes multifractal spectrum width Δα. Empirical finding on PhaseBreak data:
- Bubbles have NARROW spectrum: Δα ∈ [0.12, 0.23] (loss of complexity)
- Controls have WIDE spectrum: Δα ∈ [0.43, 0.57] (healthy complexity)

This is orthogonal to LPPLS (different physics: scaling properties vs log-periodic
oscillations). Replaces weak EWS (critical slowing down) as diagnostic layer.

Requires: pip install MFDFA
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray

log = structlog.get_logger()


@dataclass
class MFDFAResult:
    """Multifractal analysis output."""

    delta_alpha: float  # spectrum width (narrow = bubble)
    is_narrow: bool  # Δα < threshold → bubble signal
    h_q: list[float] | None  # generalized Hurst exponents
    bubble_score: float  # 0-1, higher = more likely bubble


def compute_mfdfa_width(
    prices: NDArray,
    lag_start: int = 10,
    lag_end: int = 80,
    n_q: int = 7,
) -> float | None:
    """Compute multifractal spectrum width Δα.

    Returns None if computation fails (too few points, package missing).
    """
    try:
        from MFDFA import MFDFA
    except ImportError:
        log.warning("mfdfa_not_installed", msg="pip install MFDFA")
        return None

    log_prices = np.log(np.clip(prices, 1e-10, None))

    if len(log_prices) < lag_end * 2:
        lag_end = len(log_prices) // 3

    if lag_end <= lag_start:
        return None

    lag = np.unique(np.logspace(np.log10(lag_start), np.log10(lag_end), 20).astype(int))
    q = np.linspace(-3.0, 3.0, n_q)

    try:
        lag_out, dfa = MFDFA(log_prices, lag=lag, q=q, order=1)
    except Exception:
        return None

    n_q_actual = dfa.shape[1]
    h_q = []
    for i in range(n_q_actual):
        valid = (dfa[:, i] > 0) & (lag_out > 0)
        if valid.sum() < 3:
            h_q.append(0.5)
        else:
            c = np.polyfit(np.log(lag_out[valid]), np.log(dfa[valid, i]), 1)
            h_q.append(c[0])

    h_q_arr = np.array(h_q)
    q_used = q[:n_q_actual]
    tau = q_used * h_q_arr - 1
    alpha = np.gradient(tau, q_used)

    delta_alpha = float(alpha.max() - alpha.min())
    return delta_alpha


def analyze_multifractal(
    prices: NDArray,
    narrow_threshold: float = 0.3,
) -> MFDFAResult:
    """Full MFDFA analysis with bubble scoring.

    Empirical thresholds from PhaseBreak benchmark:
    - Δα < 0.25: strong bubble signal (all known bubbles)
    - Δα ∈ [0.25, 0.35]: borderline
    - Δα > 0.40: normal (all controls)
    """
    delta_alpha = compute_mfdfa_width(prices)

    if delta_alpha is None:
        return MFDFAResult(
            delta_alpha=0.0,
            is_narrow=False,
            h_q=None,
            bubble_score=0.0,
        )

    is_narrow = delta_alpha < narrow_threshold

    # Score: map Δα to 0-1 (lower Δα → higher score)
    if delta_alpha < 0.15:
        bubble_score = 1.0
    elif delta_alpha < narrow_threshold:
        bubble_score = 0.7 + 0.3 * (narrow_threshold - delta_alpha) / (narrow_threshold - 0.15)
    elif delta_alpha < 0.4:
        bubble_score = 0.3 + 0.4 * (0.4 - delta_alpha) / (0.4 - narrow_threshold)
    else:
        bubble_score = max(0.0, 0.3 * (0.6 - delta_alpha) / 0.2)

    bubble_score = float(np.clip(bubble_score, 0, 1))

    log.info(
        "mfdfa_analysis",
        delta_alpha=round(delta_alpha, 3),
        is_narrow=is_narrow,
        bubble_score=round(bubble_score, 3),
    )

    return MFDFAResult(
        delta_alpha=round(delta_alpha, 4),
        is_narrow=is_narrow,
        h_q=None,
        bubble_score=bubble_score,
    )
