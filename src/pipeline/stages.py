"""PhaseBreak pipeline separated into 3 layers.

Layer A — Screening: data quality, HMM regime, baseline indicators
Layer B — Structural Fit: LPPLS fit, soft scoring, multi-window, tc interval
Layer C — Scientific Inference: cross-domain stats, ablation (NOT in online path)

Operational verdict comes from A+B only. C is offline analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import structlog
from numpy.typing import NDArray

from src.lppls.optimizer import LPPLSOptimizer
from src.lppls.model import LPPLSParams

log = structlog.get_logger()


@dataclass
class ScreeningResult:
    """Layer A output."""

    data_quality: str  # "OK" / "LOW_N" / "MISSING"
    n_points: int
    hmm_regime: str | None  # "NORMAL" / "GROWTH" / "BUBBLE"
    hmm_bubble_prob: float
    should_fit_lppls: bool
    reason: str


@dataclass
class StructuralFitResult:
    """Layer B output."""

    params: LPPLSParams | None
    r_squared: float
    quality_score: float  # soft filter score 0-1
    is_bubble: bool
    tc_estimate: float | None
    tc_lower: float | None  # uncertainty band
    tc_upper: float | None
    multi_window_confidence: str | None  # HIGH/MEDIUM/LOW/NO_SIGNAL
    verdict: str  # BUBBLE / POSSIBLE / NO_BUBBLE


@dataclass
class PipelineResult:
    """Combined A+B result (operational)."""

    screening: ScreeningResult
    fit: StructuralFitResult | None  # None if screening says skip
    final_verdict: str
    final_confidence: float


def run_screening(t: NDArray, values: NDArray, min_points: int = 20) -> ScreeningResult:
    """Layer A: screen data before expensive LPPLS fit."""
    n = len(t)
    if n < min_points:
        return ScreeningResult(
            data_quality="LOW_N",
            n_points=n,
            hmm_regime=None,
            hmm_bubble_prob=0.0,
            should_fit_lppls=False,
            reason=f"Too few points: {n} < {min_points}",
        )

    # Check for NaN/inf
    valid = np.isfinite(values)
    if valid.sum() < min_points:
        return ScreeningResult(
            data_quality="MISSING",
            n_points=int(valid.sum()),
            hmm_regime=None,
            hmm_bubble_prob=0.0,
            should_fit_lppls=False,
            reason=f"Too many invalid values: {n - valid.sum()}/{n}",
        )

    # HMM regime (optional — fallback to heuristic if unavailable)
    try:
        from src.lppls.regime import HMMRegimeDetector

        hmm = HMMRegimeDetector()
        regime_result = hmm.fit_predict(np.log(np.clip(values, 1e-10, None)))
        hmm_regime = regime_result.current_regime.name
        hmm_prob = regime_result.bubble_probability
        should_fit = regime_result.should_fit_lppls
    except Exception:
        hmm_regime = None
        hmm_prob = 0.5
        should_fit = True  # conservative: fit if HMM fails

    return ScreeningResult(
        data_quality="OK",
        n_points=n,
        hmm_regime=hmm_regime,
        hmm_bubble_prob=hmm_prob,
        should_fit_lppls=should_fit,
        reason="passed screening",
    )


def run_structural_fit(
    t: NDArray,
    log_price: NDArray,
    grid_size: int = 12,
    m_range: tuple = (0.1, 0.9),
    omega_range: tuple = (6.0, 13.0),
) -> StructuralFitResult:
    """Layer B: LPPLS fit + soft scoring + tc uncertainty."""
    from src.lppls.scoring import compute_quality_score
    from src.lppls.uncertainty import bootstrap_tc_uncertainty

    opt = LPPLSOptimizer(grid_size=grid_size, n_best=5, m_range=m_range, omega_range=omega_range)
    model = opt.fit(t, log_price)
    r2 = model.r_squared(t, log_price)
    params = model.params

    # Soft scoring
    quality = compute_quality_score(params, r2) if params else 0.0

    # tc uncertainty
    tc_est = params.tc if params else None
    tc_lo = tc_hi = None
    if params and params.is_bubble and r2 > 0.3:
        unc = bootstrap_tc_uncertainty(
            t, log_price, n_bootstrap=20, m_range=m_range, omega_range=omega_range
        )
        tc_lo = unc.get("tc_p10")
        tc_hi = unc.get("tc_p90")

    is_bubble = params is not None and params.is_bubble and r2 > 0.5 and quality > 0.3

    if is_bubble and quality > 0.6:
        verdict = "BUBBLE"
    elif is_bubble or quality > 0.4:
        verdict = "POSSIBLE"
    else:
        verdict = "NO_BUBBLE"

    return StructuralFitResult(
        params=params,
        r_squared=r2,
        quality_score=quality,
        is_bubble=is_bubble,
        tc_estimate=tc_est,
        tc_lower=tc_lo,
        tc_upper=tc_hi,
        multi_window_confidence=None,
        verdict=verdict,
    )


def run_full_pipeline(t: NDArray, values: NDArray, **fit_kwargs) -> PipelineResult:
    """Run Layer A + Layer B. Layer C is offline only."""
    log_price = np.log(np.clip(values, 1e-10, None))

    screening = run_screening(t, values)

    if not screening.should_fit_lppls:
        return PipelineResult(
            screening=screening,
            fit=None,
            final_verdict="NO_BUBBLE",
            final_confidence=0.0,
        )

    fit = run_structural_fit(t, log_price, **fit_kwargs)

    return PipelineResult(
        screening=screening,
        fit=fit,
        final_verdict=fit.verdict,
        final_confidence=fit.quality_score,
    )
