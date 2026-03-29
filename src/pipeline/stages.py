"""PhaseBreak pipeline separated into 3 layers.

Layer A — Screening: data quality, HMM regime, baseline indicators
Layer B — Structural Fit: LPPLS fit, soft scoring, multi-window, tc interval
Layer C — Scientific Inference: cross-domain stats, ablation (NOT in online path)

Two paths available:
  - legacy: original HMM-gated ensemble (ensemble.py)
  - v2 (recommended): screening → adaptive windows → soft scoring → tc uncertainty → HMM-weighted verdict

Operational verdict comes from A+B only. C is offline analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import structlog
from numpy.typing import NDArray

from src.lppls.optimizer import LPPLSOptimizer
from src.lppls.model import LPPLS, LPPLSParams

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


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
    windows_used: list[int] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Combined A+B result (operational)."""

    screening: ScreeningResult
    fit: StructuralFitResult | None  # None if screening says skip
    final_verdict: str
    final_confidence: float
    path: str = "v2"  # "legacy" or "v2"


# ---------------------------------------------------------------------------
# Layer A — Screening
# ---------------------------------------------------------------------------


def run_screening(
    t: NDArray,
    values: NDArray,
    min_points: int = 20,
    domain: str = "finance",
) -> ScreeningResult:
    """Layer A: screen data before expensive LPPLS fit.

    Domain-aware HMM gating:
    - finance (daily, 200+ points): full HMM gating
    - commodities (daily, but different dynamics): relaxed threshold
    - housing (quarterly, <50 points): skip HMM, always fit
    """
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

    # WHY: HMM gating is unreliable for short series (<50 points, e.g. quarterly
    # housing) and for commodity markets where HMM often misclassifies bubbles
    # as NORMAL (different return/vol dynamics vs equities).
    skip_hmm = n < 50 or domain == "housing"

    if skip_hmm:
        return ScreeningResult(
            data_quality="OK",
            n_points=n,
            hmm_regime=None,
            hmm_bubble_prob=0.5,
            should_fit_lppls=True,
            reason=f"HMM skipped (domain={domain}, n={n}): always fit",
        )

    # HMM regime (optional — fallback to heuristic if unavailable)
    try:
        from src.lppls.regime import HMMRegimeDetector

        # WHY: commodities need lower bubble threshold — HMM trained on equity
        # features often classifies commodity bubbles as NORMAL
        bubble_threshold = 0.3 if domain == "commodities" else 0.5
        hmm = HMMRegimeDetector(bubble_threshold=bubble_threshold)
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


# ---------------------------------------------------------------------------
# Layer B — Structural Fit (v2: adaptive windows + soft scoring + uncertainty)
# ---------------------------------------------------------------------------


def run_structural_fit(
    t: NDArray,
    log_price: NDArray,
    grid_size: int = 12,
    m_range: tuple = (0.1, 0.9),
    omega_range: tuple = (6.0, 13.0),
    frequency: str = "daily",
    hmm_bubble_prob: float = 0.5,
    use_adaptive_windows: bool = True,
    n_bootstrap: int = 20,
    domain: str = "finance",
) -> StructuralFitResult:
    """Layer B: LPPLS fit + soft scoring + tc uncertainty.

    v2 enhancements over legacy:
    - Adaptive window selection based on data frequency/volatility
    - HMM bubble_prob used as prior weight on verdict confidence
    - Bootstrap tc uncertainty intervals
    - Domain-specific soft scoring weights
    """
    from src.lppls.scoring import compute_quality_score
    from src.lppls.uncertainty import bootstrap_tc_uncertainty

    # v2: adaptive windows
    windows_used: list[int] = []
    if use_adaptive_windows:
        from src.lppls.windowing import select_adaptive_windows

        raw_prices = np.exp(log_price)
        windows_used = select_adaptive_windows(raw_prices, frequency=frequency)

    # Fit LPPLS
    opt = LPPLSOptimizer(grid_size=grid_size, n_best=5, m_range=m_range, omega_range=omega_range)
    model = opt.fit(t, log_price)
    r2 = model.r_squared(t, log_price)
    params = model.params

    # Domain-specific soft scoring
    quality = compute_quality_score(params, r2, domain=domain) if params else 0.0

    # v2: HMM prior weight — boost/penalize quality based on HMM agreement
    # WHY: HMM seeing BUBBLE + good LPPLS fit = stronger signal
    # HMM seeing NORMAL + marginal LPPLS fit = likely false positive
    # For housing (HMM skipped), hmm_bubble_prob=0.5 → weight=0.85 (neutral)
    hmm_weight = 0.7 + 0.3 * hmm_bubble_prob  # range [0.7, 1.0]
    weighted_quality = quality * hmm_weight

    # tc uncertainty via bootstrap
    tc_est = params.tc if params else None
    tc_lo = tc_hi = None
    if params and params.is_bubble and r2 > 0.3:
        unc = bootstrap_tc_uncertainty(
            t, log_price, n_bootstrap=n_bootstrap, m_range=m_range, omega_range=omega_range
        )
        tc_lo = unc.get("tc_p10")
        tc_hi = unc.get("tc_p90")

    # WHY: housing quarterly data produces more false positives due to fewer
    # data points → LPPLS fits noise more easily. Raise threshold for housing.
    bubble_threshold = 0.4 if domain == "housing" else 0.3
    possible_threshold = 0.5 if domain == "housing" else 0.4

    is_bubble = (
        params is not None and params.is_bubble and r2 > 0.5 and weighted_quality > bubble_threshold
    )

    # Verdict with domain-aware thresholds
    if is_bubble and weighted_quality > 0.6:
        verdict = "BUBBLE"
    elif is_bubble or weighted_quality > possible_threshold:
        verdict = "POSSIBLE"
    else:
        verdict = "NO_BUBBLE"

    return StructuralFitResult(
        params=params,
        r_squared=r2,
        quality_score=float(weighted_quality),
        is_bubble=is_bubble,
        tc_estimate=tc_est,
        tc_lower=tc_lo,
        tc_upper=tc_hi,
        multi_window_confidence=None,
        verdict=verdict,
        windows_used=windows_used,
    )


# ---------------------------------------------------------------------------
# Full v2 pipeline (recommended)
# ---------------------------------------------------------------------------


def run_full_pipeline(
    t: NDArray,
    values: NDArray,
    frequency: str = "daily",
    use_adaptive_windows: bool = True,
    n_bootstrap: int = 20,
    domain: str = "finance",
    **fit_kwargs,
) -> PipelineResult:
    """Run Layer A + Layer B (v2 path). Layer C is offline only.

    This is the recommended v2 entry point. For legacy path, use
    src.lppls.ensemble.HMMLPPLSEnsemble.analyze() directly.

    Args:
        domain: "finance", "commodities", "housing", "geology", "adversarial"
            Controls HMM gating aggressiveness and scoring weights.
    """
    log_price = np.log(np.clip(values, 1e-10, None))

    screening = run_screening(t, values, domain=domain)

    if not screening.should_fit_lppls:
        return PipelineResult(
            screening=screening,
            fit=None,
            final_verdict="NO_BUBBLE",
            final_confidence=0.0,
            path="v2",
        )

    fit = run_structural_fit(
        t,
        log_price,
        frequency=frequency,
        hmm_bubble_prob=screening.hmm_bubble_prob,
        use_adaptive_windows=use_adaptive_windows,
        n_bootstrap=n_bootstrap,
        domain=domain,
        **fit_kwargs,
    )

    return PipelineResult(
        screening=screening,
        fit=fit,
        final_verdict=fit.verdict,
        final_confidence=fit.quality_score,
        path="v2",
    )


# ---------------------------------------------------------------------------
# Legacy pipeline (backward compatibility)
# ---------------------------------------------------------------------------


def run_legacy_pipeline(t: NDArray, values: NDArray) -> PipelineResult:
    """Legacy path: original HMM-gated ensemble without v2 enhancements.

    Preserved for backward compatibility and ablation comparison.
    Uses hard filters, fixed windows, no uncertainty intervals.
    """
    log_price = np.log(np.clip(values, 1e-10, None))

    screening = run_screening(t, values)

    if not screening.should_fit_lppls:
        return PipelineResult(
            screening=screening,
            fit=None,
            final_verdict="NO_BUBBLE",
            final_confidence=0.0,
            path="legacy",
        )

    # Legacy fit: no adaptive windows, no HMM prior weight, no bootstrap
    opt = LPPLSOptimizer(grid_size=12, n_best=5)
    model = opt.fit(t, log_price)
    r2 = model.r_squared(t, log_price)
    params = model.params

    is_bubble = params is not None and params.is_bubble and r2 > 0.5

    verdict = "BUBBLE" if is_bubble else "NO_BUBBLE"

    fit = StructuralFitResult(
        params=params,
        r_squared=r2,
        quality_score=1.0 if is_bubble else 0.0,  # legacy: binary
        is_bubble=is_bubble,
        tc_estimate=params.tc if params else None,
        tc_lower=None,
        tc_upper=None,
        multi_window_confidence=None,
        verdict=verdict,
    )

    return PipelineResult(
        screening=screening,
        fit=fit,
        final_verdict=verdict,
        final_confidence=1.0 if is_bubble else 0.0,
        path="legacy",
    )
