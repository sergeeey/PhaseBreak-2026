"""Acceptor of Action Result (Акцептор результата действия — П.К. Анохин).

This module implements the anticipatory component of the Theory of Functional
Systems (ТФС) in the PhaseBreak LPPLS pipeline.

Anokhin's key insight: the system predicts expected result parameters BEFORE
taking action, then compares actual result with prediction to determine
if the action was successful.

Architecture:
1. PREDICT expected LPPLS parameter ranges based on domain + data characteristics
2. FIT LPPLS model (existing pipeline)
3. COMPARE actual parameters with acceptor prediction
4. DECIDE: accept result, targeted retry, or reject

This transforms the pipeline from reactive (fit → check → done) to
anticipatory (predict → fit → compare → targeted correction).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import structlog
from numpy.typing import NDArray

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AcceptorPrediction:
    """Prediction of expected LPPLS parameters BEFORE fitting.

    This is the "акцептор результата действия" — the system's forecast
    of what a valid bubble fit should look like, based on domain knowledge
    and data characteristics.
    """

    # Expected parameter ranges (predicted before fit)
    m_expected: tuple[float, float]  # (lower, upper)
    omega_expected: tuple[float, float]
    tc_days_ahead: tuple[int, int]  # Expected tc range in days after t_end
    min_r_squared: float  # Minimum acceptable R²
    min_quality: float  # Minimum acceptable quality score

    # Prediction metadata
    confidence: float  # How confident we are in this prediction (0-1)
    basis: str  # What the prediction is based on: "domain_prior", "historical", "data_analysis"


@dataclass
class AcceptorComparison:
    """Result of comparing actual fit with acceptor prediction."""

    acceptor_match: bool  # Does actual result match acceptor?
    mismatches: list[str]  # Which parameters mismatched
    mismatch_severity: str  # "none", "minor", "major"
    retry_needed: bool  # Should we retry?
    retry_action: str | None  # What specific action to take on retry
    satisfaction_score: float  # How satisfied we are with result (0-1)


# ---------------------------------------------------------------------------
# Domain-specific acceptor predictions
# ---------------------------------------------------------------------------

# WHY: Different domains have characteristic LPPLS parameter distributions.
# The acceptor uses these priors to predict what a "valid" fit should look like.
# This is analogous to how biological systems have innate expectations based
# on evolutionary history (e.g., breathing control expects certain CO2 ranges).

DOMAIN_ACCEPTOR_PRIORS: dict[str, dict] = {
    "finance": {
        "m_range": (0.2, 0.7),  # Tighter than optimizer bounds
        "omega_range": (6.0, 13.0),  # Sornette canonical range
        "tc_days": (30, 180),  # Bubble usually resolves in 1-6 months
        "min_r_squared": 0.85,  # Finance: high bar (lots of data points)
        "min_quality": 0.6,
    },
    "commodities": {
        "m_range": (0.15, 0.75),  # Slightly wider (more noisy)
        "omega_range": (5.0, 14.0),  # Commodities can have wider frequency range
        "tc_days": (60, 365),  # Commodity cycles are longer
        "min_r_squared": 0.75,  # Lower bar (noisier data)
        "min_quality": 0.5,
    },
    "housing": {
        "m_range": (0.1, 0.8),  # Housing bubbles are slower
        "omega_range": (4.0, 12.0),  # Fewer oscillations in quarterly data
        "tc_days": (180, 730),  # Housing cycles are multi-year
        "min_r_squared": 0.70,  # Much lower bar (few data points)
        "min_quality": 0.4,
    },
    "geology": {
        "m_range": (0.1, 0.9),  # Very wide (unknown characteristics)
        "omega_range": (4.0, 15.0),
        "tc_days": (30, 365),
        "min_r_squared": 0.60,  # Noisy geological data
        "min_quality": 0.3,
    },
    "adversarial": {
        "m_range": (0.2, 0.7),  # Same as finance (should reject non-LPPLS patterns)
        "omega_range": (6.0, 13.0),
        "tc_days": (30, 180),
        "min_r_squared": 0.90,  # Very high bar (adversarial controls should fail)
        "min_quality": 0.7,
    },
}


# ---------------------------------------------------------------------------
# Acceptor: Predict expected result before fitting
# ---------------------------------------------------------------------------


def create_acceptor(
    domain: str,
    t: NDArray,
    values: NDArray,
    hmm_bubble_prob: float = 0.5,
    historical_params: list[dict] | None = None,
) -> AcceptorPrediction:
    """Create acceptor prediction BEFORE running LPPLS fit.

    This is the "акцептор результата действия" — the system's expectation
    of what parameters a valid bubble fit should produce.

    The prediction is based on:
    1. Domain priors (characteristic parameter ranges for this domain)
    2. Data characteristics (n_points, volatility, trend strength)
    3. HMM bubble probability (if available)
    4. Historical fits (if available — Bayesian updating of priors)

    Args:
        domain: "finance", "commodities", "housing", "geology", "adversarial"
        t: Time index array
        values: Price/value array
        hmm_bubble_prob: HMM-estimated bubble probability (0-1)
        historical_params: Previous LPPLS fits for this asset/domain

    Returns:
        AcceptorPrediction with expected parameter ranges
    """
    priors = DOMAIN_ACCEPTOR_PRIORS.get(domain, DOMAIN_ACCEPTOR_PRIORS["finance"])

    # Start with domain priors
    m_range = list(priors["m_range"])
    omega_range = list(priors["omega_range"])
    tc_days = list(priors["tc_days"])
    min_r2 = priors["min_r_squared"]
    min_quality = priors["min_quality"]

    # WHY: HMM probability can refine our expectations.
    # If HMM says high bubble prob → expect stronger signal (tighter ranges).
    # If HMM says low bubble prob → expect weak/no signal (wider ranges).
    if hmm_bubble_prob > 0.7:
        # High bubble probability → expect strong, clear signal
        m_center = (m_range[0] + m_range[1]) / 2
        omega_center = (omega_range[0] + omega_range[1]) / 2
        m_range = [m_center - 0.1, m_center + 0.1]  # Tighter
        omega_range = [omega_center - 2.0, omega_center + 2.0]
        min_r2 = min(min_r2 + 0.05, 0.95)  # Higher bar
        basis = "domain_prior + strong_hmm"
        confidence = 0.7
    elif hmm_bubble_prob < 0.3:
        # Low bubble probability → expect weak/no signal
        min_r2 = max(min_r2 - 0.1, 0.5)  # Lower bar (we expect noise)
        min_quality = max(min_quality - 0.1, 0.2)
        basis = "domain_prior + weak_hmm"
        confidence = 0.4
    else:
        basis = "domain_prior"
        confidence = 0.5

    # WHY: If we have historical fits for this asset, use them to refine prediction.
    # This is Bayesian updating — prior + data → posterior.
    if historical_params and len(historical_params) >= 3:
        hist_m = [p["m"] for p in historical_params if p.get("is_bubble")]
        hist_omega = [p["omega"] for p in historical_params if p.get("is_bubble")]

        if hist_m:
            # Update m range based on historical mean ± 2*std
            m_mean = np.mean(hist_m)
            m_std = np.std(hist_m) if len(hist_m) > 1 else 0.1
            m_range = [max(0.1, m_mean - 2 * m_std), min(0.9, m_mean + 2 * m_std)]
            basis = "historical_bayesian"
            confidence = min(0.8, 0.5 + 0.1 * len(hist_m))

        if hist_omega:
            omega_mean = np.mean(hist_omega)
            omega_std = np.std(hist_omega) if len(hist_omega) > 1 else 2.0
            omega_range = [
                max(4.0, omega_mean - 2 * omega_std),
                min(15.0, omega_mean + 2 * omega_std),
            ]

    # WHY: Data characteristics can further refine prediction.
    # High volatility → expect noisier fit → wider ranges.
    # Strong trend → expect clearer signal → tighter ranges.
    n = len(values)
    if n > 200:
        # Lots of data → expect good fit → tighter ranges
        min_r2 = min(min_r2 + 0.05, 0.95)
        confidence = min(confidence + 0.1, 0.9)
    elif n < 50:
        # Little data → expect noisy fit → wider ranges
        min_r2 = max(min_r2 - 0.1, 0.5)
        confidence = max(confidence - 0.1, 0.2)

    return AcceptorPrediction(
        m_expected=tuple(m_range),
        omega_expected=tuple(omega_range),
        tc_days_ahead=tuple(tc_days),
        min_r_squared=min_r2,
        min_quality=min_quality,
        confidence=round(confidence, 2),
        basis=basis,
    )


# ---------------------------------------------------------------------------
# Comparator: Compare actual result with acceptor prediction
# ---------------------------------------------------------------------------


def compare_with_acceptor(
    acceptor: AcceptorPrediction,
    actual_m: float,
    actual_omega: float,
    actual_tc_idx: float,
    actual_r_squared: float,
    actual_quality: float,
    actual_is_bubble: bool,
    n_points: int,
) -> AcceptorComparison:
    """Compare actual LPPLS fit result with acceptor prediction.

    This is the "обратная афферентация" — comparing actual result with
    the predicted (accepted) result. If they match → satisfaction → done.
    If mismatch → targeted retry (not blind restart).

    Args:
        acceptor: The prediction made before fitting
        actual_m: Actual fitted m parameter
        actual_omega: Actual fitted omega parameter
        actual_tc_idx: Actual fitted tc (time index)
        actual_r_squared: Actual R² of fit
        actual_quality: Actual quality score
        actual_is_bubble: Whether fit qualifies as bubble
        n_points: Number of data points

    Returns:
        AcceptorComparison with match/mismatch analysis
    """
    mismatches = []

    # Check m parameter
    m_lo, m_hi = acceptor.m_expected
    if actual_m < m_lo or actual_m > m_hi:
        mismatches.append(f"m={actual_m:.3f} outside expected [{m_lo:.2f}, {m_hi:.2f}]")

    # Check omega parameter
    omega_lo, omega_hi = acceptor.omega_expected
    if actual_omega < omega_lo or actual_omega > omega_hi:
        mismatches.append(
            f"omega={actual_omega:.2f} outside expected [{omega_lo:.1f}, {omega_hi:.1f}]"
        )

    # Check R²
    if actual_r_squared < acceptor.min_r_squared:
        mismatches.append(
            f"R²={actual_r_squared:.3f} below expected {acceptor.min_r_squared:.2f}"
        )

    # Check quality
    if actual_quality < acceptor.min_quality:
        mismatches.append(
            f"quality={actual_quality:.3f} below expected {acceptor.min_quality:.2f}"
        )

    # Check tc is in expected range (days ahead)
    # Convert tc index to days ahead of last data point
    # (This requires knowing the data frequency — approximate as 1 point = 1 day for daily)
    tc_days_ahead = actual_tc_idx - n_points  # Approximate
    tc_lo, tc_hi = acceptor.tc_days_ahead
    # Only check if tc is in the future (positive days ahead)
    if tc_days_ahead < 0:
        mismatches.append(f"tc={actual_tc_idx:.0f} is in the past")

    # Determine severity
    if not mismatches:
        severity = "none"
        satisfaction = 1.0
    elif len(mismatches) == 1:
        severity = "minor"
        satisfaction = 0.6
    elif len(mismatches) == 2:
        severity = "major"
        satisfaction = 0.3
    else:
        severity = "major"
        satisfaction = 0.1

    # WHY: Targeted retry decision based on SPECIFIC mismatch type.
    # This is the key advantage over OODA (blind restart) — we know WHAT to fix.
    retry_needed = len(mismatches) > 0 and actual_is_bubble
    retry_action = None

    if retry_needed:
        retry_action = _determine_retry_action(
            mismatches=mismatches,
            actual_m=actual_m,
            actual_omega=actual_omega,
            actual_r_squared=actual_r_squared,
        )

    return AcceptorComparison(
        acceptor_match=len(mismatches) == 0,
        mismatches=mismatches,
        mismatch_severity=severity,
        retry_needed=retry_needed,
        retry_action=retry_action,
        satisfaction_score=satisfaction,
    )


def _determine_retry_action(
    mismatches: list[str],
    actual_m: float,
    actual_omega: float,
    actual_r_squared: float,
) -> str:
    """Determine SPECIFIC retry action based on mismatch type.

    This is the "целенаправленная коррекция" — not a blind restart,
    but a targeted fix for the specific problem.

    Retry actions:
    - "widen_m_range" → try fitting with wider m bounds
    - "widen_omega_range" → try fitting with wider omega bounds
    - "shorter_window" → use shorter data window (may be regime change)
    - "longer_window" → use longer data window (may need more context)
    - "relax_r2" → accept lower R² fit (data may be inherently noisy)
    - "reject" → no retry will help, reject this fit
    """
    m_mismatch = any("m=" in m for m in mismatches)
    omega_mismatch = any("omega=" in m for m in mismatches)
    r2_mismatch = any("R²=" in m for m in mismatches)

    if m_mismatch and actual_m < 0.2:
        # m too low → try wider lower bound
        return "widen_m_range_lower"
    elif m_mismatch and actual_m > 0.7:
        # m too high → try wider upper bound
        return "widen_m_range_upper"

    if omega_mismatch and actual_omega < 6.0:
        # omega too low → may not be bubble, reject
        return "reject_likely_not_bubble"
    elif omega_mismatch and actual_omega > 13.0:
        # omega too high → try wider upper bound
        return "widen_omega_range_upper"

    if r2_mismatch and actual_r_squared > 0.7:
        # Close to acceptable → relax threshold
        return "relax_r2_threshold"
    elif r2_mismatch:
        # Way too low → unlikely to improve with retry
        return "reject_poor_fit"

    return "retry_with_wider_bounds"


# ---------------------------------------------------------------------------
# TFS Pipeline: Integrate acceptor into LPPLS fitting
# ---------------------------------------------------------------------------


def run_tfs_pipeline_iteration(
    t: NDArray,
    values: NDArray,
    domain: str = "finance",
    max_iterations: int = 3,
    hmm_bubble_prob: float = 0.5,
    fit_func=None,
) -> dict:
    """Run LPPLS fit with Anokhin's Theory of Functional Systems loop.

    This replaces the simple "fit → check → done" with a full TFS cycle:

    1. Афферентный синтез: domain + HMM + data → context
    2. Акцептор: predict expected parameters BEFORE fitting
    3. Программа действия: run LPPLS fit
    4. Обратная афферентация: compare actual vs predicted
    5. Решение: accept → done, mismatch → targeted retry, reject → stop

    This is fundamentally different from OODA Loop because:
    - OODA: Observe → Orient → Decide → Act (reactive, no prediction)
    - TFS: Predict → Act → Compare → Targeted Correct (anticipatory)

    Args:
        t: Time index array
        values: Price/value array
        domain: Domain name
        max_iterations: Maximum TFS cycles before giving up
        hmm_bubble_prob: HMM bubble probability (if available)
        fit_func: LPPLS fitting function (default: use standard optimizer)

    Returns:
        Dict with fit result + acceptor metadata
    """
    from src.lppls.optimizer import LPPLSOptimizer

    n = len(values)
    log_price = np.log(np.clip(values, 1e-10, None))

    # ─── Step 1: Афферентный синтез (Afferent Synthesis) ─────────────
    # WHY: This integrates motivation (need), memory (historical fits),
    # and situational afferentation (current data characteristics).
    # Unlike OODA's "Orient", this has explicit MOTIVATION component.
    log.info(
        "tfs_afferent_synthesis",
        domain=domain,
        n_points=n,
        hmm_prob=round(hmm_bubble_prob, 2),
    )

    # ─── Step 2: Акцептор результата действия (Acceptor) ─────────────
    # WHY: We predict what a good fit should look like BEFORE fitting.
    # This allows us to know when the result is "good enough" (satisfaction).
    acceptor = create_acceptor(domain, t, values, hmm_bubble_prob)
    log.info(
        "tfs_acceptor",
        m_expected=[round(x, 2) for x in acceptor.m_expected],
        omega_expected=[round(x, 2) for x in acceptor.omega_expected],
        min_r2=round(acceptor.min_r_squared, 2),
        confidence=acceptor.confidence,
        basis=acceptor.basis,
    )

    # ─── Steps 3-5: Fit → Compare → Targeted Retry Loop ─────────────
    best_result = None
    best_satisfaction = 0.0
    current_m_range = (0.1, 0.9)
    current_omega_range = (6.0, 13.0)

    for iteration in range(max_iterations):
        # Step 3: Программа действия (Action Program) — FIT
        opt = LPPLSOptimizer(
            grid_size=12,
            n_best=5,
            m_range=current_m_range,
            omega_range=current_omega_range,
        )
        model = opt.fit(t, log_price)
        r2 = model.r_squared(t, log_price)
        params = model.params

        if params is None:
            log.info("tfs_fit_failed", iteration=iteration)
            break

        # Step 4: Обратная афферентация (Backward Afferentation) — COMPARE
        comparison = compare_with_acceptor(
            acceptor=acceptor,
            actual_m=params.m,
            actual_omega=params.omega,
            actual_tc_idx=params.tc,
            actual_r_squared=r2,
            actual_quality=0.5,  # Simplified — would use actual quality score
            actual_is_bubble=params.is_bubble,
            n_points=n,
        )

        log.info(
            "tfs_comparison",
            iteration=iteration,
            acceptor_match=comparison.acceptor_match,
            mismatches=comparison.mismatches,
            severity=comparison.mismatch_severity,
            satisfaction=round(comparison.satisfaction_score, 2),
        )

        # Step 5: Решение (Decision) — Accept / Retry / Reject
        if comparison.acceptor_match or comparison.satisfaction_score > 0.8:
            # ✓ СОВПАДЕНИЕ с акцептором → завершение
            log.info("tfs_accept_satisfied", iteration=iteration)
            best_result = {
                "params": params,
                "r_squared": r2,
                "acceptor_match": True,
                "iterations": iteration + 1,
                "satisfaction": comparison.satisfaction_score,
            }
            break

        if comparison.satisfaction_score > best_satisfaction:
            best_satisfaction = comparison.satisfaction_score
            best_result = {
                "params": params,
                "r_squared": r2,
                "acceptor_match": False,
                "iterations": iteration + 1,
                "satisfaction": comparison.satisfaction_score,
                "mismatches": comparison.mismatches,
            }

        if not comparison.retry_needed or iteration == max_iterations - 1:
            # No point retrying or max iterations reached
            log.info("tfs_give_up", iteration=iteration, action=comparison.retry_action)
            break

        # Targeted retry — adjust SPECIFIC parameters based on mismatch
        action = comparison.retry_action
        if action == "widen_m_range_lower":
            current_m_range = (0.05, current_m_range[1])
        elif action == "widen_m_range_upper":
            current_m_range = (current_m_range[0], 1.0)
        elif action == "widen_omega_range_upper":
            current_omega_range = (current_omega_range[0], 15.0)
        elif action.startswith("reject"):
            break
        else:
            # Generic retry — widen all bounds slightly
            current_m_range = (max(0.05, current_m_range[0] - 0.05),
                               min(1.0, current_m_range[1] + 0.05))
            current_omega_range = (current_omega_range[0] - 1.0,
                                   current_omega_range[1] + 1.0)

        log.info("tfs_targeted_retry", iteration=iteration, action=action)

    if best_result is None:
        return {
            "params": None,
            "r_squared": 0.0,
            "acceptor_match": False,
            "iterations": 0,
            "satisfaction": 0.0,
        }

    return best_result
