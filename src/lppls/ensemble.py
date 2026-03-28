"""HMM-gated LPPLS Ensemble Pipeline.

Strategy:
1. HMM regime detector classifies current market state
2. If BUBBLE or strong GROWTH → run LPPLS with multi-window confidence
3. If NORMAL → skip LPPLS (saves compute, reduces false positives)

This combination (HMM + LPPLS) is novel — not described in existing literature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray

from src.lppls.confidence import ConfidenceResult, MultiWindowConfidence
from src.lppls.regime import HMMRegimeDetector, MarketRegime, RegimeResult

log = structlog.get_logger()


@dataclass
class EnsembleResult:
    """Combined HMM + LPPLS analysis result."""

    regime: RegimeResult
    lppls_confidence: ConfidenceResult | None  # None if HMM said NORMAL
    final_verdict: str  # BUBBLE / POSSIBLE_BUBBLE / NO_BUBBLE
    tc_estimate: float | None
    reasoning: str

    @property
    def is_bubble(self) -> bool:
        return self.final_verdict in ("BUBBLE", "POSSIBLE_BUBBLE")


class HMMLPPLSEnsemble:
    """HMM-gated LPPLS ensemble for bubble detection.

    Reduces false positives by only running expensive LPPLS fitting
    when HMM detects a non-normal market regime.
    """

    def __init__(
        self,
        hmm_kwargs: dict | None = None,
        confidence_kwargs: dict | None = None,
        skip_normal: bool = True,
    ) -> None:
        self.hmm = HMMRegimeDetector(**(hmm_kwargs or {}))
        self.confidence = MultiWindowConfidence(**(confidence_kwargs or {}))
        self.skip_normal = skip_normal

    def analyze(self, t: NDArray, log_price: NDArray) -> EnsembleResult:
        """Run full HMM → LPPLS pipeline.

        Args:
            t: Time array (integer indices)
            log_price: Natural log of price series

        Returns:
            EnsembleResult with combined verdict
        """
        # Stage 1: HMM regime detection
        regime = self.hmm.fit_predict(log_price)

        # Stage 2: Decision — run LPPLS?
        if self.skip_normal and not regime.should_fit_lppls:
            log.info(
                "ensemble_skip_lppls",
                regime=regime.current_regime.name,
                bubble_prob=round(regime.bubble_probability, 3),
            )
            return EnsembleResult(
                regime=regime,
                lppls_confidence=None,
                final_verdict="NO_BUBBLE",
                tc_estimate=None,
                reasoning=f"HMM regime={regime.current_regime.name}, "
                f"bubble_prob={regime.bubble_probability:.3f} — skipped LPPLS",
            )

        # Stage 3: Multi-window LPPLS confidence
        lppls_result = self.confidence.evaluate(t, log_price)

        # Stage 4: Combine signals
        verdict, reasoning = self._combine(regime, lppls_result)

        log.info(
            "ensemble_complete",
            regime=regime.current_regime.name,
            lppls_confidence=lppls_result.confidence,
            verdict=verdict,
        )

        return EnsembleResult(
            regime=regime,
            lppls_confidence=lppls_result,
            final_verdict=verdict,
            tc_estimate=lppls_result.tc_median,
            reasoning=reasoning,
        )

    @staticmethod
    def _combine(regime: RegimeResult, lppls: ConfidenceResult) -> tuple[str, str]:
        """Combine HMM regime and LPPLS confidence into final verdict."""
        hmm_bubble = regime.current_regime == MarketRegime.BUBBLE
        hmm_growth = regime.current_regime == MarketRegime.GROWTH
        lppls_high = lppls.confidence == "HIGH"
        lppls_medium = lppls.confidence == "MEDIUM"

        # Both agree: strong signal
        if hmm_bubble and lppls_high:
            return "BUBBLE", "HMM=BUBBLE + LPPLS=HIGH → strong bubble signal"

        # One strong + one moderate
        if (hmm_bubble and lppls_medium) or (hmm_growth and lppls_high):
            return "POSSIBLE_BUBBLE", (
                f"HMM={regime.current_regime.name} + LPPLS={lppls.confidence} "
                "→ possible bubble, monitor closely"
            )

        # WHY: HMM=BUBBLE + LPPLS=LOW → weak but present log-periodic signal.
        # HMM alone has high false positive rate on normal growth, so require
        # at least LOW LPPLS confirmation before flagging.
        if hmm_bubble and lppls.confidence == "LOW":
            return "POSSIBLE_BUBBLE", (
                f"HMM=BUBBLE + LPPLS=LOW → weak log-periodic signal in bubble regime"
            )

        # HMM=BUBBLE but LPPLS sees nothing → HMM false positive (normal growth)
        if hmm_bubble and lppls.confidence == "NO_SIGNAL":
            return "NO_BUBBLE", (
                f"HMM=BUBBLE but LPPLS=NO_SIGNAL → no log-periodic confirmation, "
                "likely normal growth misclassified by HMM"
            )

        # HMM sees growth but LPPLS disagrees
        if hmm_growth and lppls.confidence in ("LOW", "NO_SIGNAL"):
            return "NO_BUBBLE", (
                f"HMM=GROWTH but LPPLS={lppls.confidence} → growth without log-periodic signature"
            )

        # LPPLS sees something but HMM doesn't
        if lppls_medium and not hmm_bubble:
            return "POSSIBLE_BUBBLE", (
                f"LPPLS={lppls.confidence} but HMM={regime.current_regime.name} "
                "→ weak signal, needs more data"
            )

        return "NO_BUBBLE", (
            f"HMM={regime.current_regime.name}, LPPLS={lppls.confidence} → no bubble detected"
        )
