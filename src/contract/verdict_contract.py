"""Verdict Contract — rule-based agreement between LPPLS, RAG, and final verdict.

This module enforces that the final verdict cannot contradict key internal
diagnostic fields without an explicitly recorded reason.

Rules:
1. If DS-LPPLS says NO_BUBBLE with confidence=0, final verdict cannot be BUBBLE
2. If RAG says FUNDAMENTAL_GROWTH with high confidence, final verdict cannot be BUBBLE
3. If both LPPLS and RAG confirm bubble, strengthen to BUBBLE
4. Any downgrade/upgrade must have an explicit reason recorded
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger()


@dataclass
class VerdictResult:
    """Final verdict with contract compliance."""

    verdict: str  # BUBBLE | POSSIBLE | NO_BUBBLE
    reason: str
    confidence: float
    contract_violations: list[str] = field(default_factory=list)
    lppls_verdict: str = ""
    ds_verdict: str = ""
    rag_conclusion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "contract_violations": self.contract_violations,
            "lppls_verdict": self.lppls_verdict,
            "ds_verdict": self.ds_verdict,
            "rag_conclusion": self.rag_conclusion,
        }


def compute_final_verdict(
    lppls_result: dict[str, Any],
    rag_result: dict[str, Any] | None = None,
) -> VerdictResult:
    """Compute final verdict with contract enforcement.

    Args:
        lppls_result: LPPLS fit result with keys:
            - verdict: BUBBLE | POSSIBLE | NO_BUBBLE
            - ds_verdict: DS-LPPLS verdict (optional)
            - ds_confidence: DS-LPPLS confidence (optional, 0-1)
            - quality_score: 0-1
            - sornette_pass: bool (optional)
        rag_result: RAG analysis result with keys:
            - conclusion: FUNDAMENTAL_GROWTH | SPECULATIVE_BUBBLE | MIXED
            - confidence_in_conclusion: 0-1
            - fundamental_factors: list
            - speculative_factors: list

    Returns:
        VerdictResult with contract compliance check
    """
    lppls_verdict = lppls_result.get("verdict", "NO_BUBBLE")
    ds_verdict = lppls_result.get("ds_verdict", lppls_verdict)
    ds_confidence = lppls_result.get("ds_confidence", 0.0)
    quality_score = lppls_result.get("quality_score", 0.0)
    sornette_pass = lppls_result.get("sornette_pass", True)

    rag_conclusion = "UNKNOWN"
    rag_confidence = 0.0
    if rag_result:
        rag_conclusion = rag_result.get("conclusion", "UNKNOWN")
        rag_confidence = rag_result.get("confidence_in_conclusion", 0.0)

    violations = []
    final_verdict = lppls_verdict
    reason = f"LPPLS verdict: {lppls_verdict}"
    confidence = quality_score

    # ─── Rule 1: DS-LPPLS override ───────────────────────────────────
    # If DS says NO_BUBBLE with confidence=0, final cannot be BUBBLE
    if ds_verdict == "NO_BUBBLE" and ds_confidence == 0.0:
        if lppls_verdict == "BUBBLE":
            final_verdict = "POSSIBLE"
            reason = (
                f"LPPLS single-window says BUBBLE, but DS-LPPLS confidence=0 "
                f"(valid_windows=0). Downgraded to POSSIBLE."
            )
            confidence = min(quality_score, 0.5)
            log.info(
                "contract_rule1_triggered",
                lppls_verdict=lppls_verdict,
                ds_verdict=ds_verdict,
                ds_confidence=ds_confidence,
                final_verdict=final_verdict,
            )

    # ─── Rule 2: RAG fundamental override ────────────────────────────
    # If RAG says FUNDAMENTAL_GROWTH with confidence > 0.8, final cannot be BUBBLE
    if rag_conclusion == "FUNDAMENTAL_GROWTH" and rag_confidence > 0.8:
        if final_verdict == "BUBBLE":
            final_verdict = "NO_BUBBLE"
            reason = (
                f"RAG confirms fundamental growth with confidence {rag_confidence:.2f}. "
                f"Overriding LPPLS bubble signal."
            )
            confidence = 1.0 - rag_confidence
            log.info(
                "contract_rule2_triggered",
                lppls_verdict=lppls_verdict,
                rag_conclusion=rag_conclusion,
                rag_confidence=rag_confidence,
                final_verdict=final_verdict,
            )

    # ─── Rule 3: Both confirm bubble — strengthen ────────────────────
    if lppls_verdict == "BUBBLE" and rag_conclusion == "SPECULATIVE_BUBBLE":
        final_verdict = "BUBBLE"
        reason = "Both LPPLS and RAG confirm speculative bubble"
        confidence = max(quality_score, rag_confidence)
        log.info(
            "contract_rule3_triggered",
            lppls_verdict=lppls_verdict,
            rag_conclusion=rag_conclusion,
            final_verdict=final_verdict,
        )

    # ─── Rule 4: Sornette filter failure ─────────────────────────────
    if not sornette_pass and final_verdict == "BUBBLE":
        final_verdict = "POSSIBLE"
        reason = f"Sornette filters failed. {reason}"
        confidence = min(confidence, 0.6)
        violations.append("sornette_filters_failed")
        log.info(
            "contract_rule4_triggered",
            lppls_verdict=lppls_verdict,
            sornette_pass=sornette_pass,
            final_verdict=final_verdict,
        )

    # ─── Rule 5: RAG strong speculative → upgrade NO_BUBBLE to POSSIBLE
    # Reverse of Rule 2: if RAG is very confident about speculation,
    # but LPPLS missed it, upgrade to POSSIBLE (not BUBBLE — LPPLS didn't confirm)
    if rag_conclusion == "SPECULATIVE_BUBBLE" and rag_confidence > 0.8:
        if final_verdict == "NO_BUBBLE":
            final_verdict = "POSSIBLE"
            reason = (
                f"RAG confirms speculative bubble with confidence {rag_confidence:.2f}. "
                f"Upgrading LPPLS NO_BUBBLE to POSSIBLE."
            )
            confidence = max(confidence, rag_confidence * 0.5)
            log.info(
                "contract_rule5_triggered",
                lppls_verdict=lppls_verdict,
                rag_conclusion=rag_conclusion,
                rag_confidence=rag_confidence,
                final_verdict=final_verdict,
            )

    # ─── Rule 6: Contract violation check ────────────────────────────
    # Check if final verdict contradicts internal diagnostics
    if final_verdict == "BUBBLE" and ds_verdict == "NO_BUBBLE" and ds_confidence > 0.5:
        violations.append("bubble_vs_ds_no_bubble")
        log.warning(
            "contract_violation_detected",
            violation="bubble_vs_ds_no_bubble",
            final_verdict=final_verdict,
            ds_verdict=ds_verdict,
            ds_confidence=ds_confidence,
        )

    if final_verdict == "NO_BUBBLE" and rag_conclusion == "SPECULATIVE_BUBBLE" and rag_confidence > 0.8:
        violations.append("no_bubble_vs_rag_speculative")
        log.warning(
            "contract_violation_detected",
            violation="no_bubble_vs_rag_speculative",
            final_verdict=final_verdict,
            rag_conclusion=rag_conclusion,
            rag_confidence=rag_confidence,
        )

    return VerdictResult(
        verdict=final_verdict,
        reason=reason,
        confidence=confidence,
        contract_violations=violations,
        lppls_verdict=lppls_verdict,
        ds_verdict=ds_verdict,
        rag_conclusion=rag_conclusion,
    )


def check_contract(lppls: dict, rag: dict | None, final: VerdictResult) -> list[str]:
    """Check for contract violations between LPPLS, RAG, and final verdict.

    Returns:
        List of violation descriptions
    """
    violations = list(final.contract_violations)

    # Additional checks
    if final.verdict == "BUBBLE" and final.confidence < 0.5:
        violations.append("bubble_with_low_confidence")

    if final.verdict == "NO_BUBBLE" and lppls.get("quality_score", 0) > 0.8:
        violations.append("no_bubble_vs_high_quality_lppls")

    return violations
