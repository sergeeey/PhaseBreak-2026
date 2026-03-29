"""Cross-domain (m, ω) correlation analysis.

Collects LPPLS parameters from Finance and Geology domains,
runs statistical tests to determine if phase transition signatures
are universal or domain-specific.

This is THE central scientific finding of PhaseBreak.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import structlog
from numpy.typing import NDArray
from scipy.stats import ks_2samp, mannwhitneyu, pearsonr, spearmanr

log = structlog.get_logger()


@dataclass
class DomainParams:
    """LPPLS parameters from a single domain."""

    name: str
    m_values: NDArray
    omega_values: NDArray
    r_squared: NDArray  # quality filter
    labels: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.m_values)

    def filter_quality(self, min_r2: float = 0.5) -> "DomainParams":
        """Keep only fits with R² above threshold."""
        mask = self.r_squared >= min_r2
        labels = [l for l, m in zip(self.labels, mask) if m] if self.labels else []
        return DomainParams(
            name=self.name,
            m_values=self.m_values[mask],
            omega_values=self.omega_values[mask],
            r_squared=self.r_squared[mask],
            labels=labels,
        )


@dataclass
class PairwiseTest:
    """Statistical test result between two domains."""

    domain_a: str
    domain_b: str
    parameter: str  # "m" or "omega"
    ks_statistic: float
    ks_pvalue: float
    mw_statistic: float
    mw_pvalue: float
    is_similar: bool  # both p > 0.05

    @property
    def verdict(self) -> str:
        """WHY NOT_REJECTED instead of UNIVERSAL: p>0.05 means 'cannot reject H0',
        not 'H0 confirmed'. With n<20, KS has very low power — see paper caveat."""
        if self.ks_pvalue > 0.05 and self.mw_pvalue > 0.05:
            return "NOT_REJECTED"
        elif self.ks_pvalue > 0.05 or self.mw_pvalue > 0.05:
            return "WEAK_NOT_REJECTED"
        else:
            return "REJECTED"


@dataclass
class CrossDomainResult:
    """Complete cross-domain analysis result."""

    domains: list[DomainParams]
    pairwise_tests: list[PairwiseTest]
    overall_verdict: str  # NOT_REJECTED / WEAK_NOT_REJECTED / REJECTED
    summary: str

    @property
    def n_not_rejected(self) -> int:
        return sum(1 for t in self.pairwise_tests if t.verdict == "NOT_REJECTED")

    @property
    def n_total_tests(self) -> int:
        return len(self.pairwise_tests)


def pairwise_comparison(a: DomainParams, b: DomainParams) -> list[PairwiseTest]:
    """Run KS and Mann-Whitney tests between two domains for m and ω."""
    tests = []

    for param_name, a_vals, b_vals in [
        ("m", a.m_values, b.m_values),
        ("omega", a.omega_values, b.omega_values),
    ]:
        if len(a_vals) < 3 or len(b_vals) < 3:
            tests.append(
                PairwiseTest(
                    domain_a=a.name,
                    domain_b=b.name,
                    parameter=param_name,
                    ks_statistic=0,
                    ks_pvalue=0,
                    mw_statistic=0,
                    mw_pvalue=0,
                    is_similar=False,
                )
            )
            continue

        ks = ks_2samp(a_vals, b_vals)
        mw = mannwhitneyu(a_vals, b_vals, alternative="two-sided")

        is_similar = ks.pvalue > 0.05 and mw.pvalue > 0.05

        tests.append(
            PairwiseTest(
                domain_a=a.name,
                domain_b=b.name,
                parameter=param_name,
                ks_statistic=float(ks.statistic),
                ks_pvalue=float(ks.pvalue),
                mw_statistic=float(mw.statistic),
                mw_pvalue=float(mw.pvalue),
                is_similar=is_similar,
            )
        )

    return tests


def full_cross_domain_analysis(
    domains: list[DomainParams],
    min_r2: float = 0.5,
) -> CrossDomainResult:
    """Run complete cross-domain (m, ω) analysis.

    Args:
        domains: List of DomainParams from each domain
        min_r2: Minimum R² to include a fit

    Returns:
        CrossDomainResult with all pairwise tests and overall verdict
    """
    # Filter by quality
    filtered = [d.filter_quality(min_r2) for d in domains]
    filtered = [d for d in filtered if d.n >= 3]

    if len(filtered) < 2:
        return CrossDomainResult(
            domains=filtered,
            pairwise_tests=[],
            overall_verdict="INSUFFICIENT_DATA",
            summary=f"Need ≥2 domains with ≥3 fits each. Got {len(filtered)}.",
        )

    # All pairwise comparisons
    all_tests = []
    for i in range(len(filtered)):
        for j in range(i + 1, len(filtered)):
            tests = pairwise_comparison(filtered[i], filtered[j])
            all_tests.extend(tests)

    # Overall verdict
    n_nr = sum(1 for t in all_tests if t.verdict == "NOT_REJECTED")
    n_weak = sum(1 for t in all_tests if t.verdict == "WEAK_NOT_REJECTED")
    n_total = len(all_tests)

    if n_nr == n_total:
        verdict = "NOT_REJECTED"
    elif n_nr + n_weak >= n_total * 0.5:
        verdict = "WEAK_NOT_REJECTED"
    else:
        verdict = "REJECTED"

    # Build summary
    lines = [f"Cross-domain analysis: {len(filtered)} domains, {n_total} tests"]
    for d in filtered:
        lines.append(
            f"  {d.name}: n={d.n}, m=[{d.m_values.min():.3f}, {d.m_values.max():.3f}], "
            f"ω=[{d.omega_values.min():.2f}, {d.omega_values.max():.2f}]"
        )
    lines.append(f"Verdict: {verdict} ({n_nr}/{n_total} not rejected)")

    log.info(
        "cross_domain_analysis",
        n_domains=len(filtered),
        n_tests=n_total,
        n_not_rejected=n_nr,
        verdict=verdict,
    )

    return CrossDomainResult(
        domains=filtered,
        pairwise_tests=all_tests,
        overall_verdict=verdict,
        summary="\n".join(lines),
    )


def parameter_overlap_score(domains: list[DomainParams]) -> dict[str, float]:
    """Compute overlap score between domains for m and ω.

    Overlap = fraction of combined range covered by intersection.
    1.0 = perfect overlap, 0.0 = no overlap.
    """
    if len(domains) < 2:
        return {"m_overlap": 0.0, "omega_overlap": 0.0}

    def _overlap(arrays: list[NDArray]) -> float:
        mins = [a.min() for a in arrays if len(a) > 0]
        maxs = [a.max() for a in arrays if len(a) > 0]
        if not mins:
            return 0.0
        intersection = max(0, min(maxs) - max(mins))
        union = max(maxs) - min(mins)
        return intersection / union if union > 0 else 0.0

    m_arrays = [d.m_values for d in domains if d.n > 0]
    omega_arrays = [d.omega_values for d in domains if d.n > 0]

    return {
        "m_overlap": _overlap(m_arrays),
        "omega_overlap": _overlap(omega_arrays),
    }
