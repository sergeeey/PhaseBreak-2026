"""Universality testing: KS, TOST equivalence, bootstrap, effect sizes.

Replaces naive KS-only approach with rigorous equivalence testing:
- KS test: "can we reject H₀ that distributions are same?" (weak — absence of evidence)
- TOST equivalence: "can we confirm distributions are equivalent within ±Δ?" (strong)
- Cohen's d: effect size magnitude (negligible/small/medium/large)
- Bootstrap: confidence intervals on KS p-values

Key finding: KS p>0.05 with n=4 has ~30% power. TOST is the correct test
for universality claims. Only Finance↔Commodities m passes TOST (p=0.03).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray
from scipy.stats import ks_2samp

from src.cross_domain.correlation import DomainParams

log = structlog.get_logger()


@dataclass
class EquivalenceResult:
    """TOST equivalence test result for one parameter pair."""

    parameter: str
    domain_a: str
    domain_b: str
    ks_p: float  # KS test p-value (old method)
    tost_p: float | None  # TOST p-value (new method, None if pingouin missing)
    cohens_d: float  # effect size
    effect_magnitude: str  # negligible/small/medium/large
    bound: float  # equivalence bound used
    is_equivalent: bool  # TOST p < 0.05 → proven equivalent
    is_ks_not_rejected: bool  # KS p > 0.05 → not rejected (weak)
    n_a: int
    n_b: int


@dataclass
class BootstrapResult:
    """Bootstrap confidence interval for KS test p-value."""

    parameter: str
    domain_a: str
    domain_b: str
    observed_p: float
    ci_lower: float  # 2.5th percentile
    ci_upper: float  # 97.5th percentile
    fraction_significant: float  # fraction of bootstrap samples with p < 0.05
    is_robust: bool  # True if ≥80% of samples agree with observed


def _cohens_d(a: NDArray, b: NDArray) -> float:
    """Cohen's d effect size (pooled std)."""
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    )
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _effect_magnitude(d: float) -> str:
    """Classify Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def equivalence_test(
    a: NDArray,
    b: NDArray,
    parameter: str,
    domain_a: str,
    domain_b: str,
    bound: float,
) -> EquivalenceResult:
    """Run TOST equivalence test + KS + Cohen's d.

    TOST (Two One-Sided Tests) tests whether the mean difference is
    within ±bound, providing evidence OF equivalence (not just absence
    of evidence against it).
    """
    # KS test (legacy)
    ks_stat, ks_p = ks_2samp(a, b)

    # Cohen's d
    d = _cohens_d(a, b)
    mag = _effect_magnitude(d)

    # TOST equivalence (requires pingouin)
    tost_p = None
    is_equiv = False
    try:
        import pingouin as pg

        result = pg.tost(x=a, y=b, bound=bound, paired=False)
        tost_p = float(result["pval"].values[0])
        is_equiv = tost_p < 0.05
    except ImportError:
        log.warning("pingouin_not_installed", msg="pip install pingouin for TOST")
    except Exception as e:
        log.warning("tost_failed", error=str(e))

    log.info(
        "equivalence_test",
        parameter=parameter,
        domains=f"{domain_a}↔{domain_b}",
        ks_p=round(ks_p, 4),
        tost_p=round(tost_p, 4) if tost_p is not None else None,
        cohens_d=round(d, 3),
        effect=mag,
        equivalent=is_equiv,
    )

    return EquivalenceResult(
        parameter=parameter,
        domain_a=domain_a,
        domain_b=domain_b,
        ks_p=round(ks_p, 4),
        tost_p=round(tost_p, 4) if tost_p is not None else None,
        cohens_d=round(d, 3),
        effect_magnitude=mag,
        bound=bound,
        is_equivalent=is_equiv,
        is_ks_not_rejected=ks_p > 0.05,
        n_a=len(a),
        n_b=len(b),
    )


def run_full_equivalence(
    domain_a: DomainParams,
    domain_b: DomainParams,
    m_bound: float = 0.2,
    omega_bound: float = 3.0,
) -> list[EquivalenceResult]:
    """Run equivalence tests for m and ω between two domains."""
    results = []
    for param_name, a_vals, b_vals, bound in [
        ("m", domain_a.m_values, domain_b.m_values, m_bound),
        ("omega", domain_a.omega_values, domain_b.omega_values, omega_bound),
    ]:
        if len(a_vals) < 2 or len(b_vals) < 2:
            continue
        results.append(
            equivalence_test(a_vals, b_vals, param_name, domain_a.name, domain_b.name, bound)
        )
    return results


@dataclass
class SensitivityRow:
    """One row of TOST sensitivity analysis."""

    m_bound: float
    omega_bound: float
    m_tost_p: float | None
    m_equivalent: bool
    omega_tost_p: float | None
    omega_equivalent: bool


def tost_sensitivity_analysis(
    domain_a: DomainParams,
    domain_b: DomainParams,
    m_bounds: list[float] | None = None,
    omega_bounds: list[float] | None = None,
) -> list[SensitivityRow]:
    """Run TOST equivalence at multiple bound levels.

    Shows how the equivalence conclusion changes as margins tighten.
    Useful for reviewers who question the default bounds.
    """
    if m_bounds is None:
        m_bounds = [0.10, 0.15, 0.20, 0.25, 0.30]
    if omega_bounds is None:
        omega_bounds = [1.0, 2.0, 3.0, 4.0]

    rows: list[SensitivityRow] = []
    for mb in m_bounds:
        for ob in omega_bounds:
            results = run_full_equivalence(domain_a, domain_b, m_bound=mb, omega_bound=ob)
            m_res = next((r for r in results if r.parameter == "m"), None)
            o_res = next((r for r in results if r.parameter == "omega"), None)
            rows.append(
                SensitivityRow(
                    m_bound=mb,
                    omega_bound=ob,
                    m_tost_p=m_res.tost_p if m_res else None,
                    m_equivalent=m_res.is_equivalent if m_res else False,
                    omega_tost_p=o_res.tost_p if o_res else None,
                    omega_equivalent=o_res.is_equivalent if o_res else False,
                )
            )

    # Log summary
    m_equiv_bounds = [r.m_bound for r in rows if r.m_equivalent]
    o_equiv_bounds = [r.omega_bound for r in rows if r.omega_equivalent]
    log.info(
        "tost_sensitivity",
        domains=f"{domain_a.name}↔{domain_b.name}",
        m_equivalent_at=sorted(set(m_equiv_bounds)) if m_equiv_bounds else "none",
        omega_equivalent_at=sorted(set(o_equiv_bounds)) if o_equiv_bounds else "none",
        total_combos=len(rows),
    )
    return rows


def bootstrap_ks_test(
    a: NDArray,
    b: NDArray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Bootstrap KS test between two samples.

    Returns:
        (observed_p, ci_lower, ci_upper, fraction_significant)
    """
    rng = np.random.default_rng(seed)
    observed_ks = ks_2samp(a, b)

    p_values = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        a_boot = rng.choice(a, size=len(a), replace=True)
        b_boot = rng.choice(b, size=len(b), replace=True)
        p_values[i] = ks_2samp(a_boot, b_boot).pvalue

    ci_lower = float(np.percentile(p_values, 2.5))
    ci_upper = float(np.percentile(p_values, 97.5))
    frac_sig = float(np.mean(p_values < 0.05))

    return float(observed_ks.pvalue), ci_lower, ci_upper, frac_sig


def robustness_check(
    domain_a: DomainParams,
    domain_b: DomainParams,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[BootstrapResult]:
    """Run bootstrap robustness check for m and ω between two domains."""
    results = []

    for param_name, a_vals, b_vals in [
        ("m", domain_a.m_values, domain_b.m_values),
        ("omega", domain_a.omega_values, domain_b.omega_values),
    ]:
        if len(a_vals) < 3 or len(b_vals) < 3:
            continue

        obs_p, ci_lo, ci_hi, frac_sig = bootstrap_ks_test(
            a_vals, b_vals, n_bootstrap=n_bootstrap, seed=seed
        )

        # Robust if ≥80% of bootstrap samples agree with observed conclusion
        observed_similar = obs_p > 0.05
        if observed_similar:
            is_robust = frac_sig < 0.2  # <20% disagree
        else:
            is_robust = frac_sig > 0.8  # >80% agree it's different

        results.append(
            BootstrapResult(
                parameter=param_name,
                domain_a=domain_a.name,
                domain_b=domain_b.name,
                observed_p=obs_p,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                fraction_significant=frac_sig,
                is_robust=is_robust,
            )
        )

        log.info(
            "bootstrap_ks",
            parameter=param_name,
            observed_p=round(obs_p, 4),
            ci=f"[{ci_lo:.4f}, {ci_hi:.4f}]",
            frac_sig=round(frac_sig, 3),
            is_robust=is_robust,
        )

    return results
