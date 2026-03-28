"""Bootstrap confidence intervals and robustness checks for universality claim.

Goes beyond simple KS-test: bootstrap resampling to estimate
confidence intervals on the difference between domain distributions.
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
