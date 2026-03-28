"""Cross-domain universality analysis — THE main scientific test.

Collects (m, ω) from Finance (Stage 1) and Geology (Stage 2),
runs full statistical analysis with bootstrap robustness check.
"""

import numpy as np
import pytest
import structlog

from src.cross_domain.correlation import (
    DomainParams,
    full_cross_domain_analysis,
    parameter_overlap_score,
    pairwise_comparison,
    CrossDomainResult,
)
from src.cross_domain.universality import robustness_check, BootstrapResult

log = structlog.get_logger()

# ============================================================
# Real data from Stage 1 (Finance) and Stage 2 (Geology)
# ============================================================

# Finance: from test_validation_known_bubbles.py results
# Only fits that passed Sornette filters (is_bubble=True)
FINANCE = DomainParams(
    name="Finance",
    m_values=np.array([0.637, 0.565, 0.478, 0.437]),  # btc17, btc21, dotcom, china
    omega_values=np.array([11.86, 6.00, 9.93, 10.83]),
    r_squared=np.array([0.979, 0.773, 0.962, 0.989]),
    labels=["BTC 2017", "BTC 2021", "Dotcom 2000", "China 2015"],
)

# Finance multi-window: from test_full_validation.py
FINANCE_MW = DomainParams(
    name="Finance (MW)",
    m_values=np.array([0.282, 0.519, 0.600, 0.632]),  # multi-window means
    omega_values=np.array([8.59, 7.50, 9.50, 8.33]),
    r_squared=np.array([0.97, 0.93, 0.96, 0.99]),
    labels=["BTC 2017", "BTC 2021", "Dotcom 2000", "China 2015"],
)

# Geology: from test_geo_real_data.py results (R² > 0.5 fits only)
GEOLOGY = DomainParams(
    name="Geology",
    m_values=np.array(
        [0.762, 0.456, 0.633, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.189, 0.367, 0.900, 0.900]
    ),
    omega_values=np.array(
        [6.47, 15.67, 13.33, 13.33, 8.67, 6.33, 6.33, 6.33, 13.33, 8.67, 18.00, 8.67, 8.67]
    ),
    r_squared=np.array(
        [0.973, 0.946, 0.952, 0.933, 0.887, 0.846, 0.843, 0.833, 0.827, 0.788, 0.556, 0.881, 0.861]
    ),
    labels=[
        "42UYC_ndvi_1",
        "42UYC_ndvi_2",
        "42UYC_clay",
        "42UYC_bsi",
        "43UCU_ndvi_1",
        "43UCU_clay_1",
        "43UCU_ndvi_2",
        "43UCU_clay_2",
        "43UCU_iron",
        "42UYC_bsi_2",
        "43UCU_bsi",
        "42UYC_iron_1",
        "42UYC_iron_2",
    ],
)


class TestDomainParams:
    def test_finance_data(self):
        assert FINANCE.n == 4
        assert all(FINANCE.r_squared > 0.7)

    def test_geology_data(self):
        assert GEOLOGY.n >= 10

    def test_filter_quality(self):
        geo_filtered = GEOLOGY.filter_quality(min_r2=0.8)
        assert geo_filtered.n < GEOLOGY.n
        assert all(geo_filtered.r_squared >= 0.8)


class TestPairwiseComparison:
    def test_finance_vs_geology(self):
        tests = pairwise_comparison(FINANCE, GEOLOGY)
        assert len(tests) == 2  # m and omega
        for t in tests:
            assert t.domain_a == "Finance"
            assert t.domain_b == "Geology"
            assert t.ks_pvalue >= 0
            assert t.mw_pvalue >= 0


class TestFullAnalysis:
    @pytest.fixture(scope="class")
    def result(self) -> CrossDomainResult:
        return full_cross_domain_analysis([FINANCE, FINANCE_MW, GEOLOGY], min_r2=0.5)

    def test_result_structure(self, result):
        assert isinstance(result, CrossDomainResult)
        assert len(result.domains) >= 2
        assert len(result.pairwise_tests) > 0

    def test_verdict_exists(self, result):
        assert result.overall_verdict in (
            "UNIVERSAL",
            "WEAK_UNIVERSAL",
            "DOMAIN_SPECIFIC",
            "INSUFFICIENT_DATA",
        )

    def test_print_full_analysis(self, result):
        print("\n" + "=" * 95)
        print("CROSS-DOMAIN UNIVERSALITY ANALYSIS — THE MAIN FINDING")
        print("=" * 95)
        print(f"\nDomains: {len(result.domains)}")
        for d in result.domains:
            print(
                f"  {d.name}: n={d.n}, m=[{d.m_values.min():.3f}, {d.m_values.max():.3f}], "
                f"ω=[{d.omega_values.min():.2f}, {d.omega_values.max():.2f}]"
            )

        print(f"\nPairwise tests: {result.n_total_tests}")
        print(
            f"{'Domain A':<20} {'Domain B':<20} {'Param':>6} {'KS p':>8} {'MW p':>8} {'Verdict':>15}"
        )
        print("-" * 85)
        for t in result.pairwise_tests:
            print(
                f"{t.domain_a:<20} {t.domain_b:<20} {t.parameter:>6} "
                f"{t.ks_pvalue:>8.4f} {t.mw_pvalue:>8.4f} {t.verdict:>15}"
            )

        print(f"\n{'=' * 85}")
        print(f"OVERALL VERDICT: {result.overall_verdict}")
        print(f"Universal tests: {result.n_universal}/{result.n_total_tests}")
        print(f"{'=' * 85}")


class TestParameterOverlap:
    def test_overlap_computed(self):
        overlap = parameter_overlap_score([FINANCE, GEOLOGY])
        assert 0.0 <= overlap["m_overlap"] <= 1.0
        assert 0.0 <= overlap["omega_overlap"] <= 1.0

    def test_print_overlap(self):
        overlap = parameter_overlap_score([FINANCE, GEOLOGY])
        print(f"\nParameter overlap (Finance vs Geology):")
        print(f"  m overlap:     {overlap['m_overlap']:.1%}")
        print(f"  ω overlap:     {overlap['omega_overlap']:.1%}")


class TestBootstrapRobustness:
    @pytest.fixture(scope="class")
    def bootstrap_results(self) -> list[BootstrapResult]:
        return robustness_check(FINANCE, GEOLOGY, n_bootstrap=500, seed=42)

    def test_returns_results(self, bootstrap_results):
        assert len(bootstrap_results) == 2  # m and omega

    def test_print_bootstrap(self, bootstrap_results):
        print("\n" + "=" * 80)
        print("BOOTSTRAP ROBUSTNESS CHECK (500 resamples)")
        print("=" * 80)
        for r in bootstrap_results:
            print(
                f"{r.parameter:>6}: observed p={r.observed_p:.4f}, "
                f"95% CI=[{r.ci_lower:.4f}, {r.ci_upper:.4f}], "
                f"frac_sig={r.fraction_significant:.3f}, "
                f"robust={'YES' if r.is_robust else 'NO'}"
            )
        print("=" * 80)
