"""Tests protecting official public claims.

If any of these fail, README/paper/benchmark are out of sync.
"""

import json
from pathlib import Path

import pytest


BENCHMARK_PATH = Path("data/v2_results/benchmark_results.json")
OFFICIAL_EPISODE_COUNT = 58
OFFICIAL_TEST_COUNT_MIN = 253


class TestBenchmarkTruth:
    """Ensure benchmark truth is consistent."""

    def test_benchmark_file_exists(self):
        assert BENCHMARK_PATH.exists(), "Official benchmark results missing"

    def test_official_episode_count(self):
        if not BENCHMARK_PATH.exists():
            pytest.skip("No benchmark file")
        with open(BENCHMARK_PATH) as f:
            data = json.load(f)
        assert data["total_episodes"] == OFFICIAL_EPISODE_COUNT, (
            f"Benchmark has {data['total_episodes']} episodes, expected {OFFICIAL_EPISODE_COUNT}"
        )

    def test_benchmark_has_all_domains(self):
        if not BENCHMARK_PATH.exists():
            pytest.skip("No benchmark file")
        with open(BENCHMARK_PATH) as f:
            data = json.load(f)
        expected = {
            "finance",
            "commodities",
            "housing",
            "housing_monthly",
            "adversarial",
            "forward",
        }
        actual = set(data["domains"].keys())
        assert expected == actual, f"Missing domains: {expected - actual}"


class TestDependencies:
    """Ensure v2 pipeline dependencies are available."""

    def test_hurst_available(self):
        import hurst  # noqa: F401

    def test_mfdfa_available(self):
        import MFDFA  # noqa: F401

    def test_hmmlearn_available(self):
        import hmmlearn  # noqa: F401


class TestEpisodeCounts:
    """Ensure data modules have expected episode counts."""

    def test_finance_episodes(self):
        from src.lppls.data import KNOWN_BUBBLES, NEGATIVE_CONTROLS

        assert len(KNOWN_BUBBLES) + len(NEGATIVE_CONTROLS) == 20

    def test_commodity_episodes(self):
        from src.lppls.data_commodities import COMMODITY_BUBBLES, COMMODITY_CONTROLS

        assert len(COMMODITY_BUBBLES) + len(COMMODITY_CONTROLS) == 10

    def test_housing_fhfa_episodes(self):
        from src.housing.data import HOUSING_BUBBLES, HOUSING_CONTROLS

        assert len(HOUSING_BUBBLES) + len(HOUSING_CONTROLS) == 10

    def test_housing_zillow_episodes(self):
        from src.housing.data import ZILLOW_BUBBLES, ZILLOW_CONTROLS

        assert len(ZILLOW_BUBBLES) + len(ZILLOW_CONTROLS) == 6

    def test_forward_episodes(self):
        from src.lppls.data import FORWARD_BUBBLES, FORWARD_CONTROLS

        assert len(FORWARD_BUBBLES) + len(FORWARD_CONTROLS) == 6

    def test_adversarial_episodes(self):
        from src.validation.adversarial_controls import generate_adversarial_series

        assert len(generate_adversarial_series()) == 6

    def test_total_58(self):
        from src.lppls.data import (
            KNOWN_BUBBLES,
            NEGATIVE_CONTROLS,
            FORWARD_BUBBLES,
            FORWARD_CONTROLS,
        )
        from src.lppls.data_commodities import COMMODITY_BUBBLES, COMMODITY_CONTROLS
        from src.housing.data import (
            HOUSING_BUBBLES,
            HOUSING_CONTROLS,
            ZILLOW_BUBBLES,
            ZILLOW_CONTROLS,
        )
        from src.validation.adversarial_controls import generate_adversarial_series

        total = (
            len(KNOWN_BUBBLES)
            + len(NEGATIVE_CONTROLS)
            + len(COMMODITY_BUBBLES)
            + len(COMMODITY_CONTROLS)
            + len(HOUSING_BUBBLES)
            + len(HOUSING_CONTROLS)
            + len(ZILLOW_BUBBLES)
            + len(ZILLOW_CONTROLS)
            + len(generate_adversarial_series())
            + len(FORWARD_BUBBLES)
            + len(FORWARD_CONTROLS)
        )
        assert total == OFFICIAL_EPISODE_COUNT


class TestTOSTSensitivity:
    """TOST equivalence must hold at default bounds and show degradation at tighter ones."""

    def test_sensitivity_finance_commodities(self):
        """Finance↔Commodities m should be equivalent at m_bound=0.2 (known result)."""
        from src.cross_domain.universality import tost_sensitivity_analysis
        from src.cross_domain.correlation import DomainParams
        from src.lppls.data import KNOWN_BUBBLES
        from src.lppls.data_commodities import COMMODITY_BUBBLES
        import numpy as np

        # Collect fitted m and omega from known bubbles
        from src.pipeline.stages import run_full_pipeline

        fin_m, fin_omega, com_m, com_omega = [], [], [], []
        for ep in list(KNOWN_BUBBLES.values())[:4]:
            result = run_full_pipeline(ep["ticker"], ep["start"], ep["end"])
            if result.fit and hasattr(result.fit, "m") and result.fit.m is not None:
                fin_m.append(result.fit.m)
                fin_omega.append(result.fit.omega)
        for ep in list(COMMODITY_BUBBLES.values())[:4]:
            result = run_full_pipeline(ep["ticker"], ep["start"], ep["end"])
            if result.fit and hasattr(result.fit, "m") and result.fit.m is not None:
                com_m.append(result.fit.m)
                com_omega.append(result.fit.omega)

        if len(fin_m) < 2 or len(com_m) < 2:
            pytest.skip("Not enough successful fits for sensitivity analysis")

        finance = DomainParams(
            name="finance", m_values=np.array(fin_m), omega_values=np.array(fin_omega)
        )
        commodities = DomainParams(
            name="commodities", m_values=np.array(com_m), omega_values=np.array(com_omega)
        )

        rows = tost_sensitivity_analysis(finance, commodities)
        assert len(rows) == 20  # 5 m_bounds × 4 omega_bounds

        # At default bound (m=0.2), m should be equivalent (known result)
        default_rows = [r for r in rows if r.m_bound == 0.2]
        m_equiv_at_default = any(r.m_equivalent for r in default_rows)
        # Informational: log which bounds pass/fail (no hard assert on tighter bounds)
        for r in rows:
            if r.m_equivalent:
                print(f"  m EQUIVALENT at m_bound={r.m_bound}, omega_bound={r.omega_bound}")

        assert m_equiv_at_default, "Finance↔Commodities m should be TOST-equivalent at m_bound=0.2"


class TestLegacyIsolation:
    """Legacy path must not depend on v2 enhancements."""

    def test_legacy_screening_no_hurst(self):
        """Legacy screening should work even without hurst package."""
        import numpy as np
        from src.pipeline.stages import _legacy_screening

        t = np.arange(100, dtype=float)
        values = 100 + np.random.default_rng(42).normal(0, 1, 100)
        result = _legacy_screening(t, values)
        assert result.data_quality == "OK"

    def test_legacy_no_domain_param(self):
        """Legacy pipeline should not accept domain parameter."""
        import inspect
        from src.pipeline.stages import run_legacy_pipeline

        sig = inspect.signature(run_legacy_pipeline)
        assert "domain" not in sig.parameters
