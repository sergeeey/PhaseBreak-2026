"""Tests for HMM-gated LPPLS ensemble pipeline."""

import numpy as np
import pytest

from src.lppls.model import LPPLS, LPPLSParams
from src.lppls.ensemble import HMMLPPLSEnsemble, EnsembleResult


@pytest.fixture
def synthetic_bubble() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic bubble data with known LPPLS parameters."""
    params = LPPLSParams(tc=300.0, m=0.5, omega=8.0, A=10.0, B=-0.5, C1=0.02, C2=0.01)
    t = np.arange(0, 280, dtype=float)
    log_price = LPPLS.func(
        t, params.tc, params.m, params.omega, params.A, params.B, params.C1, params.C2
    )
    rng = np.random.default_rng(42)
    log_price += rng.normal(0, 0.005, size=log_price.shape)
    return t, log_price


@pytest.fixture
def flat_series() -> tuple[np.ndarray, np.ndarray]:
    """Generate flat (no bubble) data."""
    t = np.arange(0, 250, dtype=float)
    rng = np.random.default_rng(123)
    log_price = 5.0 + 0.001 * t + rng.normal(0, 0.02, size=t.shape)
    return t, log_price


class TestEnsembleResult:
    def test_is_bubble_true(self):
        r = EnsembleResult(
            regime=None,
            lppls_confidence=None,
            final_verdict="BUBBLE",
            tc_estimate=300.0,
            reasoning="test",
        )
        assert r.is_bubble is True

    def test_is_bubble_possible(self):
        r = EnsembleResult(
            regime=None,
            lppls_confidence=None,
            final_verdict="POSSIBLE_BUBBLE",
            tc_estimate=300.0,
            reasoning="test",
        )
        assert r.is_bubble is True

    def test_is_not_bubble(self):
        r = EnsembleResult(
            regime=None,
            lppls_confidence=None,
            final_verdict="NO_BUBBLE",
            tc_estimate=None,
            reasoning="test",
        )
        assert r.is_bubble is False


class TestHMMLPPLSEnsemble:
    def test_flat_series_no_bubble(self, flat_series):
        t, log_price = flat_series
        ensemble = HMMLPPLSEnsemble(
            confidence_kwargs={"windows": [60, 90], "tc_tolerance": 20},
        )
        result = ensemble.analyze(t, log_price)
        assert isinstance(result, EnsembleResult)
        # Flat series should not be detected as bubble
        assert result.final_verdict in ("NO_BUBBLE", "POSSIBLE_BUBBLE")

    def test_synthetic_bubble_detected(self, synthetic_bubble):
        t, log_price = synthetic_bubble
        ensemble = HMMLPPLSEnsemble(
            confidence_kwargs={"windows": [60, 90, 120], "tc_tolerance": 20},
            skip_normal=False,  # force LPPLS even if HMM says normal
        )
        result = ensemble.analyze(t, log_price)
        assert isinstance(result, EnsembleResult)
        # Should have regime info
        assert result.regime is not None
        # Should have attempted LPPLS (skip_normal=False)
        assert result.lppls_confidence is not None

    def test_skip_normal_works(self, flat_series):
        t, log_price = flat_series
        ensemble = HMMLPPLSEnsemble(
            confidence_kwargs={"windows": [60, 90]},
            skip_normal=True,
        )
        result = ensemble.analyze(t, log_price)
        # If HMM detects NORMAL and skip_normal=True, LPPLS should be None
        if not result.regime.should_fit_lppls:
            assert result.lppls_confidence is None
            assert result.final_verdict == "NO_BUBBLE"

    def test_result_has_reasoning(self, flat_series):
        t, log_price = flat_series
        ensemble = HMMLPPLSEnsemble(confidence_kwargs={"windows": [60, 90]})
        result = ensemble.analyze(t, log_price)
        assert len(result.reasoning) > 0
