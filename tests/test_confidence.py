"""Tests for multi-window LPPLS confidence indicator."""

import numpy as np
import pytest

from src.lppls.model import LPPLS, LPPLSParams
from src.lppls.confidence import MultiWindowConfidence, ConfidenceResult


@pytest.fixture
def synthetic_bubble() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic bubble data with known LPPLS parameters."""
    params = LPPLSParams(tc=300.0, m=0.5, omega=8.0, A=10.0, B=-0.5, C1=0.02, C2=0.01)
    t = np.arange(0, 280, dtype=float)
    log_price = LPPLS.func(
        t, params.tc, params.m, params.omega, params.A, params.B, params.C1, params.C2
    )
    # Add small noise for realism
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


class TestConfidenceResult:
    def test_is_bubble_high(self):
        r = ConfidenceResult(
            tc_median=300.0,
            tc_std=5.0,
            confidence="HIGH",
            confidence_score=0.9,
            n_valid_windows=4,
            n_total_windows=5,
            n_agreeing=4,
        )
        assert r.is_bubble is True

    def test_is_bubble_no_signal(self):
        r = ConfidenceResult(
            tc_median=None,
            tc_std=None,
            confidence="NO_SIGNAL",
            confidence_score=0.0,
            n_valid_windows=0,
            n_total_windows=5,
            n_agreeing=0,
        )
        assert r.is_bubble is False

    def test_is_bubble_low(self):
        r = ConfidenceResult(
            tc_median=300.0,
            tc_std=50.0,
            confidence="LOW",
            confidence_score=0.2,
            n_valid_windows=1,
            n_total_windows=5,
            n_agreeing=1,
        )
        assert r.is_bubble is False


class TestMultiWindowConfidence:
    def test_synthetic_bubble_detected(self, synthetic_bubble):
        t, log_price = synthetic_bubble
        mwc = MultiWindowConfidence(windows=[60, 90, 120], tc_tolerance=20)
        result = mwc.evaluate(t, log_price)
        assert result.n_total_windows == 3
        assert result.n_valid_windows >= 1
        # Synthetic bubble should produce some signal
        assert result.confidence != "NO_SIGNAL" or result.n_valid_windows == 0

    def test_flat_series_no_bubble(self, flat_series):
        t, log_price = flat_series
        mwc = MultiWindowConfidence(windows=[60, 90, 120], tc_tolerance=15)
        result = mwc.evaluate(t, log_price)
        # Flat series should NOT detect high-confidence bubble
        assert result.confidence != "HIGH"

    def test_window_fits_populated(self, synthetic_bubble):
        t, log_price = synthetic_bubble
        mwc = MultiWindowConfidence(windows=[60, 90])
        result = mwc.evaluate(t, log_price)
        assert len(result.window_fits) == 2
        for wf in result.window_fits:
            assert wf.window_size in (60, 90)
            assert wf.t_end == int(t[-1])

    def test_short_data_skips_large_windows(self):
        t = np.arange(0, 50, dtype=float)
        log_price = np.log(100 + t)
        mwc = MultiWindowConfidence(windows=[60, 90, 120])
        result = mwc.evaluate(t, log_price)
        # All windows > len(t) should be skipped
        assert result.n_total_windows == 0

    def test_m_omega_populated_when_valid(self, synthetic_bubble):
        t, log_price = synthetic_bubble
        mwc = MultiWindowConfidence(windows=[90, 120])
        result = mwc.evaluate(t, log_price)
        if result.n_valid_windows > 0:
            assert result.m_mean is not None
            assert result.omega_mean is not None
