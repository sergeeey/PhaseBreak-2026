"""Tests for HMM regime detection."""

import numpy as np
import pytest

from src.lppls.regime import HMMRegimeDetector, MarketRegime, RegimeResult


@pytest.fixture
def bubble_like_series() -> np.ndarray:
    """Exponentially growing series (bubble-like)."""
    t = np.arange(200, dtype=float)
    rng = np.random.default_rng(42)
    log_price = 4.0 + 0.005 * t + 0.0001 * t**1.5 + rng.normal(0, 0.02, size=t.shape)
    return log_price


@pytest.fixture
def flat_series() -> np.ndarray:
    """Flat series (no trend)."""
    rng = np.random.default_rng(123)
    return 5.0 + rng.normal(0, 0.01, size=200)


class TestRegimeResult:
    def test_regime_enum(self):
        assert MarketRegime.NORMAL.value == 0
        assert MarketRegime.GROWTH.value == 1
        assert MarketRegime.BUBBLE.value == 2


class TestHMMFallback:
    """Test heuristic fallback when hmmlearn not installed."""

    def test_fallback_bubble_detection(self, bubble_like_series):
        detector = HMMRegimeDetector()
        result = detector._fallback_regime(bubble_like_series)
        assert isinstance(result, RegimeResult)
        assert result.current_regime in (MarketRegime.GROWTH, MarketRegime.BUBBLE)
        assert result.should_fit_lppls is True

    def test_fallback_flat_series(self, flat_series):
        detector = HMMRegimeDetector()
        result = detector._fallback_regime(flat_series)
        assert isinstance(result, RegimeResult)
        assert result.current_regime == MarketRegime.NORMAL
        assert result.should_fit_lppls is False

    def test_fallback_has_probabilities(self, bubble_like_series):
        detector = HMMRegimeDetector()
        result = detector._fallback_regime(bubble_like_series)
        assert 0.0 <= result.bubble_probability <= 1.0
        assert 0.0 <= result.growth_probability <= 1.0


class TestHMMRegimeDetector:
    def test_fit_predict_returns_result(self, bubble_like_series):
        detector = HMMRegimeDetector()
        result = detector.fit_predict(bubble_like_series)
        assert isinstance(result, RegimeResult)
        assert result.regime_duration >= 1
        assert len(result.regime_sequence) == len(bubble_like_series) - 1

    def test_flat_series_not_bubble(self, flat_series):
        detector = HMMRegimeDetector()
        result = detector.fit_predict(flat_series)
        # Flat series should mostly be NORMAL
        assert result.bubble_probability < 0.8

    def test_should_fit_lppls_flag(self, bubble_like_series):
        detector = HMMRegimeDetector(bubble_threshold=0.3)
        result = detector.fit_predict(bubble_like_series)
        assert isinstance(result.should_fit_lppls, bool)
