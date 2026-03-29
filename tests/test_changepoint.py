"""Tests for changepoint detector."""

import numpy as np
import pytest

from src.signals.changepoint import (
    detect_changepoint_strength,
    detect_recent_changepoint,
    ChangepointResult,
)


class TestChangepointStrength:
    def test_flat_series_low_score(self):
        rng = np.random.default_rng(42)
        series = 100 + rng.normal(0, 0.5, size=200)
        series = np.clip(series, 1, None)
        score = detect_changepoint_strength(series)
        assert score < 0.5

    def test_clear_changepoint_high_score(self):
        """Series with obvious regime change should score high."""
        rng = np.random.default_rng(42)
        part1 = 100 + rng.normal(0, 0.5, size=100)
        part2 = 120 + rng.normal(0, 3.0, size=100)
        series = np.concatenate([part1, part2])
        score = detect_changepoint_strength(series)
        assert score > 0.3

    def test_score_bounded(self):
        rng = np.random.default_rng(42)
        series = np.exp(np.cumsum(rng.normal(0, 0.02, size=300)))
        score = detect_changepoint_strength(series)
        assert 0.0 <= score <= 1.0

    def test_short_series(self):
        assert detect_changepoint_strength(np.array([1, 2, 3])) == 0.0


class TestDetectRecentChangepoint:
    def test_returns_result(self):
        rng = np.random.default_rng(42)
        series = np.exp(np.cumsum(rng.normal(0, 0.02, size=200)))
        result = detect_recent_changepoint(series)
        assert isinstance(result, ChangepointResult)
        assert 0.0 <= result.strength <= 1.0

    def test_location_in_range(self):
        rng = np.random.default_rng(42)
        series = np.exp(np.cumsum(rng.normal(0, 0.02, size=200)))
        result = detect_recent_changepoint(series)
        if result.location is not None:
            assert 0 <= result.location < len(series)

    def test_clear_regime_change(self):
        rng = np.random.default_rng(42)
        part1 = np.exp(np.cumsum(rng.normal(0.001, 0.01, size=100)))
        part2 = part1[-1] * np.exp(np.cumsum(rng.normal(-0.005, 0.05, size=100)))
        series = np.concatenate([part1, part2])
        result = detect_recent_changepoint(series)
        assert result.cusum_max > 0
        assert result.variance_ratio > 1.0

    def test_short_series_safe(self):
        result = detect_recent_changepoint(np.array([1.0, 2.0, 3.0]))
        assert result.strength == 0.0
        assert result.is_changepoint is False
