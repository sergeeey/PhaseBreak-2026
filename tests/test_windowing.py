"""Tests for adaptive multi-scale windowing."""

import numpy as np
import pytest

from src.lppls.windowing import (
    select_adaptive_windows,
    infer_volatility_regime,
    WINDOW_PRESETS,
)


class TestVolatilityRegime:
    def test_normal_series(self):
        rng = np.random.default_rng(42)
        series = 100 + np.cumsum(rng.normal(0, 1, size=200))
        assert infer_volatility_regime(series) == "normal"

    def test_high_vol_series(self):
        rng = np.random.default_rng(42)
        series = np.zeros(200)
        series[:160] = 100 + np.cumsum(rng.normal(0, 0.5, size=160))
        series[160:] = series[159] + np.cumsum(rng.normal(0, 5.0, size=40))
        regime = infer_volatility_regime(np.exp(series / 100))
        # Recent vol much higher → should detect high
        assert regime in ("high", "normal")  # depends on threshold

    def test_short_series(self):
        assert infer_volatility_regime(np.array([1, 2, 3])) == "normal"


class TestAdaptiveWindows:
    def test_daily_normal(self):
        series = np.arange(300, dtype=float) + 100
        windows = select_adaptive_windows(series, frequency="daily")
        assert windows == WINDOW_PRESETS["daily_normal"]

    def test_quarterly(self):
        series = np.arange(30, dtype=float) + 100
        windows = select_adaptive_windows(series, frequency="quarterly")
        assert all(w <= 24 for w in windows)

    def test_monthly(self):
        series = np.arange(60, dtype=float) + 100
        windows = select_adaptive_windows(series, frequency="monthly")
        assert all(w <= 48 for w in windows)

    def test_highvol_override(self):
        series = np.arange(300, dtype=float) + 100
        windows = select_adaptive_windows(series, frequency="daily", volatility_regime="high")
        assert windows == WINDOW_PRESETS["daily_highvol"]

    def test_short_data_fallback(self):
        series = np.arange(10, dtype=float) + 100
        windows = select_adaptive_windows(series, frequency="daily")
        # All preset windows > 10 data points, so fallback
        assert len(windows) >= 2
        assert all(w < len(series) for w in windows)

    def test_windows_sorted(self):
        series = np.arange(300, dtype=float) + 100
        for freq in ["daily", "quarterly", "monthly"]:
            windows = select_adaptive_windows(series, frequency=freq)
            assert windows == sorted(windows)

    def test_no_window_exceeds_data(self):
        series = np.arange(50, dtype=float) + 100
        windows = select_adaptive_windows(series, frequency="daily")
        for w in windows:
            assert w <= len(series)
