"""Tests for LPPLS model selection metrics."""

import numpy as np
import pytest

from src.lppls.metrics import aic, bic, durbin_watson, residual_normality


class TestAIC:
    def test_lower_sse_better(self):
        assert aic(100, 7, 10.0) < aic(100, 7, 100.0)

    def test_more_params_penalized(self):
        assert aic(100, 7, 50.0) < aic(100, 10, 50.0)

    def test_zero_sse_returns_inf(self):
        assert aic(100, 7, 0.0) == float("inf")

    def test_known_value(self):
        # n=100, k=7, sse=100 → 100*ln(1) + 14 = 14
        result = aic(100, 7, 100.0)
        assert result == pytest.approx(14.0, abs=0.01)


class TestBIC:
    def test_stricter_than_aic(self):
        # BIC penalizes more for large n
        a = aic(1000, 7, 50.0)
        b = bic(1000, 7, 50.0)
        assert b > a

    def test_lower_sse_better(self):
        assert bic(100, 7, 10.0) < bic(100, 7, 100.0)


class TestDurbinWatson:
    def test_no_autocorrelation(self):
        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 1, size=500)
        dw = durbin_watson(residuals)
        assert 1.5 < dw < 2.5

    def test_perfect_positive_autocorrelation(self):
        residuals = np.arange(100, dtype=float)
        dw = durbin_watson(residuals)
        assert dw < 0.5

    def test_perfect_fit_returns_2(self):
        residuals = np.zeros(100)
        dw = durbin_watson(residuals)
        assert dw == pytest.approx(2.0)


class TestResidualNormality:
    def test_normal_residuals(self):
        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 1, size=1000)
        result = residual_normality(residuals)
        assert abs(result["skewness"]) < 0.5
        assert abs(result["kurtosis"] - 3.0) < 1.0

    def test_short_data(self):
        result = residual_normality(np.array([1.0, 2.0]))
        assert "skewness" in result
        assert "kurtosis" in result
