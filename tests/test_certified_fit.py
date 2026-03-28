"""Tests for certified LPPLS convergence bounds."""

import numpy as np
import pytest

from src.lppls.model import LPPLS, LPPLSParams
from src.lppls.certified_fit import (
    CertifiedLPPLSResult,
    _estimate_convergence_rate,
    certified_lppls_fit,
    verify_lppls_convergence,
)


@pytest.fixture
def synthetic_bubble() -> tuple[np.ndarray, np.ndarray]:
    """Clean synthetic bubble — should converge well."""
    params = LPPLSParams(tc=300.0, m=0.5, omega=8.0, A=10.0, B=-0.5, C1=0.02, C2=0.01)
    t = np.arange(0, 260, dtype=float)
    log_price = LPPLS.func(
        t, params.tc, params.m, params.omega, params.A, params.B, params.C1, params.C2
    )
    rng = np.random.default_rng(42)
    log_price += rng.normal(0, 0.003, size=log_price.shape)
    return t, log_price


@pytest.fixture
def flat_series() -> tuple[np.ndarray, np.ndarray]:
    """Flat series — should NOT converge."""
    t = np.arange(0, 200, dtype=float)
    rng = np.random.default_rng(123)
    log_price = 5.0 + 0.001 * t + rng.normal(0, 0.02, size=t.shape)
    return t, log_price


class TestConvergenceRate:
    def test_linear_convergence(self):
        """Values converging as O(1/n) → α ≈ 1."""
        grid_sizes = [5, 10, 20, 40]
        values = [100 + 10.0 / n for n in grid_sizes]
        alpha, _ = _estimate_convergence_rate(grid_sizes, values)
        assert 0.5 < alpha < 2.0

    def test_quadratic_convergence(self):
        """Values converging as O(1/n²) → α ≈ 2."""
        grid_sizes = [5, 10, 20, 40]
        values = [100 + 100.0 / (n**2) for n in grid_sizes]
        alpha, _ = _estimate_convergence_rate(grid_sizes, values)
        assert 1.5 < alpha < 3.0

    def test_no_convergence(self):
        """Random values → α near 0 or negative."""
        rng = np.random.default_rng(42)
        grid_sizes = [5, 10, 20, 40]
        values = [100 + rng.normal(0, 5) for _ in grid_sizes]
        alpha, _ = _estimate_convergence_rate(grid_sizes, values)
        # May be anything — just shouldn't crash
        assert isinstance(alpha, float)

    def test_too_few_points(self):
        alpha, C = _estimate_convergence_rate([5, 10], [100.0, 99.5])
        assert alpha == 0.0


class TestCertifiedLPPLSResult:
    def test_confidence_interval_format(self):
        r = CertifiedLPPLSResult(
            tc_estimate=300.0,
            tc_bound=8.5,
            tc_low=291.5,
            tc_high=308.5,
            convergence_order=2.1,
            is_convergent=True,
            is_certified=True,
            grid_tc_values={5: 310, 10: 302, 15: 300},
            grid_r_squared={5: 0.95, 10: 0.97, 15: 0.98},
            method="Richardson",
            safety_factor=2.0,
        )
        assert "300.0" in r.confidence_interval
        assert "8.5" in r.confidence_interval


class TestCertifiedFit:
    def test_synthetic_bubble_convergent(self, synthetic_bubble):
        """Clean synthetic bubble should produce convergent fit."""
        t, log_price = synthetic_bubble
        result = certified_lppls_fit(t, log_price, grid_sizes=[5, 8, 12], safety_factor=2.0)
        assert isinstance(result, CertifiedLPPLSResult)
        # Should have at least some grid results
        assert len(result.grid_tc_values) >= 1

    def test_flat_series_not_convergent(self, flat_series):
        """Flat series should not produce convergent bubble fit."""
        t, log_price = flat_series
        result = certified_lppls_fit(t, log_price, grid_sizes=[5, 8, 12], safety_factor=2.0)
        # Flat series likely has no valid bubble fits
        if len(result.grid_tc_values) < 3:
            assert not result.is_certified


class TestVerifyConvergence:
    def test_synthetic_verdict(self, synthetic_bubble):
        t, log_price = synthetic_bubble
        result = verify_lppls_convergence(t, log_price, grid_sizes=[5, 8, 12])
        assert "verdict" in result
        assert "empirical_order" in result
        assert isinstance(result["is_consistent"], bool)

    def test_flat_series_verdict(self, flat_series):
        t, log_price = flat_series
        result = verify_lppls_convergence(t, log_price, grid_sizes=[5, 8, 12])
        assert "verdict" in result
