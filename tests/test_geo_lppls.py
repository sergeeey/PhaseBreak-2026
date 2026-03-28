"""Tests for geological LPPLS fitting on spectral data."""

import numpy as np
import pytest

from src.lppls.model import LPPLS, LPPLSParams
from src.geo.geo_lppls import (
    GeoLPPLSResult,
    fit_geo_lppls,
    fit_geo_multiwindow,
    compare_with_finance,
    GEO_BOUNDS,
)


def _make_synthetic_geo_series(
    tc: float = 200.0,
    m: float = 0.5,
    omega: float = 10.0,
    n_points: int = 150,
    noise: float = 0.003,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic LPPLS signal simulating spectral index evolution.

    Returns (t, values) where values are in spectral index range (0.2-0.8).
    """
    params = LPPLSParams(tc=tc, m=m, omega=omega, A=-0.3, B=-0.1, C1=0.005, C2=0.003)
    t = np.arange(0, n_points, dtype=float)
    log_vals = LPPLS.func(
        t, params.tc, params.m, params.omega, params.A, params.B, params.C1, params.C2
    )
    rng = np.random.default_rng(seed)
    log_vals += rng.normal(0, noise, size=log_vals.shape)
    # Convert to spectral index range (exp of small negative values → 0.5-0.9)
    values = np.exp(log_vals)
    return t, values


class TestGeoLPPLSFit:
    def test_synthetic_geo_signal(self):
        t, values = _make_synthetic_geo_series()
        result = fit_geo_lppls(t, values, index_name="ndvi_synthetic", grid_size=8)
        assert isinstance(result, GeoLPPLSResult)
        assert result.index_name == "ndvi_synthetic"
        assert result.r_squared > 0.5

    def test_flat_spectral_no_signal(self):
        rng = np.random.default_rng(123)
        t = np.arange(100, dtype=float)
        values = 0.5 + rng.normal(0, 0.01, size=100)
        values = np.clip(values, 0.01, 1.0)
        result = fit_geo_lppls(t, values, index_name="flat")
        assert isinstance(result, GeoLPPLSResult)

    def test_geo_bounds_wider_omega(self):
        """Geological omega range (4-25) is wider than finance (6-13)."""
        assert GEO_BOUNDS["omega_range"] == (4.0, 25.0)
        assert GEO_BOUNDS["m_range"] == (0.1, 0.9)

    def test_metrics_populated(self):
        t, values = _make_synthetic_geo_series()
        result = fit_geo_lppls(t, values, grid_size=8)
        assert result.aic < float("inf") or result.params is None
        assert result.bic < float("inf") or result.params is None
        assert 0.0 <= result.durbin_watson <= 4.0 or result.params is None


class TestGeoMultiWindow:
    def test_synthetic_multiwindow(self):
        t, values = _make_synthetic_geo_series(n_points=180)
        result = fit_geo_multiwindow(t, values, index_name="ndvi_mw", windows=[40, 60, 80])
        assert isinstance(result, GeoLPPLSResult)

    def test_short_series(self):
        t = np.arange(30, dtype=float)
        values = 0.5 + 0.001 * t
        result = fit_geo_multiwindow(t, values, windows=[10, 15, 20])
        assert isinstance(result, GeoLPPLSResult)


class TestCrossDomainComparison:
    def test_similar_distributions(self):
        """Same distribution → should detect similarity."""
        rng = np.random.default_rng(42)
        geo_m = rng.uniform(0.3, 0.7, size=20)
        geo_omega = rng.uniform(7.0, 12.0, size=20)
        fin_m = rng.uniform(0.3, 0.7, size=20)
        fin_omega = rng.uniform(7.0, 12.0, size=20)

        geo_results = [
            GeoLPPLSResult(
                index_name=f"test_{i}",
                params=None,
                r_squared=0.9,
                rmse=0.01,
                aic=0,
                bic=0,
                durbin_watson=2.0,
                is_phase_transition=True,
                tc_days=100,
                confidence=None,
                m=float(geo_m[i]),
                omega=float(geo_omega[i]),
            )
            for i in range(20)
        ]

        result = compare_with_finance(geo_results, fin_m, fin_omega)
        assert result["are_similar"] == True

    def test_different_distributions(self):
        """Different distributions → should detect difference."""
        rng = np.random.default_rng(42)
        geo_results = [
            GeoLPPLSResult(
                index_name=f"test_{i}",
                params=None,
                r_squared=0.9,
                rmse=0.01,
                aic=0,
                bic=0,
                durbin_watson=2.0,
                is_phase_transition=True,
                tc_days=100,
                confidence=None,
                m=float(rng.uniform(0.1, 0.3)),
                omega=float(rng.uniform(15.0, 25.0)),
            )
            for i in range(20)
        ]
        fin_m = rng.uniform(0.5, 0.9, size=20)
        fin_omega = rng.uniform(6.0, 10.0, size=20)

        result = compare_with_finance(geo_results, fin_m, fin_omega)
        assert result["are_similar"] == False

    def test_insufficient_data(self):
        result = compare_with_finance([], np.array([0.5, 0.6]), np.array([8.0, 9.0]))
        assert result["are_similar"] is None
        assert "insufficient" in result["reason"]
