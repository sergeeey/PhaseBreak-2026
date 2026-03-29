"""Tests for wavelet spectral diagnostics."""

import numpy as np
import pytest

from src.signals.wavelet_lppls import (
    extract_wavelet_diagnostics,
    score_oscillation_stability,
    compare_lppls_wavelet,
    WaveletDiagnostics,
)


class TestWaveletDiagnostics:
    def test_oscillatory_series(self):
        """Series with clear oscillation should be detected."""
        t = np.arange(200, dtype=float)
        series = (
            100 + 5 * np.sin(2 * np.pi * t / 20) + np.random.default_rng(42).normal(0, 0.5, 200)
        )
        diag = extract_wavelet_diagnostics(series)
        assert isinstance(diag, WaveletDiagnostics)
        assert diag.dominant_period is not None
        assert diag.n_significant_peaks >= 1

    def test_flat_series(self):
        rng = np.random.default_rng(42)
        series = 100 + rng.normal(0, 0.1, size=200)
        diag = extract_wavelet_diagnostics(series)
        assert diag.ridge_stability < 0.8

    def test_short_series(self):
        diag = extract_wavelet_diagnostics(np.array([1.0, 2.0, 3.0]))
        assert diag.dominant_period is None
        assert diag.has_oscillation is False

    def test_score_bounded(self):
        t = np.arange(200, dtype=float)
        series = 100 + 3 * np.sin(2 * np.pi * t / 15)
        diag = extract_wavelet_diagnostics(series)
        score = score_oscillation_stability(diag)
        assert 0.0 <= score <= 1.0

    def test_score_empty(self):
        diag = WaveletDiagnostics(None, 0.0, 0.0, 0, False)
        assert score_oscillation_stability(diag) == 0.0


class TestCompareLPPLSWavelet:
    def test_both_detect(self):
        diag = WaveletDiagnostics(20.0, 0.8, 3.0, 2, True)
        result = compare_lppls_wavelet(lppls_omega=8.0, wavelet_diag=diag)
        assert result["agreement"] == "BOTH_DETECT"
        assert "caveat" in result

    def test_lppls_only(self):
        diag = WaveletDiagnostics(20.0, 0.1, 0.5, 0, False)
        result = compare_lppls_wavelet(lppls_omega=8.0, wavelet_diag=diag)
        assert result["agreement"] == "LPPLS_ONLY"

    def test_insufficient(self):
        diag = WaveletDiagnostics(None, 0.0, 0.0, 0, False)
        result = compare_lppls_wavelet(lppls_omega=None, wavelet_diag=diag)
        assert result["agreement"] == "INSUFFICIENT_DATA"
