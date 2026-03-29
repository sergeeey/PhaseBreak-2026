"""Wavelet spectral diagnostics for LPPLS validation.

IMPORTANT: wavelet frequency (ordinary time) ≠ LPPLS ω (log-time).
This module is an EXPLORATORY diagnostic, not a direct ω validator.

Uses CWT (Continuous Wavelet Transform) to:
1. Check if oscillatory structure exists in the series
2. Estimate oscillation stability (ridge consistency)
3. Provide auxiliary confidence modifier

Requires: scipy.signal.cwt
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray

log = structlog.get_logger()


@dataclass
class WaveletDiagnostics:
    """Wavelet analysis output."""

    dominant_period: float | None  # dominant oscillation period (data points)
    ridge_stability: float  # 0-1, how consistent the dominant frequency is
    spectral_energy: float  # total energy in oscillatory band
    n_significant_peaks: int  # number of significant spectral peaks
    has_oscillation: bool  # diagnostic: oscillatory structure detected


def extract_wavelet_diagnostics(
    series: NDArray,
    sampling_rate: float = 1.0,
    min_period: int = 5,
    max_period: int = 60,
) -> WaveletDiagnostics:
    """Extract wavelet-based spectral diagnostics from time series.

    Args:
        series: Time series (prices, indices)
        sampling_rate: Samples per unit time (1.0 for daily)
        min_period: Minimum oscillation period to consider
        max_period: Maximum oscillation period

    Returns:
        WaveletDiagnostics with oscillation characteristics
    """
    if len(series) < max_period:
        max_period = len(series) // 2

    if len(series) < 2 * min_period:
        return WaveletDiagnostics(
            dominant_period=None,
            ridge_stability=0.0,
            spectral_energy=0.0,
            n_significant_peaks=0,
            has_oscillation=False,
        )

    # Detrend
    t = np.arange(len(series), dtype=float)
    coeffs = np.polyfit(t, series, deg=1)
    detrended = series - np.polyval(coeffs, t)

    # CWT with Morlet wavelet (manual — scipy.signal.cwt removed in scipy 1.12+)
    widths = np.arange(min_period, max_period + 1)
    try:
        cwtm = np.zeros((len(widths), len(detrended)), dtype=complex)
        for i, w in enumerate(widths):
            # Morlet wavelet at scale w
            M = min(6 * w, len(detrended) - 1)
            if M < 3:
                continue
            t_wav = np.arange(-M // 2, M // 2 + 1, dtype=float) / max(w, 1)
            wavelet = np.exp(1j * 5 * t_wav) * np.exp(-0.5 * t_wav**2) * (np.pi**-0.25)
            cwtm[i] = np.convolve(detrended, wavelet, mode="same")
    except Exception:
        return WaveletDiagnostics(
            dominant_period=None,
            ridge_stability=0.0,
            spectral_energy=0.0,
            n_significant_peaks=0,
            has_oscillation=False,
        )

    power = np.abs(cwtm) ** 2

    # Dominant period: which width has max average power
    avg_power = power.mean(axis=1)
    dominant_idx = int(np.argmax(avg_power))
    dominant_period = float(widths[dominant_idx])

    # Ridge stability: how consistent is the peak across time
    peak_widths_per_time = np.argmax(power, axis=0)
    ridge_std = np.std(peak_widths_per_time)
    ridge_stability = max(0.0, 1.0 - ridge_std / len(widths))

    # Spectral energy (normalized)
    total_energy = float(np.sum(avg_power))
    noise_energy = float(np.median(avg_power)) * len(avg_power)
    spectral_ratio = total_energy / (noise_energy + 1e-10)

    # Count significant peaks (above 2× median)
    threshold = 2.0 * np.median(avg_power)
    n_peaks = int(np.sum(avg_power > threshold))

    has_osc = ridge_stability > 0.3 and n_peaks >= 1

    log.info(
        "wavelet_diagnostics",
        dominant_period=round(dominant_period, 1),
        ridge_stability=round(ridge_stability, 3),
        n_peaks=n_peaks,
        has_oscillation=has_osc,
    )

    return WaveletDiagnostics(
        dominant_period=dominant_period,
        ridge_stability=ridge_stability,
        spectral_energy=spectral_ratio,
        n_significant_peaks=n_peaks,
        has_oscillation=has_osc,
    )


def score_oscillation_stability(diag: WaveletDiagnostics) -> float:
    """Convert wavelet diagnostics to single stability score (0-1)."""
    if diag.dominant_period is None:
        return 0.0
    return float(
        np.clip(
            0.5 * diag.ridge_stability
            + 0.3 * min(1.0, diag.n_significant_peaks / 3)
            + 0.2 * min(1.0, diag.spectral_energy / 5),
            0,
            1,
        )
    )


def compare_lppls_wavelet(
    lppls_omega: float | None,
    wavelet_diag: WaveletDiagnostics,
) -> dict:
    """Compare LPPLS and wavelet findings (diagnostic only).

    NOTE: Direct frequency comparison is NOT valid (different time domains).
    This returns a qualitative agreement assessment.
    """
    if lppls_omega is None or wavelet_diag.dominant_period is None:
        return {"agreement": "INSUFFICIENT_DATA", "details": "missing parameters"}

    lppls_has_signal = True  # if we have omega, LPPLS found oscillation
    wavelet_has_signal = wavelet_diag.has_oscillation

    if lppls_has_signal and wavelet_has_signal:
        agreement = "BOTH_DETECT"
    elif lppls_has_signal and not wavelet_has_signal:
        agreement = "LPPLS_ONLY"
    elif not lppls_has_signal and wavelet_has_signal:
        agreement = "WAVELET_ONLY"
    else:
        agreement = "NEITHER"

    return {
        "agreement": agreement,
        "lppls_omega": lppls_omega,
        "wavelet_dominant_period": wavelet_diag.dominant_period,
        "wavelet_stability": wavelet_diag.ridge_stability,
        "caveat": "frequencies in different time domains — qualitative comparison only",
    }
