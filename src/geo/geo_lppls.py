"""LPPLS with geological bounds and spectral data adapter.

Applies LPPLS phase transition detection to Sentinel-2 temporal series.
Key differences from financial LPPLS:
- omega range wider (4-25) for geological oscillation timescales
- tc range longer (up to 2 years)
- Input is spectral index value (0-1) instead of log-price
- B > 0 can indicate geological "buildup" (not just B < 0 for bubble)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray

from src.lppls.model import LPPLS, LPPLSParams
from src.lppls.optimizer import LPPLSOptimizer
from src.lppls.confidence import MultiWindowConfidence, ConfidenceResult
from src.lppls.metrics import aic, bic, durbin_watson

log = structlog.get_logger()

# WHY: geological processes oscillate at different scales than financial
GEO_BOUNDS = {
    "m_range": (0.1, 0.9),
    "omega_range": (4.0, 25.0),  # wider than finance (6-13)
}


@dataclass
class GeoLPPLSResult:
    """LPPLS fit result for geological spectral data."""

    index_name: str
    params: LPPLSParams | None
    r_squared: float
    rmse: float
    aic: float
    bic: float
    durbin_watson: float
    is_phase_transition: bool  # detected log-periodic signal
    tc_days: float | None  # days from start of series to predicted event
    confidence: ConfidenceResult | None  # multi-window if available
    m: float | None
    omega: float | None


def fit_geo_lppls(
    t: NDArray,
    values: NDArray,
    index_name: str = "unknown",
    grid_size: int = 12,
    use_log: bool = True,
) -> GeoLPPLSResult:
    """Fit LPPLS to a geological spectral time series.

    Args:
        t: Time array (days since first observation)
        values: Spectral index values (e.g., NDVI 0-1)
        index_name: Name for logging
        grid_size: Optimizer grid size
        use_log: Whether to log-transform values (True for LPPLS standard)

    Returns:
        GeoLPPLSResult with fit quality metrics
    """
    # Prepare data
    if use_log:
        # WHY: LPPLS is defined on log-price. For spectral indices (0-1),
        # shift to positive range before log.
        safe_values = np.clip(values, 1e-6, None)
        y = np.log(safe_values)
    else:
        y = values.copy()

    # Fit with geological bounds
    optimizer = LPPLSOptimizer(
        m_range=GEO_BOUNDS["m_range"],
        omega_range=GEO_BOUNDS["omega_range"],
        grid_size=grid_size,
        n_best=5,
    )

    try:
        model = optimizer.fit(t, y)
    except Exception as e:
        log.warning("geo_lppls_fit_failed", index=index_name, error=str(e))
        return GeoLPPLSResult(
            index_name=index_name,
            params=None,
            r_squared=0.0,
            rmse=0.0,
            aic=float("inf"),
            bic=float("inf"),
            durbin_watson=0.0,
            is_phase_transition=False,
            tc_days=None,
            confidence=None,
            m=None,
            omega=None,
        )

    r2 = model.r_squared(t, y)
    rmse_val = model.rmse(t, y)
    residuals = model.residuals(t, y)
    n = len(t)
    sse = float(np.sum(residuals**2))
    aic_val = aic(n, 7, sse)
    bic_val = bic(n, 7, sse)
    dw = durbin_watson(residuals)

    p = model.params
    is_pt = p is not None and p.is_bubble and r2 > 0.5

    log.info(
        "geo_lppls_fit",
        index=index_name,
        r2=round(r2, 4),
        is_phase_transition=is_pt,
        m=round(p.m, 4) if p else None,
        omega=round(p.omega, 4) if p else None,
        tc=round(p.tc, 1) if p else None,
    )

    return GeoLPPLSResult(
        index_name=index_name,
        params=p,
        r_squared=r2,
        rmse=rmse_val,
        aic=aic_val,
        bic=bic_val,
        durbin_watson=dw,
        is_phase_transition=is_pt,
        tc_days=p.tc if p else None,
        confidence=None,
        m=p.m if p else None,
        omega=p.omega if p else None,
    )


def fit_geo_multiwindow(
    t: NDArray,
    values: NDArray,
    index_name: str = "unknown",
    windows: list[int] | None = None,
    use_log: bool = True,
) -> GeoLPPLSResult:
    """Fit LPPLS with multi-window confidence on geological data.

    Args:
        t: Time array (days)
        values: Spectral index values
        index_name: For logging
        windows: Window sizes in data points (not days)
        use_log: Log-transform values

    Returns:
        GeoLPPLSResult with multi-window confidence
    """
    if use_log:
        safe_values = np.clip(values, 1e-6, None)
        y = np.log(safe_values)
    else:
        y = values.copy()

    # Single fit first
    result = fit_geo_lppls(t, y, index_name=index_name, use_log=False)

    # Multi-window confidence
    win = windows or [20, 30, 40, 50]
    mwc = MultiWindowConfidence(
        windows=win,
        tc_tolerance=30,  # geological events: ±30 days tolerance
        min_r_squared=0.4,  # lower threshold for geological data (noisier)
        optimizer_kwargs={
            "m_range": GEO_BOUNDS["m_range"],
            "omega_range": GEO_BOUNDS["omega_range"],
        },
    )

    try:
        conf = mwc.evaluate(t, y)
        result = GeoLPPLSResult(
            index_name=result.index_name,
            params=result.params,
            r_squared=result.r_squared,
            rmse=result.rmse,
            aic=result.aic,
            bic=result.bic,
            durbin_watson=result.durbin_watson,
            is_phase_transition=result.is_phase_transition or conf.is_bubble,
            tc_days=conf.tc_median if conf.tc_median else result.tc_days,
            confidence=conf,
            m=conf.m_mean if conf.m_mean else result.m,
            omega=conf.omega_mean if conf.omega_mean else result.omega,
        )
    except Exception as e:
        log.warning("geo_multiwindow_failed", index=index_name, error=str(e))

    return result


def compare_with_finance(
    geo_results: list[GeoLPPLSResult],
    finance_m: NDArray,
    finance_omega: NDArray,
) -> dict:
    """Compare (m, ω) parameters between geological and financial fits.

    This is THE key cross-domain analysis for the paper.

    Returns dict with:
        - geo_m, geo_omega: arrays from geological fits
        - finance_m, finance_omega: passed in
        - ks_m, ks_omega: Kolmogorov-Smirnov test results
        - are_similar: bool (p > 0.05 → same distribution)
    """
    from scipy.stats import ks_2samp

    geo_m = np.array([r.m for r in geo_results if r.m is not None])
    geo_omega = np.array([r.omega for r in geo_results if r.omega is not None])

    if len(geo_m) < 3 or len(finance_m) < 3:
        return {
            "geo_m": geo_m,
            "geo_omega": geo_omega,
            "finance_m": finance_m,
            "finance_omega": finance_omega,
            "ks_m": None,
            "ks_omega": None,
            "are_similar": None,
            "reason": "insufficient data (need ≥3 fits per domain)",
        }

    ks_m = ks_2samp(geo_m, finance_m)
    ks_omega = ks_2samp(geo_omega, finance_omega)

    # p > 0.05 → distributions NOT significantly different → universal!
    are_similar = ks_m.pvalue > 0.05 and ks_omega.pvalue > 0.05

    log.info(
        "cross_domain_comparison",
        ks_m_p=round(ks_m.pvalue, 4),
        ks_omega_p=round(ks_omega.pvalue, 4),
        are_similar=are_similar,
    )

    return {
        "geo_m": geo_m,
        "geo_omega": geo_omega,
        "finance_m": finance_m,
        "finance_omega": finance_omega,
        "ks_m": {"statistic": ks_m.statistic, "pvalue": ks_m.pvalue},
        "ks_omega": {"statistic": ks_omega.statistic, "pvalue": ks_omega.pvalue},
        "are_similar": are_similar,
    }
