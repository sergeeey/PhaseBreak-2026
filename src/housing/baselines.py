"""Simple baselines for housing bubble detection.

Compares LPPLS with naive approaches:
1. Rolling CAGR spike — if annualized growth > threshold → bubble
2. Z-score of price acceleration — if z > threshold → bubble
3. Deviation from long-run trend — if > 2 std → bubble
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray

log = structlog.get_logger()


@dataclass
class BaselineResult:
    """Result of a baseline bubble detector."""

    method: str
    is_bubble: bool
    score: float  # higher = more bubble-like
    threshold: float


def rolling_cagr_detector(
    values: NDArray, window: int = 8, threshold: float = 0.10
) -> BaselineResult:
    """Detect bubble via rolling compound annual growth rate.

    For quarterly data: window=8 = 2 years.
    Threshold=0.10 = 10% annualized growth.
    """
    if len(values) < window + 1:
        return BaselineResult("CAGR", False, 0.0, threshold)

    # Annualized growth over last window quarters
    start_val = values[-window - 1]
    end_val = values[-1]
    if start_val <= 0:
        return BaselineResult("CAGR", False, 0.0, threshold)

    years = window / 4  # quarters to years
    cagr = (end_val / start_val) ** (1 / years) - 1
    return BaselineResult("CAGR", cagr > threshold, float(cagr), threshold)


def zscore_acceleration_detector(
    values: NDArray, window: int = 4, threshold: float = 2.0
) -> BaselineResult:
    """Detect bubble via z-score of price acceleration.

    Acceleration = second difference of log prices.
    If recent acceleration z-score > threshold → bubble signal.
    """
    if len(values) < window + 3:
        return BaselineResult("Z-Accel", False, 0.0, threshold)

    log_vals = np.log(values)
    returns = np.diff(log_vals)
    accel = np.diff(returns)

    if len(accel) < window:
        return BaselineResult("Z-Accel", False, 0.0, threshold)

    recent = accel[-window:]
    historical = accel[:-window] if len(accel) > window else accel

    mean_hist = np.mean(historical)
    std_hist = np.std(historical)
    if std_hist < 1e-10:
        return BaselineResult("Z-Accel", False, 0.0, threshold)

    z = (np.mean(recent) - mean_hist) / std_hist
    return BaselineResult("Z-Accel", z > threshold, float(z), threshold)


def trend_deviation_detector(values: NDArray, threshold_std: float = 2.0) -> BaselineResult:
    """Detect bubble via deviation from long-run linear trend.

    Fit linear trend, if last value > mean + threshold * std of residuals → bubble.
    """
    if len(values) < 10:
        return BaselineResult("trend_dev", False, 0.0, threshold_std)

    t = np.arange(len(values), dtype=float)
    log_vals = np.log(values)

    # Linear trend
    coeffs = np.polyfit(t, log_vals, deg=1)
    trend = np.polyval(coeffs, t)
    residuals = log_vals - trend

    std_res = np.std(residuals)
    if std_res < 1e-10:
        return BaselineResult("trend_dev", False, 0.0, threshold_std)

    last_dev = residuals[-1] / std_res
    return BaselineResult("trend_dev", last_dev > threshold_std, float(last_dev), threshold_std)


def run_all_baselines(values: NDArray) -> list[BaselineResult]:
    """Run all baseline detectors on a price series."""
    return [
        rolling_cagr_detector(values),
        zscore_acceleration_detector(values),
        trend_deviation_detector(values),
    ]
