"""Conformal prediction for LPPLS critical time intervals.

SECONDARY uncertainty method for PhaseBreak v2.
Requires a calibration set of |tc_predicted - tc_actual| errors from known
episodes. Provides coverage guarantee (1-alpha) but is limited to domains
where ground truth tc is available.

Primary method: bootstrap (uncertainty.py) — works standalone.

Converts point-estimate tc into calibrated prediction intervals:
  tc_estimate → (tc_lower, tc_upper) with coverage guarantee.

Method: split conformal prediction on historical residuals.
  1. Calibrate on known bubbles: compute |tc_predicted - tc_actual|
  2. Use quantile of calibration errors as interval half-width
  3. Coverage: 1-alpha (default 90%)

Does NOT modify LPPLS optimizer or fit logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray

log = structlog.get_logger()


@dataclass(frozen=True)
class TCInterval:
    """Calibrated prediction interval for critical time."""

    tc_estimate: float
    tc_lower: float
    tc_upper: float
    alpha: float  # significance level (0.1 = 90% coverage)
    half_width: float  # interval radius
    calibration_n: int  # number of calibration points used
    method: str = "split_conformal"


@dataclass
class TCCalibrator:
    """Fitted calibrator for tc prediction intervals."""

    errors: NDArray  # absolute calibration errors
    quantile_value: float  # q_{1-alpha} of errors
    alpha: float
    n_calibration: int

    def predict_interval(self, tc_estimate: float) -> TCInterval:
        """Compute prediction interval for a new tc estimate."""
        return TCInterval(
            tc_estimate=tc_estimate,
            tc_lower=tc_estimate - self.quantile_value,
            tc_upper=tc_estimate + self.quantile_value,
            alpha=self.alpha,
            half_width=self.quantile_value,
            calibration_n=self.n_calibration,
        )


def fit_tc_calibrator(
    errors: NDArray,
    alpha: float = 0.1,
) -> TCCalibrator:
    """Fit a conformal calibrator from historical tc prediction errors.

    Args:
        errors: Array of |tc_predicted - tc_actual| for known episodes
        alpha: Significance level (0.1 = 90% coverage target)

    Returns:
        TCCalibrator that can produce intervals for new predictions
    """
    if len(errors) == 0:
        log.warning("empty_calibration_set")
        return TCCalibrator(
            errors=np.array([]),
            quantile_value=float("inf"),
            alpha=alpha,
            n_calibration=0,
        )

    errors = np.abs(errors).astype(float)
    errors = errors[np.isfinite(errors)]

    if len(errors) == 0:
        return TCCalibrator(
            errors=np.array([]),
            quantile_value=float("inf"),
            alpha=alpha,
            n_calibration=0,
        )

    # Conformal quantile: ceil((n+1)(1-alpha))/n -th quantile
    n = len(errors)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    quantile_val = float(np.quantile(errors, q_level))

    log.info(
        "tc_calibrator_fitted",
        n=n,
        alpha=alpha,
        quantile=round(quantile_val, 2),
        q_level=round(q_level, 3),
    )

    return TCCalibrator(
        errors=errors,
        quantile_value=quantile_val,
        alpha=alpha,
        n_calibration=n,
    )


def calibrate_confidence(raw_score: float, calibrator: TCCalibrator) -> float:
    """Convert raw confidence score (0-1) into calibrated probability.

    Uses empirical CDF of calibration errors:
    calibrated = fraction of calibration errors that are <= predicted half-width * (1-raw_score)
    """
    if calibrator.n_calibration == 0 or raw_score <= 0:
        return 0.0

    # Higher raw_score → smaller expected error → higher calibrated confidence
    expected_error = calibrator.quantile_value * (1 - raw_score)
    calibrated = float(np.mean(calibrator.errors <= expected_error))
    return min(1.0, max(0.0, calibrated))
