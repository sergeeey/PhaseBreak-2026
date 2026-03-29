"""Tests for conformal prediction / calibrated tc intervals."""

import numpy as np
import pytest

from src.lppls.conformal import (
    TCCalibrator,
    TCInterval,
    fit_tc_calibrator,
    calibrate_confidence,
)


class TestFitCalibrator:
    def test_basic_fit(self):
        errors = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        cal = fit_tc_calibrator(errors, alpha=0.1)
        assert isinstance(cal, TCCalibrator)
        assert cal.n_calibration == 5
        assert cal.quantile_value > 0

    def test_90_coverage(self):
        """Quantile should cover ~90% of calibration errors at alpha=0.1."""
        rng = np.random.default_rng(42)
        errors = rng.exponential(10, size=100)
        cal = fit_tc_calibrator(errors, alpha=0.1)
        coverage = np.mean(errors <= cal.quantile_value)
        assert coverage >= 0.85  # allow small margin

    def test_empty_errors(self):
        cal = fit_tc_calibrator(np.array([]), alpha=0.1)
        assert cal.quantile_value == float("inf")
        assert cal.n_calibration == 0

    def test_single_error(self):
        cal = fit_tc_calibrator(np.array([10.0]), alpha=0.1)
        assert cal.n_calibration == 1
        assert cal.quantile_value >= 10.0

    def test_infinite_errors_filtered(self):
        errors = np.array([5.0, np.inf, 10.0, np.nan, 15.0])
        cal = fit_tc_calibrator(errors, alpha=0.1)
        assert cal.n_calibration == 3


class TestTCInterval:
    def test_interval_ordering(self):
        errors = np.array([5, 10, 15, 20])
        cal = fit_tc_calibrator(errors, alpha=0.1)
        interval = cal.predict_interval(tc_estimate=300.0)
        assert isinstance(interval, TCInterval)
        assert interval.tc_lower <= interval.tc_estimate <= interval.tc_upper

    def test_interval_width(self):
        errors = np.array([5, 10, 15, 20])
        cal = fit_tc_calibrator(errors, alpha=0.1)
        interval = cal.predict_interval(tc_estimate=300.0)
        assert interval.half_width == interval.tc_upper - interval.tc_estimate
        assert interval.half_width == interval.tc_estimate - interval.tc_lower

    def test_wider_alpha_narrower_interval(self):
        errors = np.array([5, 10, 15, 20, 25])
        cal_90 = fit_tc_calibrator(errors, alpha=0.1)  # 90% coverage
        cal_80 = fit_tc_calibrator(errors, alpha=0.2)  # 80% coverage
        assert cal_80.quantile_value <= cal_90.quantile_value


class TestCalibrateConfidence:
    def test_zero_score(self):
        cal = fit_tc_calibrator(np.array([5, 10, 15]), alpha=0.1)
        assert calibrate_confidence(0.0, cal) == 0.0

    def test_high_score(self):
        cal = fit_tc_calibrator(np.array([5, 10, 15]), alpha=0.1)
        result = calibrate_confidence(0.9, cal)
        assert 0.0 <= result <= 1.0

    def test_empty_calibrator(self):
        cal = fit_tc_calibrator(np.array([]), alpha=0.1)
        assert calibrate_confidence(0.5, cal) == 0.0


class TestOnRealErrors:
    """Test with actual PhaseBreak tc errors from finance validation."""

    def test_finance_calibration(self):
        # Real tc errors from multi-window validation (days)
        finance_errors = np.array([1, 19, 10, 16])  # BTC17, BTC21, Dotcom, China
        cal = fit_tc_calibrator(finance_errors, alpha=0.1)

        # Predict interval for a new tc
        interval = cal.predict_interval(tc_estimate=350.0)
        print(f"\nFinance tc calibration (n={cal.n_calibration}):")
        print(f"  90% interval: tc = {interval.tc_estimate} ± {interval.half_width:.1f} days")
        print(f"  Range: [{interval.tc_lower:.1f}, {interval.tc_upper:.1f}]")

        # Interval should be reasonable (not infinite, not zero)
        assert 0 < interval.half_width < 100
