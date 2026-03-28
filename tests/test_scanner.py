"""Tests for LPPLS rolling window scanner."""

import numpy as np
import pytest

from src.lppls.model import LPPLS, LPPLSParams
from src.lppls.scanner import LPPLSScanner, ScanPoint


@pytest.fixture
def synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic LPPLS data for scanner testing."""
    params = LPPLSParams(tc=300.0, m=0.5, omega=8.0, A=10.0, B=-0.5, C1=0.02, C2=0.01)
    t = np.arange(0, 280, dtype=float)
    log_price = LPPLS.func(
        t, params.tc, params.m, params.omega, params.A, params.B, params.C1, params.C2
    )
    rng = np.random.default_rng(42)
    log_price += rng.normal(0, 0.005, size=log_price.shape)
    return t, log_price


class TestLPPLSScanner:
    def test_scan_returns_list(self, synthetic_data):
        t, log_price = synthetic_data
        scanner = LPPLSScanner(
            step=50,
            min_data_points=120,
            confidence_kwargs={"windows": [60, 90], "tc_tolerance": 20},
        )
        results = scanner.scan(t, log_price)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(sp, ScanPoint) for sp in results)

    def test_scan_t_end_increases(self, synthetic_data):
        t, log_price = synthetic_data
        scanner = LPPLSScanner(
            step=30,
            min_data_points=120,
            confidence_kwargs={"windows": [60, 90], "tc_tolerance": 20},
        )
        results = scanner.scan(t, log_price)
        t_ends = [sp.t_end for sp in results]
        assert t_ends == sorted(t_ends)

    def test_short_data_empty_scan(self):
        t = np.arange(50, dtype=float)
        log_price = np.log(100 + t)
        scanner = LPPLSScanner(min_data_points=120)
        results = scanner.scan(t, log_price)
        assert len(results) == 0

    def test_extract_confidence_series(self, synthetic_data):
        t, log_price = synthetic_data
        scanner = LPPLSScanner(
            step=50,
            min_data_points=120,
            confidence_kwargs={"windows": [60, 90], "tc_tolerance": 20},
        )
        results = scanner.scan(t, log_price)
        times, scores = LPPLSScanner.extract_confidence_series(results)
        assert len(times) == len(results)
        assert len(scores) == len(results)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_empty_series(self):
        times, scores = LPPLSScanner.extract_confidence_series([])
        assert len(times) == 0
        assert len(scores) == 0
