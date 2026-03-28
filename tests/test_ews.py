"""Tests for Critical Slowing Down early warning signals."""

import numpy as np
import pytest

from src.ews.critical_slowing import compute_ews, EWSResult, _detrend
from src.lppls.model import LPPLS, LPPLSParams


class TestDetrend:
    def test_linear_removes_trend(self):
        t = np.arange(100, dtype=float)
        series = 5.0 + 0.1 * t
        residuals = _detrend(series, method="linear")
        assert abs(np.mean(residuals)) < 0.1
        assert np.std(residuals) < 0.01

    def test_none_returns_copy(self):
        series = np.array([1.0, 2.0, 3.0])
        result = _detrend(series, method="none")
        np.testing.assert_array_equal(result, series)


class TestConstantSeries:
    def test_no_slowing(self):
        """Constant series with noise → not classified as slowing."""
        rng = np.random.default_rng(42)
        series = 5.0 + rng.normal(0, 0.1, size=200)
        result = compute_ews(series, window=30, detrend="none")
        assert isinstance(result, EWSResult)
        # is_slowing requires BOTH tau_var > 0.3 AND tau_ac1 > 0.3
        assert result.is_slowing is False


class TestApproachingBifurcation:
    def test_rising_variance_detected(self):
        """Series with increasing variance → EWS should detect."""
        rng = np.random.default_rng(42)
        n = 300
        # Variance increases linearly with time
        series = np.zeros(n)
        for i in range(n):
            noise_scale = 0.01 + 0.1 * (i / n)  # 0.01 → 0.11
            series[i] = rng.normal(0, noise_scale)
        result = compute_ews(series, window=40)
        assert result.kendall_tau_variance > 0.3
        assert result.ews_score > 0.5

    def test_rising_ac1_detected(self):
        """AR(1) process with increasing autocorrelation → slowing down."""
        rng = np.random.default_rng(42)
        n = 300
        series = np.zeros(n)
        for i in range(1, n):
            rho = 0.1 + 0.8 * (i / n)  # AC1: 0.1 → 0.9
            series[i] = rho * series[i - 1] + rng.normal(0, 0.1)
        result = compute_ews(series, window=40)
        assert result.kendall_tau_ac1 > 0.2


class TestSyntheticBubble:
    def test_lppls_bubble_shows_slowing(self):
        """LPPLS bubble near tc should show CSD indicators."""
        params = LPPLSParams(tc=300, m=0.5, omega=8.0, A=10, B=-0.5, C1=0.02, C2=0.01)
        t = np.arange(0, 280, dtype=float)
        log_price = LPPLS.func(
            t, params.tc, params.m, params.omega, params.A, params.B, params.C1, params.C2
        )
        rng = np.random.default_rng(42)
        log_price += rng.normal(0, 0.01, size=log_price.shape)

        result = compute_ews(log_price, window=40)
        assert isinstance(result, EWSResult)
        # Bubble should show increasing variance as tc approaches
        assert result.ews_score > 0.3


class TestEWSScore:
    def test_score_range(self):
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, size=200)
        result = compute_ews(series, window=30)
        assert 0.0 <= result.ews_score <= 1.0

    def test_metadata(self):
        series = np.random.default_rng(42).normal(0, 1, size=200)
        result = compute_ews(series, window=30)
        assert result.window_size == 30
        assert result.n_points == 200


class TestRealBubblesEWS:
    """Test EWS on known bubbles vs negative controls using yfinance."""

    @pytest.fixture(scope="class")
    def ews_results(self) -> dict[str, EWSResult]:
        from src.lppls.data import KNOWN_BUBBLES, NEGATIVE_CONTROLS, load_yfinance

        results = {}

        for name, config in list(KNOWN_BUBBLES.items())[:3]:  # Top 3 bubbles
            try:
                ds = load_yfinance(**config)
                result = compute_ews(ds.log_price, window=min(40, len(ds.log_price) // 5))
                results[f"bubble_{name}"] = result
            except Exception:
                continue

        for name, config in list(NEGATIVE_CONTROLS.items())[:3]:  # Top 3 controls
            try:
                ds = load_yfinance(**config)
                result = compute_ews(ds.log_price, window=min(40, len(ds.log_price) // 5))
                results[f"control_{name}"] = result
            except Exception:
                continue

        return results

    def test_bubbles_higher_ews_than_controls(self, ews_results):
        """Known bubbles should generally have higher EWS scores than controls."""
        bubble_scores = [r.ews_score for k, r in ews_results.items() if k.startswith("bubble")]
        control_scores = [r.ews_score for k, r in ews_results.items() if k.startswith("control")]

        if bubble_scores and control_scores:
            # At least some bubbles should have higher EWS than control mean
            assert max(bubble_scores) > np.mean(control_scores)

    def test_print_ews_comparison(self, ews_results):
        print("\n" + "=" * 80)
        print("EARLY WARNING SIGNALS — Known Bubbles vs Negative Controls")
        print("=" * 80)
        print(f"{'Name':<30} {'τ_var':>7} {'τ_ac1':>7} {'EWS':>6} {'Slowing':>8}")
        print("-" * 80)
        for name, r in sorted(ews_results.items()):
            print(
                f"{name:<30} {r.kendall_tau_variance:>7.3f} {r.kendall_tau_ac1:>7.3f} "
                f"{r.ews_score:>6.3f} {'YES' if r.is_slowing else 'NO':>8}"
            )
        print("=" * 80)
