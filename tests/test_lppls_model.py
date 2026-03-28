"""Tests for LPPLS model core math."""

import numpy as np
import pytest

from src.lppls.model import LPPLS, LPPLSParams


@pytest.fixture
def sample_params() -> LPPLSParams:
    """Known parameters for a synthetic bubble."""
    return LPPLSParams(tc=300.0, m=0.5, omega=8.0, A=10.0, B=-0.5, C1=0.02, C2=0.01)


@pytest.fixture
def synthetic_data(sample_params: LPPLSParams) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic LPPLS data with known parameters."""
    t = np.arange(0, 250, dtype=float)
    p = sample_params
    log_price = LPPLS.func(t, p.tc, p.m, p.omega, p.A, p.B, p.C1, p.C2)
    return t, log_price


class TestLPPLSParams:
    def test_is_bubble_negative_B(self):
        p = LPPLSParams(tc=100, m=0.5, omega=8, A=10, B=-0.5, C1=0.01, C2=0.01)
        assert p.is_bubble is True

    def test_is_not_bubble_positive_B(self):
        p = LPPLSParams(tc=100, m=0.5, omega=8, A=10, B=0.5, C1=0.01, C2=0.01)
        assert p.is_bubble is False

    def test_C_magnitude(self):
        p = LPPLSParams(tc=100, m=0.5, omega=8, A=10, B=-0.5, C1=0.03, C2=0.04)
        assert abs(p.C - 0.05) < 1e-10

    def test_phi_angle(self):
        p = LPPLSParams(tc=100, m=0.5, omega=8, A=10, B=-0.5, C1=1.0, C2=0.0)
        assert abs(p.phi) < 1e-10  # atan2(0, 1) = 0

    def test_damping_ratio(self):
        p = LPPLSParams(tc=100, m=0.5, omega=8, A=10, B=-1.0, C1=0.1, C2=0.0)
        assert p.damping == pytest.approx(10.0, rel=0.01)


class TestLPPLSFunc:
    def test_output_shape(self, sample_params):
        t = np.arange(0, 100, dtype=float)
        p = sample_params
        result = LPPLS.func(t, p.tc, p.m, p.omega, p.A, p.B, p.C1, p.C2)
        assert result.shape == t.shape

    def test_monotonic_near_tc(self, sample_params):
        """Near tc, the function should increase (bubble growth)."""
        p = sample_params
        t_near = np.array([280.0, 285.0, 290.0, 295.0])
        vals = LPPLS.func(t_near, p.tc, p.m, p.omega, p.A, p.B, p.C1, p.C2)
        # With B < 0 and m < 1, function increases as t → tc
        assert vals[-1] > vals[0]

    def test_no_nan_for_valid_input(self, sample_params):
        t = np.arange(0, 250, dtype=float)
        p = sample_params
        result = LPPLS.func(t, p.tc, p.m, p.omega, p.A, p.B, p.C1, p.C2)
        assert not np.any(np.isnan(result))


class TestOLS:
    def test_perfect_recovery(self, sample_params, synthetic_data):
        """OLS should perfectly recover linear params from noiseless data."""
        t, log_price = synthetic_data
        p = sample_params
        A, B, C1, C2 = LPPLS.solve_linear(t, log_price, p.tc, p.m, p.omega)
        assert A == pytest.approx(p.A, abs=1e-6)
        assert B == pytest.approx(p.B, abs=1e-6)
        assert C1 == pytest.approx(p.C1, abs=1e-6)
        assert C2 == pytest.approx(p.C2, abs=1e-6)

    def test_sse_zero_for_perfect_data(self, sample_params, synthetic_data):
        t, log_price = synthetic_data
        p = sample_params
        sse = LPPLS.sse_for_nonlinear(t, log_price, p.tc, p.m, p.omega)
        assert sse < 1e-10


class TestLPPLSModel:
    def test_predict_matches_func(self, sample_params, synthetic_data):
        t, log_price = synthetic_data
        model = LPPLS(params=sample_params)
        predicted = model.predict(t)
        np.testing.assert_allclose(predicted, log_price, atol=1e-8)

    def test_r_squared_perfect(self, sample_params, synthetic_data):
        t, log_price = synthetic_data
        model = LPPLS(params=sample_params)
        assert model.r_squared(t, log_price) == pytest.approx(1.0, abs=1e-8)

    def test_rmse_zero_for_perfect(self, sample_params, synthetic_data):
        t, log_price = synthetic_data
        model = LPPLS(params=sample_params)
        assert model.rmse(t, log_price) < 1e-8

    def test_predict_without_params_raises(self):
        model = LPPLS()
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(np.array([1.0, 2.0]))

    def test_noisy_data_r_squared(self, sample_params, synthetic_data):
        """With noise, R² should be high but not 1.0."""
        t, log_price = synthetic_data
        rng = np.random.default_rng(42)
        noisy = log_price + rng.normal(0, 0.01, size=log_price.shape)
        model = LPPLS(params=sample_params)
        r2 = model.r_squared(t, noisy)
        assert 0.9 < r2 < 1.0
