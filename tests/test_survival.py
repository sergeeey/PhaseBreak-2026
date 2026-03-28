"""Tests for Doomsday fraud survival analysis."""

import numpy as np
import pandas as pd
import pytest

from src.survival.synthetic_data import (
    generate_fraud_timelines,
    generate_observed_snapshot,
    FraudDatasetConfig,
)
from src.survival.doomsday import (
    doomsday_posterior,
    predict_remaining,
    fit_weibull_prior,
    weibull_pdf,
    batch_doomsday_features,
    DoomsdayPrediction,
)
from src.survival.fraud_survival import (
    fit_and_compare,
    add_doomsday_features,
    SurvivalResult,
)


class TestSyntheticData:
    def test_generates_dataframe(self):
        df = generate_fraud_timelines()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 500
        assert "lifetime" in df.columns
        assert "detected" in df.columns
        assert "velocity" in df.columns

    def test_censoring_rate(self):
        cfg = FraudDatasetConfig(n_schemes=1000, censoring_rate=0.3)
        df = generate_fraud_timelines(cfg)
        actual_rate = 1 - df["detected"].mean()
        assert 0.2 < actual_rate < 0.4  # ~30% ± tolerance

    def test_positive_lifetimes(self):
        df = generate_fraud_timelines()
        assert (df["lifetime"] > 0).all()

    def test_scheme_types(self):
        df = generate_fraud_timelines()
        assert set(df["scheme_type"].unique()).issubset({0, 1, 2})

    def test_observed_snapshot(self):
        df = generate_fraud_timelines(FraudDatasetConfig(n_schemes=100))
        snap = generate_observed_snapshot(df)
        assert "n_observed" in snap.columns
        assert "obs_fraction" in snap.columns
        assert (snap["n_observed"] >= 1).all()
        assert (snap["obs_fraction"] <= 1.0).all()


class TestWeibullPrior:
    def test_fit_returns_shape_scale(self):
        rng = np.random.default_rng(42)
        lifetimes = rng.weibull(1.5, size=200) * 200
        shape, scale = fit_weibull_prior(lifetimes)
        assert shape > 0
        assert scale > 0

    def test_weibull_pdf_integrates(self):
        n = np.arange(1, 1000, dtype=float)
        pdf = weibull_pdf(n, shape=1.5, scale=200)
        assert 0.9 < pdf.sum() < 1.1  # approximate integral


class TestDoomsdayPosterior:
    def test_posterior_sums_to_one(self):
        N, post = doomsday_posterior(50, shape=1.5, scale=200)
        assert abs(post.sum() - 1.0) < 1e-6

    def test_posterior_starts_at_n_observed(self):
        N, post = doomsday_posterior(100, shape=1.5, scale=200)
        assert N[0] == 100

    def test_early_observation_more_remaining(self):
        """Observing early (n=10) → more remaining predicted than late (n=500)."""
        pred_early = predict_remaining(10, shape=1.5, scale=200)
        pred_late = predict_remaining(500, shape=1.5, scale=200)
        assert pred_early.remaining_median > pred_late.remaining_median


class TestPredictRemaining:
    def test_returns_prediction(self):
        pred = predict_remaining(50, shape=1.5, scale=200)
        assert isinstance(pred, DoomsdayPrediction)
        assert pred.n_observed == 50
        assert pred.n_median >= 50
        assert pred.remaining_median >= 0

    def test_credible_interval_ordering(self):
        pred = predict_remaining(50, shape=1.5, scale=200)
        assert pred.n_lower <= pred.n_median <= pred.n_upper

    def test_late_observation_less_remaining(self):
        """Observing at n=500 → less remaining than n=10."""
        pred_early = predict_remaining(10, shape=1.5, scale=200)
        pred_late = predict_remaining(500, shape=1.5, scale=200)
        assert pred_late.remaining_median < pred_early.remaining_median

    def test_batch_features(self):
        n_obs = np.array([10, 50, 100, 200])
        features = batch_doomsday_features(n_obs, shape=1.5, scale=200)
        assert features.shape == (4, 3)
        assert np.all(np.isfinite(features))


class TestFraudSurvival:
    @pytest.fixture(scope="class")
    def fraud_data(self) -> pd.DataFrame:
        df = generate_fraud_timelines(FraudDatasetConfig(n_schemes=300, seed=42))
        df = generate_observed_snapshot(df)
        return df

    def test_add_doomsday_features(self, fraud_data):
        df = add_doomsday_features(fraud_data, weibull_shape=1.5, weibull_scale=200)
        assert "doomsday_pct" in df.columns
        assert "remaining_frac" in df.columns
        assert "log_remaining" in df.columns

    def test_fit_and_compare(self, fraud_data):
        result = fit_and_compare(fraud_data)
        assert isinstance(result, SurvivalResult)
        assert 0.4 < result.c_index_baseline < 1.0
        assert 0.4 < result.c_index_doomsday < 1.0
        assert result.n_samples > 0

    def test_doomsday_improves_or_matches(self, fraud_data):
        """Doomsday features should not hurt C-index significantly."""
        result = fit_and_compare(fraud_data)
        # Allow up to 2% degradation (regularization noise)
        assert result.c_index_doomsday >= result.c_index_baseline - 0.02

    def test_print_results(self, fraud_data):
        result = fit_and_compare(fraud_data)
        print("\n" + "=" * 70)
        print("FRAUD SURVIVAL — Cox Model Comparison")
        print("=" * 70)
        print(f"Samples: {result.n_samples}")
        print(f"Weibull prior: shape={result.weibull_shape:.3f}, scale={result.weibull_scale:.1f}")
        print(f"\nBaseline C-index:  {result.c_index_baseline:.4f}")
        print(f"Doomsday C-index:  {result.c_index_doomsday:.4f}")
        print(f"Improvement:       {result.improvement:+.4f} ({result.improvement_pct:+.2f}%)")
        print(f"\nBaseline coefficients: {result.baseline_coefficients}")
        print(f"Doomsday coefficients: {result.doomsday_coefficients}")
        print("=" * 70)
