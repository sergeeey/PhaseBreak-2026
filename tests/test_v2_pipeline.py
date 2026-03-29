"""Tests for P0: pipeline separation, scoring, uncertainty, splits, adversarial, metrics."""

import numpy as np
import pytest
from src.lppls.model import LPPLS, LPPLSParams
from src.lppls.scoring import (
    compute_quality_score,
    score_m,
    score_B,
    score_omega,
    get_score_breakdown,
)
from src.lppls.uncertainty import bootstrap_tc_uncertainty
from src.pipeline.stages import run_screening, run_structural_fit, run_full_pipeline
from src.validation.splits import get_split, SplitProtocol
from src.validation.adversarial_controls import (
    generate_adversarial_series,
    run_adversarial_benchmark,
)
from src.validation.metrics_early_warning import compute_ew_metrics, summarize_ew_metrics


# ============ P0.1: Pipeline Separation ============
class TestPipelineLayers:
    def test_screening_ok(self):
        t = np.arange(100, dtype=float)
        values = 100 + t * 0.5 + np.random.default_rng(42).normal(0, 1, 100)
        result = run_screening(t, values)
        assert result.data_quality == "OK"
        assert result.n_points == 100

    def test_screening_too_short(self):
        result = run_screening(np.arange(5, dtype=float), np.array([1, 2, 3, 4, 5.0]))
        assert result.data_quality == "LOW_N"
        assert result.should_fit_lppls is False

    def test_full_pipeline_returns_result(self):
        params = LPPLSParams(tc=150, m=0.5, omega=8.0, A=4.6, B=-0.3, C1=0.01, C2=0.005)
        t = np.arange(120, dtype=float)
        values = np.exp(
            LPPLS.func(
                t, params.tc, params.m, params.omega, params.A, params.B, params.C1, params.C2
            )
        )
        values += np.random.default_rng(42).normal(0, 0.1, len(values))
        values = np.clip(values, 1, None)
        result = run_full_pipeline(t, values, grid_size=8)
        assert result.final_verdict in ("BUBBLE", "POSSIBLE", "NO_BUBBLE")
        assert result.screening.data_quality == "OK"


# ============ P0.2: Soft Scoring ============
class TestSoftScoring:
    def test_good_params_high_score(self):
        p = LPPLSParams(tc=300, m=0.5, omega=8.0, A=10, B=-0.5, C1=0.02, C2=0.01)
        score = compute_quality_score(p, r_squared=0.95)
        assert score > 0.5

    def test_bad_params_low_score(self):
        p = LPPLSParams(tc=300, m=0.9, omega=13.0, A=10, B=-0.001, C1=0.02, C2=0.01)
        score = compute_quality_score(p, r_squared=0.3)
        assert score < 0.3

    def test_none_params(self):
        assert compute_quality_score(None, 0.9) == 0.0

    def test_score_m_peak(self):
        assert score_m(0.5) > score_m(0.15)
        assert score_m(0.5) > score_m(0.85)

    def test_score_B_negative(self):
        assert score_B(-0.5) > score_B(-0.001)
        assert score_B(0.5) == 0.0

    def test_breakdown(self):
        p = LPPLSParams(tc=300, m=0.5, omega=8.0, A=10, B=-0.5, C1=0.02, C2=0.01)
        bd = get_score_breakdown(p, 0.95)
        assert "quality_total" in bd
        assert all(0 <= v <= 1 for v in bd.values())


# ============ P0.3: tc Uncertainty ============
class TestTCUncertainty:
    def test_bootstrap_returns_dict(self):
        params = LPPLSParams(tc=200, m=0.5, omega=8.0, A=4.6, B=-0.3, C1=0.01, C2=0.005)
        t = np.arange(150, dtype=float)
        lp = LPPLS.func(
            t, params.tc, params.m, params.omega, params.A, params.B, params.C1, params.C2
        )
        lp += np.random.default_rng(42).normal(0, 0.005, len(lp))
        result = bootstrap_tc_uncertainty(t, lp, n_bootstrap=10, grid_size=6)
        assert "tc_median" in result
        assert "tc_p10" in result
        assert "tc_p90" in result

    def test_uncertainty_interval_order(self):
        params = LPPLSParams(tc=200, m=0.5, omega=8.0, A=4.6, B=-0.3, C1=0.01, C2=0.005)
        t = np.arange(150, dtype=float)
        lp = LPPLS.func(
            t, params.tc, params.m, params.omega, params.A, params.B, params.C1, params.C2
        )
        lp += np.random.default_rng(42).normal(0, 0.005, len(lp))
        result = bootstrap_tc_uncertainty(t, lp, n_bootstrap=15, grid_size=6)
        if result["tc_p10"] is not None:
            assert result["tc_p10"] <= result["tc_median"] <= result["tc_p90"]


# ============ P0.4: Triple Split ============
class TestSplits:
    def test_finance_split(self):
        s = get_split("finance")
        assert isinstance(s, SplitProtocol)
        assert len(s.train) >= 2
        assert len(s.test) >= 2
        # No overlap
        assert not set(s.train) & set(s.test)

    def test_housing_split(self):
        s = get_split("housing")
        assert not set(s.train) & set(s.test)

    def test_commodities_split(self):
        s = get_split("commodities")
        assert len(s.train) >= 2


# ============ P0.5: Adversarial Controls ============
class TestAdversarial:
    def test_generate_cases(self):
        cases = generate_adversarial_series()
        assert len(cases) == 6
        n_positive = sum(1 for _, _, _, exp in cases if exp)
        assert n_positive == 1  # only real_lppls_signal

    def test_benchmark_with_dummy_detector(self):
        result = run_adversarial_benchmark(lambda t, v: False)  # always says NO
        assert result["fp"] == 0
        assert result["fn"] == 1  # missed the real signal

    def test_benchmark_structure(self):
        result = run_adversarial_benchmark(lambda t, v: True)  # always says YES
        assert result["tp"] == 1
        assert result["fp"] == 5  # all negatives flagged
        assert "precision" in result
        assert "adversarial_fp_rate" in result


# ============ P0.6: Early Warning Metrics ============
class TestEWMetrics:
    def test_compute_metrics(self):
        m = compute_ew_metrics(
            name="test",
            tc_estimate=300,
            tc_lower=290,
            tc_upper=310,
            actual_peak_idx=305,
            last_data_idx=280,
            quality_score=0.8,
            verdict="BUBBLE",
        )
        assert m.lead_time == 20
        assert m.tc_error == 5
        assert m.tc_interval_width == 20
        assert m.tc_interval_coverage is True

    def test_coverage_miss(self):
        m = compute_ew_metrics(
            name="test",
            tc_estimate=300,
            tc_lower=290,
            tc_upper=310,
            actual_peak_idx=320,
            last_data_idx=280,
            quality_score=0.5,
            verdict="POSSIBLE",
        )
        assert m.tc_interval_coverage is False

    def test_summarize(self):
        metrics = [
            compute_ew_metrics("a", 300, 290, 310, 305, 280, 0.8, "BUBBLE"),
            compute_ew_metrics("b", 200, 190, 210, 195, 180, 0.7, "BUBBLE"),
        ]
        s = summarize_ew_metrics(metrics)
        assert s["n_episodes"] == 2
        assert s["interval_coverage"] is not None
        assert 0 <= s["mean_quality_score"] <= 1
