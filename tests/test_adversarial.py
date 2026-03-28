"""Tests for adversarial AI council validation."""

import numpy as np
import pytest
import structlog

from src.lppls.model import LPPLS, LPPLSParams
from src.validation.council_validator import (
    run_council,
    CouncilVerdict,
    AgentRole,
    _heuristic_opinion,
    _heuristic_arbiter,
)
from src.validation.experiment_runner import (
    run_experiment,
    generate_synthetic_signals,
    ExperimentSummary,
)

log = structlog.get_logger()


# Good bubble params (should get BUBBLE verdict)
GOOD_BUBBLE = LPPLSParams(tc=300, m=0.5, omega=8.0, A=10, B=-0.5, C1=0.02, C2=0.01)
# Bad params (boundary m, tiny B — should get NO_BUBBLE)
BAD_PARAMS = LPPLSParams(tc=300, m=0.9, omega=13.0, A=10, B=-0.001, C1=0.02, C2=0.01)


class TestHeuristicOpinions:
    def test_bull_good_bubble(self):
        opinion = _heuristic_opinion(AgentRole.BULL, GOOD_BUBBLE, r_squared=0.95, durbin_watson=1.8)
        assert opinion.verdict == "BUBBLE"
        assert opinion.confidence >= 0.5

    def test_bear_bad_params(self):
        opinion = _heuristic_opinion(AgentRole.BEAR, BAD_PARAMS, r_squared=0.95, durbin_watson=0.5)
        assert opinion.verdict == "NO_BUBBLE"
        assert len(opinion.key_arguments) >= 2

    def test_skeptic_boundary_m(self):
        opinion = _heuristic_opinion(
            AgentRole.SKEPTIC, BAD_PARAMS, r_squared=0.95, durbin_watson=0.5
        )
        assert opinion.verdict == "UNCERTAIN"
        assert any("boundary" in a for a in opinion.key_arguments)

    def test_no_params(self):
        opinion = _heuristic_opinion(AgentRole.BULL, None, r_squared=0.0, durbin_watson=2.0)
        assert opinion.verdict == "NO_BUBBLE"
        assert opinion.confidence == 0.9


class TestHeuristicArbiter:
    def test_unanimous_bubble(self):
        from src.validation.council_validator import AgentOpinion

        opinions = [
            AgentOpinion(
                role=AgentRole.BULL,
                verdict="BUBBLE",
                confidence=0.9,
                reasoning="",
                key_arguments=[],
            ),
            AgentOpinion(
                role=AgentRole.BEAR,
                verdict="BUBBLE",
                confidence=0.7,
                reasoning="",
                key_arguments=[],
            ),
            AgentOpinion(
                role=AgentRole.SKEPTIC,
                verdict="BUBBLE",
                confidence=0.6,
                reasoning="",
                key_arguments=[],
            ),
        ]
        verdict, conf, _ = _heuristic_arbiter(opinions)
        assert verdict == "BUBBLE"
        assert conf > 0.7

    def test_majority_no_bubble(self):
        from src.validation.council_validator import AgentOpinion

        opinions = [
            AgentOpinion(
                role=AgentRole.BULL,
                verdict="BUBBLE",
                confidence=0.6,
                reasoning="",
                key_arguments=[],
            ),
            AgentOpinion(
                role=AgentRole.BEAR,
                verdict="NO_BUBBLE",
                confidence=0.8,
                reasoning="",
                key_arguments=[],
            ),
            AgentOpinion(
                role=AgentRole.SKEPTIC,
                verdict="NO_BUBBLE",
                confidence=0.7,
                reasoning="",
                key_arguments=[],
            ),
        ]
        verdict, conf, _ = _heuristic_arbiter(opinions)
        assert verdict == "NO_BUBBLE"


class TestCouncilHeuristic:
    def test_good_bubble_verdict(self):
        verdict = run_council(
            GOOD_BUBBLE,
            r_squared=0.95,
            durbin_watson=1.8,
            n_observations=200,
            model="__heuristic__",
        )
        assert isinstance(verdict, CouncilVerdict)
        assert verdict.final_verdict in ("BUBBLE", "NO_BUBBLE", "UNCERTAIN")
        assert len(verdict.opinions) == 3

    def test_bad_params_verdict(self):
        verdict = run_council(
            BAD_PARAMS, r_squared=0.5, durbin_watson=0.5, n_observations=50, model="__heuristic__"
        )
        assert isinstance(verdict, CouncilVerdict)
        assert verdict.final_verdict == "NO_BUBBLE"

    def test_no_params_verdict(self):
        verdict = run_council(
            None, r_squared=0.0, durbin_watson=2.0, n_observations=100, model="__heuristic__"
        )
        assert verdict.final_verdict == "NO_BUBBLE"


class TestSyntheticSignals:
    def test_generates_correct_counts(self):
        signals = generate_synthetic_signals(n_real=10, n_noise=10)
        assert len(signals) == 20
        n_real = sum(1 for _, _, is_real in signals if is_real)
        n_noise = sum(1 for _, _, is_real in signals if not is_real)
        assert n_real == 10
        assert n_noise == 10

    def test_signal_shapes(self):
        signals = generate_synthetic_signals(n_real=5, n_noise=5)
        for t, log_price, _ in signals:
            assert len(t) == len(log_price)
            assert len(t) >= 50


class TestExperiment:
    def test_small_heuristic_experiment(self):
        """Run small experiment with heuristic council (no Ollama)."""
        summary = run_experiment(n_real=10, n_noise=10, use_ollama=False, seed=42)
        assert isinstance(summary, ExperimentSummary)
        assert summary.n_total == 20
        assert summary.lppls_tp + summary.lppls_fn == 10
        assert summary.lppls_fp + summary.lppls_tn == 10

    def test_print_experiment(self):
        summary = run_experiment(n_real=20, n_noise=20, use_ollama=False, seed=42)
        print("\n" + "=" * 70)
        print("ADVERSARIAL COUNCIL EXPERIMENT (heuristic, 40 signals)")
        print("=" * 70)
        print(f"\n{'':>30} {'LPPLS':>10} {'Council':>10}")
        print(f"{'True Positives':>30} {summary.lppls_tp:>10} {summary.council_tp:>10}")
        print(f"{'False Positives':>30} {summary.lppls_fp:>10} {summary.council_fp:>10}")
        print(f"{'False Negatives':>30} {summary.lppls_fn:>10} {summary.council_fn:>10}")
        print(f"{'True Negatives':>30} {summary.lppls_tn:>10} {summary.council_tn:>10}")
        print(
            f"{'Precision':>30} {summary.lppls_precision:>10.1%} {summary.council_precision:>10.1%}"
        )
        print(f"{'Recall':>30} {summary.lppls_recall:>10.1%} {summary.council_recall:>10.1%}")
        print(
            f"\nPrecision improvement: {summary.precision_improvement:+.1%} ({summary.precision_improvement_pct:+.1f}%)"
        )
        print("=" * 70)
