"""Tests for Anokhin's Theory of Functional Systems (ТФС) implementation.

Tests the acceptor module:
1. Acceptor prediction (before fit)
2. Comparison with actual results
3. Targeted retry logic
4. Full TFS pipeline vs current pipeline
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.acceptor import (
    create_acceptor,
    compare_with_acceptor,
    run_tfs_pipeline_iteration,
    AcceptorPrediction,
    DOMAIN_ACCEPTOR_PRIORS,
)
from src.pipeline.stages import run_full_pipeline


# ---------------------------------------------------------------------------
# Test Acceptor Prediction
# ---------------------------------------------------------------------------


class TestAcceptorPrediction:
    """Test that acceptor creates reasonable predictions BEFORE fitting."""

    def test_finance_acceptor(self):
        """Finance domain should have tight, well-defined expectations."""
        np.random.seed(42)
        n = 200
        t = np.arange(n, dtype=float)
        values = np.random.randn(n).cumsum() + 100

        acceptor = create_acceptor("finance", t, values, hmm_bubble_prob=0.5)

        assert isinstance(acceptor, AcceptorPrediction)
        assert 0.1 < acceptor.m_expected[0] < acceptor.m_expected[1] < 0.9
        assert 4.0 < acceptor.omega_expected[0] < acceptor.omega_expected[1] < 15.0
        assert 0.5 < acceptor.min_r_squared < 1.0
        assert 0.0 < acceptor.confidence < 1.0
        assert acceptor.basis in ["domain_prior", "domain_prior + strong_hmm", "domain_prior + weak_hmm"]

    def test_commodities_acceptor(self):
        """Commodities should have wider expectations (noisier)."""
        np.random.seed(42)
        n = 150
        t = np.arange(n, dtype=float)
        values = np.random.randn(n).cumsum() + 100

        acceptor = create_acceptor("commodities", t, values)

        # Commodities should have lower R² bar than finance
        finance_acceptor = create_acceptor("finance", t, values)
        assert acceptor.min_r_squared <= finance_acceptor.min_r_squared

    def test_housing_acceptor(self):
        """Housing should have lowest bar (quarterly, few points)."""
        np.random.seed(42)
        n = 40  # Typical housing data (10 years quarterly)
        t = np.arange(n, dtype=float)
        values = np.random.randn(n).cumsum() + 100

        acceptor = create_acceptor("housing", t, values)

        # Housing should have lowest R² bar
        assert acceptor.min_r_squared < 0.75
        # But still reasonable bounds on parameters
        assert acceptor.m_expected[0] > 0.05
        assert acceptor.m_expected[1] < 0.85

    def test_strong_hmm_tightens_expectations(self):
        """High HMM bubble probability → expect stronger signal."""
        np.random.seed(42)
        n = 200
        t = np.arange(n, dtype=float)
        values = np.random.randn(n).cumsum() + 100

        acceptor_weak = create_acceptor("finance", t, values, hmm_bubble_prob=0.2)
        acceptor_strong = create_acceptor("finance", t, values, hmm_bubble_prob=0.8)

        # Strong HMM should have tighter m range
        weak_width = acceptor_weak.m_expected[1] - acceptor_weak.m_expected[0]
        strong_width = acceptor_strong.m_expected[1] - acceptor_strong.m_expected[0]
        assert strong_width <= weak_width

        # Strong HMM should have higher R² bar
        assert acceptor_strong.min_r_squared >= acceptor_weak.min_r_squared

    def test_historical_bayesian_update(self):
        """Historical fits should update acceptor predictions."""
        np.random.seed(42)
        n = 200
        t = np.arange(n, dtype=float)
        values = np.random.randn(n).cumsum() + 100

        # Simulate 5 historical bubble fits
        historical = [
            {"m": 0.45, "omega": 8.0, "is_bubble": True},
            {"m": 0.50, "omega": 9.0, "is_bubble": True},
            {"m": 0.42, "omega": 7.5, "is_bubble": True},
            {"m": 0.48, "omega": 8.5, "is_bubble": True},
            {"m": 0.55, "omega": 9.5, "is_bubble": True},
        ]

        acceptor = create_acceptor("finance", t, values, historical_params=historical)

        # m range should be centered around historical mean (0.48)
        m_center = (acceptor.m_expected[0] + acceptor.m_expected[1]) / 2
        assert 0.35 < m_center < 0.60  # Should be near 0.48

        # Confidence should increase with historical data
        assert acceptor.confidence > 0.5
        assert acceptor.basis == "historical_bayesian"


# ---------------------------------------------------------------------------
# Test Comparison with Acceptor
# ---------------------------------------------------------------------------


class TestAcceptorComparison:
    """Test that comparison correctly identifies matches/mismatches."""

    def test_perfect_match(self):
        """Result within acceptor bounds → perfect match."""
        acceptor = AcceptorPrediction(
            m_expected=(0.3, 0.6),
            omega_expected=(7.0, 10.0),
            tc_days_ahead=(30, 180),
            min_r_squared=0.85,
            min_quality=0.6,
            confidence=0.7,
            basis="domain_prior",
        )

        comparison = compare_with_acceptor(
            acceptor=acceptor,
            actual_m=0.45,
            actual_omega=8.5,
            actual_tc_idx=250,  # 50 days ahead of n=200
            actual_r_squared=0.92,
            actual_quality=0.75,
            actual_is_bubble=True,
            n_points=200,
        )

        assert comparison.acceptor_match is True
        assert len(comparison.mismatches) == 0
        assert comparison.mismatch_severity == "none"
        assert comparison.retry_needed is False
        assert comparison.satisfaction_score == 1.0

    def test_m_mismatch(self):
        """m outside expected range → mismatch."""
        acceptor = AcceptorPrediction(
            m_expected=(0.3, 0.6),
            omega_expected=(7.0, 10.0),
            tc_days_ahead=(30, 180),
            min_r_squared=0.85,
            min_quality=0.6,
            confidence=0.7,
            basis="domain_prior",
        )

        comparison = compare_with_acceptor(
            acceptor=acceptor,
            actual_m=0.15,  # Too low!
            actual_omega=8.5,
            actual_tc_idx=250,
            actual_r_squared=0.92,
            actual_quality=0.75,
            actual_is_bubble=True,
            n_points=200,
        )

        assert comparison.acceptor_match is False
        assert any("m=" in m for m in comparison.mismatches)
        assert comparison.retry_needed is True

    def test_multiple_mismatches(self):
        """Multiple parameter mismatches → low satisfaction."""
        acceptor = AcceptorPrediction(
            m_expected=(0.3, 0.6),
            omega_expected=(7.0, 10.0),
            tc_days_ahead=(30, 180),
            min_r_squared=0.85,
            min_quality=0.6,
            confidence=0.7,
            basis="domain_prior",
        )

        comparison = compare_with_acceptor(
            acceptor=acceptor,
            actual_m=0.15,  # Too low
            actual_omega=14.0,  # Too high
            actual_tc_idx=250,
            actual_r_squared=0.70,  # Below min
            actual_quality=0.40,  # Below min
            actual_is_bubble=True,
            n_points=200,
        )

        assert len(comparison.mismatches) >= 3
        assert comparison.mismatch_severity == "major"
        assert comparison.satisfaction_score < 0.3

    def test_retry_action_m_too_low(self):
        """m too low → widen lower bound."""
        acceptor = AcceptorPrediction(
            m_expected=(0.3, 0.6),
            omega_expected=(7.0, 10.0),
            tc_days_ahead=(30, 180),
            min_r_squared=0.85,
            min_quality=0.6,
            confidence=0.7,
            basis="domain_prior",
        )

        comparison = compare_with_acceptor(
            acceptor=acceptor,
            actual_m=0.12,  # Way too low
            actual_omega=8.5,
            actual_tc_idx=250,
            actual_r_squared=0.90,
            actual_quality=0.70,
            actual_is_bubble=True,
            n_points=200,
        )

        assert comparison.retry_action == "widen_m_range_lower"

    def test_retry_action_omega_too_high(self):
        """omega too high → widen upper bound."""
        acceptor = AcceptorPrediction(
            m_expected=(0.3, 0.6),
            omega_expected=(7.0, 10.0),
            tc_days_ahead=(30, 180),
            min_r_squared=0.85,
            min_quality=0.6,
            confidence=0.7,
            basis="domain_prior",
        )

        comparison = compare_with_acceptor(
            acceptor=acceptor,
            actual_m=0.45,
            actual_omega=14.5,  # Way too high
            actual_tc_idx=250,
            actual_r_squared=0.90,
            actual_quality=0.70,
            actual_is_bubble=True,
            n_points=200,
        )

        assert comparison.retry_action == "widen_omega_range_upper"


# ---------------------------------------------------------------------------
# Test Full TFS Pipeline
# ---------------------------------------------------------------------------


class TestTFSPipeline:
    """Test the full TFS pipeline (predict → fit → compare → retry)."""

    def test_tfs_converges_on_good_data(self):
        """TFS should accept result quickly on clear bubble-like data."""
        np.random.seed(42)
        n = 200
        t = np.arange(n, dtype=float)
        # Create bubble-like data: exponential growth + oscillations
        tc = n + 30
        values = np.exp(0.01 * t) + 0.1 * np.sin(0.5 * t) + 0.05 * np.random.randn(n)
        values = values * 100

        result = run_tfs_pipeline_iteration(
            t=t,
            values=values,
            domain="finance",
            max_iterations=3,
            hmm_bubble_prob=0.6,
        )

        assert result is not None
        assert "params" in result
        assert "iterations" in result
        assert result["iterations"] <= 3  # Should not exceed max

    def test_tfs_rejects_random_walk(self):
        """TFS should reject pure random walk (no bubble structure)."""
        np.random.seed(123)
        n = 200
        t = np.arange(n, dtype=float)
        values = np.random.randn(n).cumsum() + 100  # Pure random walk

        result = run_tfs_pipeline_iteration(
            t=t,
            values=values,
            domain="finance",
            max_iterations=2,
            hmm_bubble_prob=0.3,  # Low bubble probability
        )

        # Should either have low satisfaction or no acceptor match
        if result["params"] is not None:
            assert result["satisfaction"] < 0.8 or not result["acceptor_match"]

    def test_tfs_vs_standard_pipeline(self):
        """Compare TFS pipeline with standard pipeline."""
        np.random.seed(42)
        n = 200
        t = np.arange(n, dtype=float)
        values = np.exp(0.01 * t) * 100 + 5 * np.random.randn(n)

        # Standard pipeline
        std_result = run_full_pipeline(t, values, domain="finance")

        # TFS pipeline
        tfs_result = run_tfs_pipeline_iteration(
            t=t,
            values=values,
            domain="finance",
            max_iterations=2,
            hmm_bubble_prob=std_result.screening.hmm_bubble_prob,
        )

        # Both should produce results
        assert std_result is not None
        assert tfs_result is not None

        # TFS should track metadata about iterations
        assert "iterations" in tfs_result
        assert "satisfaction" in tfs_result

        # If TFS accepted the result, satisfaction should be high
        if tfs_result["acceptor_match"]:
            assert tfs_result["satisfaction"] > 0.7


# ---------------------------------------------------------------------------
# Test Domain-Specific Acceptors
# ---------------------------------------------------------------------------


class TestDomainAcceptors:
    """Test that all domain acceptors are properly configured."""

    @pytest.mark.parametrize("domain", ["finance", "commodities", "housing", "geology", "adversarial"])
    def test_domain_acceptor_exists(self, domain):
        """Each domain should have acceptor priors."""
        assert domain in DOMAIN_ACCEPTOR_PRIORS
        priors = DOMAIN_ACCEPTOR_PRIORS[domain]

        assert "m_range" in priors
        assert "omega_range" in priors
        assert "tc_days" in priors
        assert "min_r_squared" in priors
        assert "min_quality" in priors

    def test_adversarial_has_high_bar(self):
        """Adversarial controls should have very high acceptance bar."""
        adv = DOMAIN_ACCEPTOR_PRIORS["adversarial"]
        fin = DOMAIN_ACCEPTOR_PRIORS["finance"]

        # Adversarial should be harder to pass than finance
        assert adv["min_r_squared"] >= fin["min_r_squared"]
        assert adv["min_quality"] >= fin["min_quality"]


# ---------------------------------------------------------------------------
# Test TFS vs OODA Comparison
# ---------------------------------------------------------------------------


class TestTFSvsOODA:
    """Demonstrate the difference between TFS (anticipatory) and OODA (reactive)."""

    def test_tfs_has_prediction_before_action(self):
        """TFS creates acceptor BEFORE fitting (unlike OODA which starts with Observe)."""
        np.random.seed(42)
        n = 100
        t = np.arange(n, dtype=float)
        values = np.random.randn(n).cumsum() + 100

        # TFS: prediction comes first
        acceptor = create_acceptor("finance", t, values)
        assert acceptor is not None
        assert acceptor.m_expected is not None  # Has prediction BEFORE any fitting

    def test_tfs_knows_when_to_stop(self):
        """TFS has built-in satisfaction criterion (unlike OODA's continuous loop)."""
        acceptor = AcceptorPrediction(
            m_expected=(0.3, 0.6),
            omega_expected=(7.0, 10.0),
            tc_days_ahead=(30, 180),
            min_r_squared=0.85,
            min_quality=0.6,
            confidence=0.7,
            basis="domain_prior",
        )

        # Perfect match → satisfaction = 1.0 → done!
        comparison = compare_with_acceptor(
            acceptor=acceptor,
            actual_m=0.45,
            actual_omega=8.5,
            actual_tc_idx=150,
            actual_r_squared=0.92,
            actual_quality=0.75,
            actual_is_bubble=True,
            n_points=100,
        )

        assert comparison.satisfaction_score == 1.0
        assert comparison.retry_needed is False
        # This is the key difference from OODA: TFS KNOWS when to stop
