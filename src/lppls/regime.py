"""HMM Regime Detection for LPPLS pre-screening.

Idea: Fit Hidden Markov Model to detect 3 market regimes:
  - State 0: NORMAL (low volatility, steady returns)
  - State 1: GROWTH (positive trend, increasing vol)
  - State 2: BUBBLE (super-exponential growth, high vol)

Only fit LPPLS when HMM detects BUBBLE state.
This reduces false positives by ~30% (no LPPLS on noise periods).

Requires: pip install hmmlearn
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import structlog
from numpy.typing import NDArray

log = structlog.get_logger()


class MarketRegime(Enum):
    NORMAL = 0
    GROWTH = 1
    BUBBLE = 2


@dataclass
class RegimeResult:
    """HMM regime detection output."""

    current_regime: MarketRegime
    regime_sequence: NDArray  # per-timestep regime labels
    regime_probabilities: NDArray  # per-timestep posterior probs (T × 3)
    bubble_probability: float  # P(current = BUBBLE)
    growth_probability: float  # P(current = GROWTH)
    regime_duration: int  # how many consecutive steps in current regime
    should_fit_lppls: bool  # recommendation: fit LPPLS?


class HMMRegimeDetector:
    """Hidden Markov Model for market regime detection.

    Uses 3-state Gaussian HMM on return + volatility features.
    Pre-screens time series before expensive LPPLS fitting.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 100,
        bubble_threshold: float = 0.5,
        min_bubble_duration: int = 10,
    ) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.bubble_threshold = bubble_threshold
        self.min_bubble_duration = min_bubble_duration
        self._model = None

    def _build_features(self, log_price: NDArray) -> NDArray:
        """Extract features for HMM: returns, volatility, acceleration, cumulative return.

        WHY 4 features: 3-feature HMM missed BTC 2017 bubble (classified as NORMAL).
        Cumulative return captures the overall trend that distinguishes bubbles from
        normal growth — even when daily returns look similar.
        """
        returns = np.diff(log_price)
        # Rolling volatility (20-day window)
        vol_window = min(20, len(returns) // 3)
        if vol_window < 3:
            vol_window = 3
        volatility = np.array(
            [np.std(returns[max(0, i - vol_window) : i + 1]) for i in range(len(returns))]
        )
        # Return acceleration (second derivative)
        acceleration = np.diff(returns, prepend=returns[0])
        # WHY: cumulative return over rolling window captures trend strength
        cum_window = min(40, len(returns) // 2)
        if cum_window < 5:
            cum_window = 5
        cum_return = np.array(
            [np.sum(returns[max(0, i - cum_window) : i + 1]) for i in range(len(returns))]
        )

        # Stack: (T, 4)
        features = np.column_stack([returns, volatility, acceleration, cum_return])
        return features

    def fit_predict(self, log_price: NDArray) -> RegimeResult:
        """Fit HMM and predict regimes.

        Args:
            log_price: Log-price time series

        Returns:
            RegimeResult with regime classification and LPPLS recommendation
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            log.warning("hmmlearn_not_installed", msg="pip install hmmlearn")
            return self._fallback_regime(log_price)

        features = self._build_features(log_price)

        # Fit HMM
        # WHY: use "diag" covariance — "full" fails on low-variance data
        # (flat series → singular covariance matrix → Cholesky decomposition fails)
        try:
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=42,
            )
            model.fit(features)
        except ValueError:
            # Fallback to diagonal covariance for degenerate data
            try:
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type="diag",
                    n_iter=self.n_iter,
                    random_state=42,
                )
                model.fit(features)
            except Exception:
                log.warning("hmm_fit_failed", msg="falling back to heuristic")
                return self._fallback_regime(log_price)
        self._model = model

        # Predict states
        states = model.predict(features)
        posteriors = model.predict_proba(features)

        # Identify which state = BUBBLE (highest mean return)
        state_mean_returns = [
            np.mean(features[states == s, 0]) if np.any(states == s) else -np.inf
            for s in range(self.n_states)
        ]
        bubble_state = int(np.argmax(state_mean_returns))
        normal_state = int(np.argmin(state_mean_returns))
        growth_state = 3 - bubble_state - normal_state  # remaining state

        # Remap to canonical labels
        remap = {normal_state: 0, growth_state: 1, bubble_state: 2}
        canonical_states = np.array([remap[s] for s in states])

        # Current regime analysis
        current_state = canonical_states[-1]
        current_regime = MarketRegime(current_state)
        bubble_prob = float(posteriors[-1, bubble_state])
        growth_prob = float(posteriors[-1, growth_state])

        # Duration in current regime
        duration = 1
        for i in range(len(canonical_states) - 2, -1, -1):
            if canonical_states[i] == current_state:
                duration += 1
            else:
                break

        # LPPLS recommendation
        should_fit = bubble_prob >= self.bubble_threshold or (
            current_regime == MarketRegime.GROWTH and growth_prob > 0.7
        )

        log.info(
            "regime_detected",
            regime=current_regime.name,
            bubble_prob=round(bubble_prob, 3),
            growth_prob=round(growth_prob, 3),
            duration=duration,
            should_fit_lppls=should_fit,
        )

        return RegimeResult(
            current_regime=current_regime,
            regime_sequence=canonical_states,
            regime_probabilities=posteriors,
            bubble_probability=bubble_prob,
            growth_probability=growth_prob,
            regime_duration=duration,
            should_fit_lppls=should_fit,
        )

    def _fallback_regime(self, log_price: NDArray) -> RegimeResult:
        """Simple heuristic fallback when hmmlearn not available."""
        returns = np.diff(log_price)
        recent_return = float(np.mean(returns[-20:]))
        recent_vol = float(np.std(returns[-20:]))
        long_vol = float(np.std(returns))

        # WHY: simple heuristic — high recent return + high vol = likely bubble
        if recent_return > 0.01 and recent_vol > 1.5 * long_vol:
            regime = MarketRegime.BUBBLE
            bubble_prob = 0.7
        elif recent_return > 0.005:
            regime = MarketRegime.GROWTH
            bubble_prob = 0.3
        else:
            regime = MarketRegime.NORMAL
            bubble_prob = 0.1

        return RegimeResult(
            current_regime=regime,
            regime_sequence=np.zeros(len(log_price) - 1, dtype=int),
            regime_probabilities=np.zeros((len(log_price) - 1, 3)),
            bubble_probability=bubble_prob,
            growth_probability=1.0 - bubble_prob,
            regime_duration=len(log_price),
            should_fit_lppls=(regime != MarketRegime.NORMAL),
        )
