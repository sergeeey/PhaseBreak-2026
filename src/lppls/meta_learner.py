"""Ensemble meta-learner: combine multiple signals into single bubble probability.

Combines:
- LPPLS quality_score (soft scoring)
- HMM bubble_probability
- Changepoint strength (CUSUM)
- R² of LPPLS fit

Via logistic regression trained on train split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray

log = structlog.get_logger()


@dataclass
class MetaFeatures:
    """Feature vector for meta-learner."""

    quality_score: float
    hmm_bubble_prob: float
    changepoint_strength: float
    r_squared: float
    is_bubble_expected: bool | None = None  # label (None for inference)


@dataclass
class MetaPrediction:
    """Output of meta-learner."""

    probability: float
    predicted_bubble: bool
    features_used: dict[str, float]


class LogisticMetaLearner:
    """Simple logistic regression meta-learner.

    WHY not sklearn: avoid heavy dependency for 4 features.
    Logistic regression with gradient descent is trivial to implement.
    """

    def __init__(self, lr: float = 0.1, n_iter: int = 200, threshold: float = 0.5) -> None:
        self.lr = lr
        self.n_iter = n_iter
        self.threshold = threshold
        self.weights: NDArray | None = None
        self.bias: float = 0.0

    @staticmethod
    def _sigmoid(z: NDArray) -> NDArray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, features: list[MetaFeatures]) -> None:
        """Train on labeled feature vectors."""
        X = np.array(
            [
                [f.quality_score, f.hmm_bubble_prob, f.changepoint_strength, f.r_squared]
                for f in features
            ]
        )
        y = np.array([float(f.is_bubble_expected) for f in features])

        n, d = X.shape
        self.weights = np.zeros(d)
        self.bias = 0.0

        for _ in range(self.n_iter):
            z = X @ self.weights + self.bias
            pred = self._sigmoid(z)
            error = pred - y

            self.weights -= self.lr * (X.T @ error) / n
            self.bias -= self.lr * np.mean(error)

        log.info(
            "meta_learner_fitted",
            n_samples=n,
            weights=np.round(self.weights, 3).tolist(),
            bias=round(self.bias, 3),
        )

    def predict(self, features: MetaFeatures) -> MetaPrediction:
        """Predict bubble probability for a single episode."""
        if self.weights is None:
            return MetaPrediction(
                probability=features.quality_score,
                predicted_bubble=features.quality_score > self.threshold,
                features_used={},
            )

        x = np.array(
            [
                features.quality_score,
                features.hmm_bubble_prob,
                features.changepoint_strength,
                features.r_squared,
            ]
        )

        prob = float(self._sigmoid(x @ self.weights + self.bias))

        return MetaPrediction(
            probability=round(prob, 4),
            predicted_bubble=prob > self.threshold,
            features_used={
                "quality_score": features.quality_score,
                "hmm_bubble_prob": features.hmm_bubble_prob,
                "changepoint_strength": features.changepoint_strength,
                "r_squared": features.r_squared,
            },
        )
