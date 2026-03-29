"""Meta-calibration: convert raw quality scores into calibrated probabilities.

Uses isotonic regression on historical (quality_score, actual_outcome) pairs
to learn monotonic mapping: quality → P(actual bubble).

This ensures quality_score=0.7 really means ~70% of episodes with that score
were actual bubbles, not just "passes our filters well".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray

log = structlog.get_logger()


@dataclass
class CalibrationResult:
    """Output of calibration."""

    raw_score: float
    calibrated_prob: float
    n_calibration: int
    method: str


def _isotonic_fit(scores: NDArray, outcomes: NDArray) -> tuple[NDArray, NDArray]:
    """Fit isotonic regression (pool adjacent violators).

    Returns sorted (score_bins, calibrated_probs) for lookup.
    """
    order = np.argsort(scores)
    scores_sorted = scores[order]
    outcomes_sorted = outcomes[order].astype(float)

    # Pool Adjacent Violators Algorithm (PAVA)
    n = len(scores_sorted)
    calibrated = outcomes_sorted.copy()
    weights = np.ones(n)

    i = 0
    while i < n - 1:
        if calibrated[i] > calibrated[i + 1]:
            # Merge blocks
            j = i + 1
            while j < n and calibrated[j] <= calibrated[i]:
                j += 1
            # Pool i..j-1
            total_weight = weights[i:j].sum()
            pooled = np.sum(calibrated[i:j] * weights[i:j]) / total_weight
            calibrated[i:j] = pooled
            weights[i:j] = total_weight / (j - i)
            # Backtrack
            if i > 0:
                i -= 1
            else:
                i = j
        else:
            i += 1

    return scores_sorted, calibrated


class MetaCalibrator:
    """Calibrates quality scores against observed outcomes.

    Train on (quality_score, was_actual_bubble) pairs from historical episodes.
    Then use to convert new quality scores → calibrated probabilities.
    """

    def __init__(self) -> None:
        self._score_bins: NDArray | None = None
        self._calibrated_probs: NDArray | None = None
        self._n_calibration: int = 0

    def fit(self, scores: NDArray, outcomes: NDArray) -> None:
        """Fit calibrator on historical data.

        Args:
            scores: quality scores from pipeline (0-1)
            outcomes: binary labels (1=actual bubble, 0=not)
        """
        scores = np.asarray(scores, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)

        if len(scores) < 3:
            log.warning("calibration_too_few", n=len(scores))
            self._n_calibration = len(scores)
            return

        self._score_bins, self._calibrated_probs = _isotonic_fit(scores, outcomes)
        self._n_calibration = len(scores)

        log.info(
            "calibrator_fitted",
            n=len(scores),
            score_range=f"[{scores.min():.2f}, {scores.max():.2f}]",
            outcome_rate=f"{outcomes.mean():.2f}",
        )

    def calibrate(self, raw_score: float) -> CalibrationResult:
        """Convert raw quality score to calibrated probability."""
        if self._score_bins is None or len(self._score_bins) < 2:
            return CalibrationResult(
                raw_score=raw_score,
                calibrated_prob=raw_score,  # passthrough if not fitted
                n_calibration=self._n_calibration,
                method="passthrough",
            )

        # Interpolate
        calibrated = float(np.interp(raw_score, self._score_bins, self._calibrated_probs))
        calibrated = max(0.0, min(1.0, calibrated))

        return CalibrationResult(
            raw_score=round(raw_score, 4),
            calibrated_prob=round(calibrated, 4),
            n_calibration=self._n_calibration,
            method="isotonic",
        )


def build_finance_calibrator() -> MetaCalibrator:
    """Build calibrator from known finance benchmark results.

    Uses hardcoded results from v2 benchmark (train split only, not test).
    """
    # WHY: only use train split episodes (from splits.py) to avoid data leakage
    # Train: dotcom_2000, china_2015, sp500_2013_steady, nasdaq_2016_normal
    train_scores = np.array(
        [
            0.77,  # dotcom_2000 (bubble)
            0.74,  # china_2015 (bubble)
            0.29,  # sp500_2013_steady (control)
            0.00,  # nasdaq_2016_normal (control)
        ]
    )
    train_outcomes = np.array([1, 1, 0, 0])

    calibrator = MetaCalibrator()
    calibrator.fit(train_scores, train_outcomes)
    return calibrator
