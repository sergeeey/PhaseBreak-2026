"""Rolling Window Scanner for LPPLS analysis.

Scans a time series with a sliding window, running multi-window
confidence analysis at each step. Produces a time series of
bubble confidence scores — the DS LPPLS Confidence Indicator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray

from src.lppls.confidence import ConfidenceResult, MultiWindowConfidence

log = structlog.get_logger()


@dataclass
class ScanPoint:
    """Single scan result at a specific time point."""

    t_end: int
    confidence: ConfidenceResult


class LPPLSScanner:
    """Scan time series with rolling multi-window LPPLS analysis.

    Produces a confidence time series showing when bubble signals emerge.
    """

    def __init__(
        self,
        step: int = 5,
        min_data_points: int = 120,
        confidence_kwargs: dict | None = None,
    ) -> None:
        """
        Args:
            step: advance window by N points between scans
            min_data_points: minimum data length before starting scan
            confidence_kwargs: passed to MultiWindowConfidence
        """
        self.step = step
        self.min_data_points = min_data_points
        self.mwc = MultiWindowConfidence(**(confidence_kwargs or {}))

    def scan(self, t: NDArray, log_price: NDArray) -> list[ScanPoint]:
        """Run rolling scan over the time series.

        Returns list of ScanPoints, one per step, each with full
        multi-window confidence assessment.
        """
        n = len(t)
        results: list[ScanPoint] = []

        for end_idx in range(self.min_data_points, n, self.step):
            t_slice = t[:end_idx]
            lp_slice = log_price[:end_idx]

            try:
                confidence = self.mwc.evaluate(t_slice, lp_slice)
                results.append(ScanPoint(t_end=int(t[end_idx - 1]), confidence=confidence))
            except Exception as e:
                log.warning("scan_point_failed", t_end=int(t[end_idx - 1]), error=str(e))
                continue

        log.info("scan_complete", n_points=len(results), n_total=n)
        return results

    @staticmethod
    def extract_confidence_series(scan_results: list[ScanPoint]) -> tuple[NDArray, NDArray]:
        """Extract (time, confidence_score) arrays from scan results.

        Useful for plotting the DS LPPLS Confidence Indicator over time.
        """
        if not scan_results:
            return np.array([]), np.array([])
        times = np.array([sp.t_end for sp in scan_results])
        scores = np.array([sp.confidence.confidence_score for sp in scan_results])
        return times, scores
