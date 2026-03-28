"""LPPLS Multi-Window Confidence Indicator.

Based on Sornette et al. (2015) "Real-time prediction and post-mortem analysis
of the Shanghai 2015 stock market bubble and crash."

Strategy:
- Fit LPPLS on multiple overlapping windows ending at t_end
- If tc converges (±tolerance) across 3+ windows → HIGH confidence
- If tc diverges → NOISE (likely false positive)

This is the DS LPPLS Confidence Indicator — raises precision 30-50%.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import structlog
from numpy.typing import NDArray

from src.lppls.model import LPPLS, LPPLSParams
from src.lppls.optimizer import LPPLSOptimizer

log = structlog.get_logger()

# WHY: Sornette uses windows of 60, 90, 120, 180, 252 trading days
DEFAULT_WINDOWS = [60, 90, 120, 180, 252]
TC_TOLERANCE_DAYS = 15  # tc must agree within ±15 days across windows


@dataclass
class WindowFit:
    """Result of LPPLS fit on a single window."""

    window_size: int
    t_start: int
    t_end: int
    params: LPPLSParams | None
    r_squared: float
    is_valid: bool  # passes Sornette filters


@dataclass
class ConfidenceResult:
    """Multi-window confidence assessment."""

    tc_median: float | None  # median tc across valid windows
    tc_std: float | None  # std dev of tc estimates
    confidence: str  # HIGH / MEDIUM / LOW / NO_SIGNAL
    confidence_score: float  # 0.0 - 1.0
    n_valid_windows: int  # how many windows gave valid fits
    n_total_windows: int  # total windows attempted
    n_agreeing: int  # windows with tc within tolerance
    window_fits: list[WindowFit] = field(default_factory=list)
    m_mean: float | None = None  # mean m across valid windows
    omega_mean: float | None = None  # mean omega across valid windows

    @property
    def is_bubble(self) -> bool:
        return self.confidence in ("HIGH", "MEDIUM") and self.tc_median is not None


class MultiWindowConfidence:
    """DS LPPLS Confidence Indicator via multi-window fitting.

    Fits LPPLS on multiple windows of different lengths, all ending at the
    same t_end. Consensus on tc across windows indicates real signal.
    """

    def __init__(
        self,
        windows: list[int] | None = None,
        tc_tolerance: int = TC_TOLERANCE_DAYS,
        min_r_squared: float = 0.5,
        optimizer_kwargs: dict | None = None,
    ) -> None:
        self.windows = windows or DEFAULT_WINDOWS
        self.tc_tolerance = tc_tolerance
        self.min_r_squared = min_r_squared
        self.optimizer_kwargs = optimizer_kwargs or {}

    def evaluate(self, t: NDArray, log_price: NDArray) -> ConfidenceResult:
        """Run multi-window LPPLS analysis.

        Args:
            t: Full time array
            log_price: Full log-price array

        Returns:
            ConfidenceResult with consensus assessment
        """
        t_end = int(t[-1])
        fits: list[WindowFit] = []

        for w in self.windows:
            if w > len(t):
                continue

            # Window: last w data points
            t_window = t[-w:]
            lp_window = log_price[-w:]

            try:
                optimizer = LPPLSOptimizer(
                    grid_size=8,  # smaller grid for speed (multi-window = many fits)
                    n_best=3,
                    **self.optimizer_kwargs,
                )
                model = optimizer.fit(t_window, lp_window)
                r2 = model.r_squared(t_window, lp_window)

                is_valid = (
                    model.params is not None and model.params.is_bubble and r2 >= self.min_r_squared
                )

                fits.append(
                    WindowFit(
                        window_size=w,
                        t_start=int(t_window[0]),
                        t_end=t_end,
                        params=model.params,
                        r_squared=r2,
                        is_valid=is_valid,
                    )
                )
            except Exception as e:
                log.warning("window_fit_failed", window=w, error=str(e))
                fits.append(
                    WindowFit(
                        window_size=w,
                        t_start=t_end - w,
                        t_end=t_end,
                        params=None,
                        r_squared=0.0,
                        is_valid=False,
                    )
                )

        # Collect tc from valid fits
        valid_fits = [f for f in fits if f.is_valid and f.params is not None]
        n_valid = len(valid_fits)
        n_total = len(fits)

        if n_valid == 0:
            return ConfidenceResult(
                tc_median=None,
                tc_std=None,
                confidence="NO_SIGNAL",
                confidence_score=0.0,
                n_valid_windows=0,
                n_total_windows=n_total,
                n_agreeing=0,
                window_fits=fits,
            )

        tc_values = np.array([f.params.tc for f in valid_fits])
        m_values = np.array([f.params.m for f in valid_fits])
        omega_values = np.array([f.params.omega for f in valid_fits])

        tc_median = float(np.median(tc_values))
        tc_std = float(np.std(tc_values))

        # Count windows agreeing on tc within tolerance
        n_agreeing = int(np.sum(np.abs(tc_values - tc_median) <= self.tc_tolerance))

        # Confidence scoring
        agreement_ratio = n_agreeing / n_total
        valid_ratio = n_valid / n_total

        if agreement_ratio >= 0.6 and n_agreeing >= 3:
            confidence = "HIGH"
            score = min(1.0, 0.7 + 0.3 * agreement_ratio)
        elif agreement_ratio >= 0.4 and n_agreeing >= 2:
            confidence = "MEDIUM"
            score = 0.4 + 0.3 * agreement_ratio
        elif n_valid >= 1:
            confidence = "LOW"
            score = 0.1 + 0.2 * valid_ratio
        else:
            confidence = "NO_SIGNAL"
            score = 0.0

        log.info(
            "multi_window_done",
            n_valid=n_valid,
            n_total=n_total,
            n_agreeing=n_agreeing,
            tc_median=round(tc_median, 1),
            tc_std=round(tc_std, 1),
            confidence=confidence,
            score=round(score, 3),
        )

        return ConfidenceResult(
            tc_median=tc_median,
            tc_std=tc_std,
            confidence=confidence,
            confidence_score=score,
            n_valid_windows=n_valid,
            n_total_windows=n_total,
            n_agreeing=n_agreeing,
            window_fits=fits,
            m_mean=float(np.mean(m_values)),
            omega_mean=float(np.mean(omega_values)),
        )
