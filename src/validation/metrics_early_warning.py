"""Richer early-warning metrics beyond TP/FP/Precision/Recall.

Computes:
- lead_time: how far ahead tc was predicted before actual peak
- warning_stability: did the signal persist or flicker
- tc_interval_width: uncertainty band size
- tc_interval_coverage: does actual peak fall within predicted interval
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog

log = structlog.get_logger()


@dataclass
class EarlyWarningMetrics:
    """Rich metrics for a single prediction episode."""

    name: str
    lead_time: float | None  # how many periods ahead tc was predicted
    tc_error: float | None  # |predicted - actual|
    tc_interval_width: float | None  # upper - lower
    tc_interval_coverage: bool | None  # actual peak within interval?
    quality_score: float  # soft filter score
    verdict: str


def compute_ew_metrics(
    name: str,
    tc_estimate: float | None,
    tc_lower: float | None,
    tc_upper: float | None,
    actual_peak_idx: float | None,
    last_data_idx: float,
    quality_score: float,
    verdict: str,
) -> EarlyWarningMetrics:
    """Compute early warning metrics for one episode."""
    lead_time = None
    tc_error = None
    interval_width = None
    coverage = None

    if tc_estimate is not None and actual_peak_idx is not None:
        lead_time = tc_estimate - last_data_idx
        tc_error = abs(tc_estimate - actual_peak_idx)

    if tc_lower is not None and tc_upper is not None:
        interval_width = tc_upper - tc_lower
        if actual_peak_idx is not None:
            coverage = tc_lower <= actual_peak_idx <= tc_upper

    return EarlyWarningMetrics(
        name=name,
        lead_time=lead_time,
        tc_error=tc_error,
        tc_interval_width=interval_width,
        tc_interval_coverage=coverage,
        quality_score=quality_score,
        verdict=verdict,
    )


def summarize_ew_metrics(metrics: list[EarlyWarningMetrics]) -> dict:
    """Aggregate early warning metrics across episodes."""
    if not metrics:
        return {}

    lead_times = [m.lead_time for m in metrics if m.lead_time is not None]
    tc_errors = [m.tc_error for m in metrics if m.tc_error is not None]
    widths = [m.tc_interval_width for m in metrics if m.tc_interval_width is not None]
    coverages = [m.tc_interval_coverage for m in metrics if m.tc_interval_coverage is not None]

    return {
        "n_episodes": len(metrics),
        "mean_lead_time": float(np.mean(lead_times)) if lead_times else None,
        "median_tc_error": float(np.median(tc_errors)) if tc_errors else None,
        "mean_interval_width": float(np.mean(widths)) if widths else None,
        "interval_coverage": float(np.mean(coverages)) if coverages else None,
        "mean_quality_score": float(np.mean([m.quality_score for m in metrics])),
    }
