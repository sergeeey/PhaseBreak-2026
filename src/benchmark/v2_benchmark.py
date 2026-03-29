"""PhaseBreak v2 Official Benchmark.

Single reproducible script that runs v2 pipeline on all domains
and produces official headline results + metrics.

Usage:
    python -m src.benchmark.v2_benchmark

Outputs:
    data/v2_results/benchmark_results.json
    data/v2_results/ablation.json
    Console summary tables
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import structlog

from src.lppls.data import KNOWN_BUBBLES, NEGATIVE_CONTROLS, load_yfinance
from src.lppls.data_commodities import (
    COMMODITY_BUBBLES,
    COMMODITY_CONTROLS,
    load_commodity_episode,
)
from src.pipeline.stages import run_full_pipeline, run_legacy_pipeline, PipelineResult
from src.validation.adversarial_controls import generate_adversarial_series
from src.validation.metrics_early_warning import compute_ew_metrics, summarize_ew_metrics

log = structlog.get_logger()

RESULTS_DIR = Path("data/v2_results")


@dataclass
class EpisodeResult:
    """Result for a single episode."""

    name: str
    domain: str
    expected_bubble: bool
    v2_verdict: str
    v2_quality_score: float
    v2_is_bubble: bool
    tc_estimate: float | None
    tc_lower: float | None
    tc_upper: float | None
    r_squared: float
    hmm_regime: str | None
    hmm_bubble_prob: float
    correct: bool
    path: str


def _run_episode(
    name: str,
    domain: str,
    t: np.ndarray,
    values: np.ndarray,
    expected_bubble: bool,
    frequency: str = "daily",
) -> EpisodeResult:
    """Run v2 pipeline on one episode and return structured result."""
    result = run_full_pipeline(t, values, frequency=frequency, n_bootstrap=15)

    predicted_bubble = result.final_verdict in ("BUBBLE", "POSSIBLE")
    correct = predicted_bubble == expected_bubble

    tc_est = tc_lo = tc_hi = None
    r2 = 0.0
    quality = 0.0
    if result.fit:
        tc_est = result.fit.tc_estimate
        tc_lo = result.fit.tc_lower
        tc_hi = result.fit.tc_upper
        r2 = result.fit.r_squared
        quality = result.fit.quality_score

    return EpisodeResult(
        name=name,
        domain=domain,
        expected_bubble=expected_bubble,
        v2_verdict=result.final_verdict,
        v2_quality_score=round(quality, 4),
        v2_is_bubble=predicted_bubble,
        tc_estimate=round(tc_est, 1) if tc_est else None,
        tc_lower=round(tc_lo, 1) if tc_lo else None,
        tc_upper=round(tc_hi, 1) if tc_hi else None,
        r_squared=round(r2, 4),
        hmm_regime=result.screening.hmm_regime,
        hmm_bubble_prob=round(result.screening.hmm_bubble_prob, 3),
        correct=correct,
        path=result.path,
    )


def run_finance_benchmark() -> list[EpisodeResult]:
    """Run v2 pipeline on all finance episodes."""
    results = []

    for name, config in KNOWN_BUBBLES.items():
        log.info("benchmark_finance_bubble", episode=name)
        try:
            ds = load_yfinance(**config)
            r = _run_episode(name, "finance", ds.t, ds.prices, expected_bubble=True)
            results.append(r)
        except Exception as e:
            log.warning("episode_failed", episode=name, error=str(e))

    for name, config in NEGATIVE_CONTROLS.items():
        log.info("benchmark_finance_control", episode=name)
        try:
            ds = load_yfinance(**config)
            r = _run_episode(name, "finance", ds.t, ds.prices, expected_bubble=False)
            results.append(r)
        except Exception as e:
            log.warning("episode_failed", episode=name, error=str(e))

    return results


def run_commodity_benchmark() -> list[EpisodeResult]:
    """Run v2 pipeline on all commodity episodes."""
    results = []

    for name in COMMODITY_BUBBLES:
        log.info("benchmark_commodity_bubble", episode=name)
        try:
            ds = load_commodity_episode(name)
            r = _run_episode(name, "commodities", ds.t, ds.prices, expected_bubble=True)
            results.append(r)
        except Exception as e:
            log.warning("episode_failed", episode=name, error=str(e))

    for name in COMMODITY_CONTROLS:
        log.info("benchmark_commodity_control", episode=name)
        try:
            ds = load_commodity_episode(name)
            r = _run_episode(name, "commodities", ds.t, ds.prices, expected_bubble=False)
            results.append(r)
        except Exception as e:
            log.warning("episode_failed", episode=name, error=str(e))

    return results


def run_housing_benchmark() -> list[EpisodeResult]:
    """Run v2 pipeline on all housing episodes (quarterly frequency)."""
    from src.housing.data import HOUSING_BUBBLES, HOUSING_CONTROLS, load_housing_episode

    results = []

    for name in HOUSING_BUBBLES:
        log.info("benchmark_housing_bubble", episode=name)
        try:
            ds = load_housing_episode(name)
            r = _run_episode(
                name, "housing", ds.t, ds.values, expected_bubble=True, frequency="quarterly"
            )
            results.append(r)
        except Exception as e:
            log.warning("episode_failed", episode=name, error=str(e))

    for name in HOUSING_CONTROLS:
        log.info("benchmark_housing_control", episode=name)
        try:
            ds = load_housing_episode(name)
            r = _run_episode(
                name, "housing", ds.t, ds.values, expected_bubble=False, frequency="quarterly"
            )
            results.append(r)
        except Exception as e:
            log.warning("episode_failed", episode=name, error=str(e))

    return results


def run_adversarial_benchmark() -> list[EpisodeResult]:
    """Run v2 pipeline on adversarial synthetic cases."""
    cases = generate_adversarial_series()
    results = []

    for name, t, values, expected in cases:
        log.info("benchmark_adversarial", episode=name)
        r = _run_episode(name, "adversarial", t, values, expected_bubble=expected)
        results.append(r)

    return results


def compute_domain_summary(results: list[EpisodeResult], domain: str) -> dict:
    """Compute TP/FP/FN/TN and derived metrics for a domain."""
    domain_results = [r for r in results if r.domain == domain]
    if not domain_results:
        return {"domain": domain, "n": 0}

    tp = sum(1 for r in domain_results if r.expected_bubble and r.v2_is_bubble)
    fp = sum(1 for r in domain_results if not r.expected_bubble and r.v2_is_bubble)
    fn = sum(1 for r in domain_results if r.expected_bubble and not r.v2_is_bubble)
    tn = sum(1 for r in domain_results if not r.expected_bubble and not r.v2_is_bubble)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # tc interval metrics for detected bubbles
    detected = [r for r in domain_results if r.v2_is_bubble and r.tc_lower is not None]
    mean_interval_width = (
        float(np.mean([r.tc_upper - r.tc_lower for r in detected])) if detected else None
    )

    quality_scores = [r.v2_quality_score for r in domain_results if r.v2_quality_score > 0]
    mean_quality = float(np.mean(quality_scores)) if quality_scores else 0.0

    return {
        "domain": domain,
        "n": len(domain_results),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "fp_rate": round(fp_rate, 3),
        "mean_quality_score": round(mean_quality, 3),
        "mean_tc_interval_width": round(mean_interval_width, 1) if mean_interval_width else None,
        "episodes": [
            {
                "name": r.name,
                "expected": r.expected_bubble,
                "verdict": r.v2_verdict,
                "quality": r.v2_quality_score,
                "correct": r.correct,
                "tc": r.tc_estimate,
                "tc_interval": f"[{r.tc_lower}, {r.tc_upper}]" if r.tc_lower else None,
                "hmm": r.hmm_regime,
            }
            for r in domain_results
        ],
    }


def run_full_benchmark() -> dict:
    """Run complete v2 benchmark across all domains."""
    start = time.time()
    all_results: list[EpisodeResult] = []

    log.info("=== PhaseBreak v2 Official Benchmark ===")

    # Finance
    log.info("--- Finance domain ---")
    all_results.extend(run_finance_benchmark())

    # Commodities
    log.info("--- Commodities domain ---")
    all_results.extend(run_commodity_benchmark())

    # Housing
    log.info("--- Housing domain ---")
    all_results.extend(run_housing_benchmark())

    # Adversarial
    log.info("--- Adversarial controls ---")
    all_results.extend(run_adversarial_benchmark())

    elapsed = time.time() - start

    # Summaries
    summaries = {}
    for domain in ["finance", "commodities", "housing", "adversarial"]:
        summaries[domain] = compute_domain_summary(all_results, domain)

    # Global
    total_correct = sum(1 for r in all_results if r.correct)
    total = len(all_results)

    benchmark = {
        "version": "v2",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "total_episodes": total,
        "total_correct": total_correct,
        "overall_accuracy": round(total_correct / total, 3) if total > 0 else 0,
        "domains": summaries,
    }

    return benchmark


def save_results(benchmark: dict) -> Path:
    """Save benchmark results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "benchmark_results.json"
    with open(path, "w") as f:
        json.dump(benchmark, f, indent=2, default=str)
    log.info("results_saved", path=str(path))
    return path


def print_summary(benchmark: dict) -> None:
    """Print human-readable summary to console."""
    print("\n" + "=" * 70)
    print("PhaseBreak v2 Official Benchmark Results")
    print("=" * 70)
    print(f"Time: {benchmark['timestamp']} ({benchmark['elapsed_seconds']}s)")
    print(f"Episodes: {benchmark['total_episodes']}")
    print(f"Accuracy: {benchmark['overall_accuracy']:.1%}")
    print()

    for domain, summary in benchmark["domains"].items():
        if summary["n"] == 0:
            continue
        print(f"--- {domain.upper()} ---")
        print(f"  TP={summary['tp']} FP={summary['fp']} FN={summary['fn']} TN={summary['tn']}")
        print(
            f"  Precision={summary['precision']:.1%} "
            f"Recall={summary['recall']:.1%} "
            f"FP_rate={summary['fp_rate']:.1%}"
        )
        if summary.get("mean_quality_score"):
            print(f"  Mean quality={summary['mean_quality_score']:.3f}")
        if summary.get("mean_tc_interval_width"):
            print(f"  Mean tc interval width={summary['mean_tc_interval_width']:.1f}")
        print()

        for ep in summary.get("episodes", []):
            mark = "OK" if ep["correct"] else "MISS"
            print(
                f"    [{mark}] {ep['name']}: {ep['verdict']} "
                f"(q={ep['quality']:.2f}, hmm={ep['hmm']})"
            )
        print()


if __name__ == "__main__":
    benchmark = run_full_benchmark()
    save_results(benchmark)
    print_summary(benchmark)
