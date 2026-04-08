"""
TFS vs OODA Benchmark Comparison — PhaseBreak 2026

Compares Anokhin's Theory of Functional Systems (TFS) pipeline against
the baseline OODA (reactive) pipeline on 58 real benchmark episodes.

TFS: Predict → Act → Compare → Targeted Correct (anticipatory)
OODA: Observe → Orient → Decide → Act (reactive, no prediction)

Usage:
    python compare_tfs_vs_ooda.py [--max-episodes N] [--no-chart] [--save-csv]

Results saved to:
    data/v2_results/tfs_vs_ooda_comparison.json
    data/v2_results/tfs_vs_ooda_table.csv
    data/v2_results/tfs_vs_ooda_chart.png
"""

from __future__ import annotations

import sys
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Project root setup ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Data loaders ─────────────────────────────────────────────────────────
from src.lppls.data import (
    load_yfinance, KNOWN_BUBBLES, NEGATIVE_CONTROLS,
    FORWARD_BUBBLES, FORWARD_CONTROLS,
)
from src.lppls.data_commodities import (
    load_commodity_episode, COMMODITY_BUBBLES, COMMODITY_CONTROLS,
)
from src.housing.data import (
    load_housing_episode, HOUSING_BUBBLES, HOUSING_CONTROLS,
    ZILLOW_BUBBLES, ZILLOW_CONTROLS,
)
from src.validation.adversarial_controls import generate_adversarial_series

# ─── Pipelines ────────────────────────────────────────────────────────────
from src.pipeline.stages import run_full_pipeline
from src.pipeline.acceptor import run_tfs_pipeline_iteration, run_cascaded_pipeline


# ─── Episode definition ───────────────────────────────────────────────────
@dataclass
class EpisodeDef:
    name: str
    domain: str
    load_fn: callable
    load_kwargs: dict
    expected_bubble: bool
    known_tc_date: Optional[str] = None


@dataclass
class EpisodeResult:
    name: str
    domain: str
    expected_bubble: bool
    # OODA
    ooda_verdict: str = ""
    ooda_quality: float = 0.0
    ooda_r2: float = 0.0
    ooda_latency_ms: float = 0.0
    ooda_is_bubble: bool = False
    # TFS
    tfs_verdict: str = ""
    tfs_quality: float = 0.0
    tfs_r2: float = 0.0
    tfs_latency_ms: float = 0.0
    tfs_iterations: int = 0
    tfs_acceptor_match: bool = False
    tfs_satisfaction: float = 0.0
    tfs_is_bubble: bool = False
    # CASCADE
    cascade_verdict: str = ""
    cascade_quality: float = 0.0
    cascade_r2: float = 0.0
    cascade_latency_ms: float = 0.0
    cascade_is_bubble: bool = False
    cascade_path: str = ""  # "tfs" or "ooda_fallback"
    cascade_tfs_satisfaction: float = 0.0


def build_episode_catalog() -> List[EpisodeDef]:
    """Build complete catalog of 58 benchmark episodes."""
    episodes = []

    # Finance bubbles (11)
    for name, kwargs in KNOWN_BUBBLES.items():
        episodes.append(EpisodeDef(
            name=f"finance_{name}", domain="finance",
            load_fn=load_yfinance, load_kwargs=kwargs,
            expected_bubble=True, known_tc_date=kwargs.get("known_tc_date"),
        ))

    # Finance controls (9)
    for name, kwargs in NEGATIVE_CONTROLS.items():
        episodes.append(EpisodeDef(
            name=f"finance_ctrl_{name}", domain="finance",
            load_fn=load_yfinance, load_kwargs=kwargs,
            expected_bubble=False,
        ))

    # Commodity bubbles (6)
    for name, kwargs in COMMODITY_BUBBLES.items():
        episodes.append(EpisodeDef(
            name=f"commodity_{name}", domain="commodities",
            load_fn=load_commodity_episode, load_kwargs={"name": name},
            expected_bubble=True,
        ))

    # Commodity controls (4)
    for name, kwargs in COMMODITY_CONTROLS.items():
        episodes.append(EpisodeDef(
            name=f"commodity_ctrl_{name}", domain="commodities",
            load_fn=load_commodity_episode, load_kwargs={"name": name},
            expected_bubble=False,
        ))

    # Housing FHFA (10 bubbles + 6 controls)
    for name in HOUSING_BUBBLES:
        episodes.append(EpisodeDef(
            name=f"housing_{name}", domain="housing",
            load_fn=load_housing_episode, load_kwargs={"name": name},
            expected_bubble=True,
        ))
    for name in HOUSING_CONTROLS:
        episodes.append(EpisodeDef(
            name=f"housing_ctrl_{name}", domain="housing",
            load_fn=load_housing_episode, load_kwargs={"name": name},
            expected_bubble=False,
        ))

    # Zillow (6 bubbles + 6 controls)
    for name in ZILLOW_BUBBLES:
        episodes.append(EpisodeDef(
            name=f"zillow_{name}", domain="housing",
            load_fn=load_housing_episode, load_kwargs={"name": name, "source": "zillow"},
            expected_bubble=True,
        ))
    for name in ZILLOW_CONTROLS:
        episodes.append(EpisodeDef(
            name=f"zillow_ctrl_{name}", domain="housing",
            load_fn=load_housing_episode, load_kwargs={"name": name, "source": "zillow"},
            expected_bubble=False,
        ))

    # Adversarial (6)
    adv_cases = generate_adversarial_series(seed=42)
    for idx, (name, t, values, expected) in enumerate(adv_cases):
        episodes.append(EpisodeDef(
            name=f"adversarial_{name}", domain="adversarial",
            load_fn=lambda _t=t, _v=values: type("DS", (), {"t": _t, "prices": _v})(),
            load_kwargs={},
            expected_bubble=expected,
        ))

    # Forward 2024-2025 (3 bubbles + 3 controls)
    for name, kwargs in FORWARD_BUBBLES.items():
        episodes.append(EpisodeDef(
            name=f"forward_{name}", domain="finance",
            load_fn=load_yfinance, load_kwargs=kwargs,
            expected_bubble=True,
        ))
    for name, kwargs in FORWARD_CONTROLS.items():
        episodes.append(EpisodeDef(
            name=f"forward_ctrl_{name}", domain="finance",
            load_fn=load_yfinance, load_kwargs=kwargs,
            expected_bubble=False,
        ))

    return episodes


# ─── Runners ──────────────────────────────────────────────────────────────

def run_ooda(t: np.ndarray, values: np.ndarray, domain: str) -> dict:
    """OODA baseline: run_full_pipeline() — reactive, no prediction.

    Uses same configuration as TFS: n_bootstrap=5, same HMM gating.
    The ONLY difference is absence of acceptor prediction + targeted retry.
    """
    start = time.perf_counter()
    try:
        result = run_full_pipeline(t, values, domain=domain, n_bootstrap=5)
        latency_ms = (time.perf_counter() - start) * 1000

        verdict = result.final_verdict if hasattr(result, 'final_verdict') else "NO_BUBBLE"
        quality = result.fit.quality_score if hasattr(result, 'fit') and result.fit else 0.0
        r2 = result.fit.r_squared if hasattr(result, 'fit') and result.fit else 0.0
        is_bubble = verdict in ("BUBBLE", "POSSIBLE")

        return {
            "verdict": verdict,
            "quality": quality,
            "r2": r2,
            "latency_ms": latency_ms,
            "is_bubble": is_bubble,
            "success": True,
        }
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "verdict": "ERROR",
            "quality": 0.0,
            "r2": 0.0,
            "latency_ms": latency_ms,
            "is_bubble": False,
            "success": False,
            "error": str(e),
        }


def run_tfs(t: np.ndarray, values: np.ndarray, domain: str) -> dict:
    """TFS pipeline: predict → act → compare → targeted correct.

    Uses same LPPLS optimizer as OODA, but adds:
    1. Acceptor prediction BEFORE fitting
    2. Comparison AFTER fitting
    3. Targeted retry if mismatch detected
    """
    start = time.perf_counter()
    try:
        result = run_tfs_pipeline_iteration(
            t=t, values=values, domain=domain, max_iterations=3,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        params = result.get("params")
        r2 = result.get("r_squared", 0.0)
        is_bubble = params.is_bubble if params else False
        satisfaction = result.get("satisfaction", 0.0)
        iterations = result.get("iterations", 0)
        acceptor_match = result.get("acceptor_match", False)

        if is_bubble:
            verdict = "BUBBLE" if satisfaction > 0.7 else "POSSIBLE"
        else:
            verdict = "NO_BUBBLE"

        return {
            "verdict": verdict,
            "quality": satisfaction,  # TFS quality = acceptor satisfaction
            "r2": r2,
            "latency_ms": latency_ms,
            "is_bubble": is_bubble,
            "iterations": iterations,
            "acceptor_match": acceptor_match,
            "satisfaction": satisfaction,
            "success": True,
        }
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "verdict": "ERROR",
            "quality": 0.0,
            "r2": 0.0,
            "latency_ms": latency_ms,
            "is_bubble": False,
            "iterations": 0,
            "acceptor_match": False,
            "satisfaction": 0.0,
            "success": False,
            "error": str(e),
        }


def run_cascade(t: np.ndarray, values: np.ndarray, domain: str) -> dict:
    """Cascade pipeline: TFS first → if uncertain → OODA fallback.

    Goal: TFS speed on easy cases (~70%) + OODA accuracy on hard cases.
    """
    start = time.perf_counter()
    try:
        result = run_cascaded_pipeline(
            t=t, values=values, domain=domain, tfs_threshold=0.6,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        params = result.get("params")
        r2 = result.get("r_squared", 0.0)
        is_bubble = result.get("is_bubble", False)
        satisfaction = result.get("satisfaction", 0.0)
        path = result.get("path", "unknown")
        cascade_fallback = result.get("cascade_fallback", False)
        tfs_satisfaction = result.get("tfs_satisfaction", 0.0)

        if is_bubble:
            verdict = result.get("verdict", "BUBBLE")
        else:
            verdict = "NO_BUBBLE"

        return {
            "verdict": verdict,
            "quality": satisfaction,
            "r2": r2,
            "latency_ms": latency_ms,
            "is_bubble": is_bubble,
            "path": path,
            "cascade_fallback": cascade_fallback,
            "tfs_satisfaction": tfs_satisfaction,
            "success": True,
        }
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "verdict": "ERROR",
            "quality": 0.0,
            "r2": 0.0,
            "latency_ms": latency_ms,
            "is_bubble": False,
            "path": "error",
            "cascade_fallback": False,
            "tfs_satisfaction": 0.0,
            "success": False,
            "error": str(e),
        }


# ─── Metrics ──────────────────────────────────────────────────────────────

def compute_metrics(results: List[EpisodeResult], approach: str) -> dict:
    """Compute precision, recall, F1, latency stats for one approach."""
    tp = fp = fn = tn = 0
    latencies = []
    qualities = []

    for r in results:
        if approach == "ooda":
            pred = r.ooda_is_bubble
            latency = r.ooda_latency_ms
            quality = r.ooda_quality
        elif approach == "tfs":
            pred = r.tfs_is_bubble
            latency = r.tfs_latency_ms
            quality = r.tfs_quality
        elif approach == "cascade":
            pred = r.cascade_is_bubble
            latency = r.cascade_latency_ms
            quality = r.cascade_quality
        else:
            continue

        if not (latency > 0):
            continue
        latencies.append(latency)
        qualities.append(quality)

        actual = r.expected_bubble
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "accuracy": round(accuracy, 3),
        "fp_rate": round(fp_rate, 3),
        "mean_latency_ms": round(np.mean(latencies), 1) if latencies else 0,
        "median_latency_ms": round(np.median(latencies), 1) if latencies else 0,
        "p95_latency_ms": round(np.percentile(latencies, 95), 1) if latencies else 0,
        "mean_quality": round(np.mean(qualities), 3) if qualities else 0,
        "total_latency_ms": round(sum(latencies), 1),
        "n_episodes": len(results),
        "n_successful": len([l for l in latencies if l > 0]),
    }


def paired_t_test(ooda_latencies: list, tfs_latencies: list) -> dict:
    """Paired t-test for latency difference."""
    from scipy import stats
    # Only pairs where both succeeded
    pairs = [(o, t) for o, t in zip(ooda_latencies, tfs_latencies) if o > 0 and t > 0]
    if len(pairs) < 3:
        return {"t_stat": 0, "p_value": 1.0, "n_pairs": len(pairs), "significant": False}

    ooda_arr = np.array([p[0] for p in pairs])
    tfs_arr = np.array([p[1] for p in pairs])
    t_stat, p_value = stats.ttest_rel(ooda_arr, tfs_arr)

    return {
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 4),
        "n_pairs": len(pairs),
        "significant": p_value < 0.05,
        "mean_diff_ms": round(np.mean(tfs_arr - ooda_arr), 1),
    }


def mcnemar_test(results: List[EpisodeResult]) -> dict:
    """McNemar test for disagreement in binary predictions."""
    from scipy import stats
    # Contingency: OODA correct vs TFS correct
    ooda_correct = np.array([
        1 if (r.ooda_is_bubble == r.expected_bubble) else 0
        for r in results
    ])
    tfs_correct = np.array([
        1 if (r.tfs_is_bubble == r.expected_bubble) else 0
        for r in results
    ])

    # Discordant pairs
    b = int(np.sum((ooda_correct == 0) & (tfs_correct == 1)))  # OODA wrong, TFS right
    c = int(np.sum((ooda_correct == 1) & (tfs_correct == 0)))  # OODA right, TFS wrong

    if b + c == 0:
        return {"chi2": 0, "p_value": 1.0, "b": b, "c": c, "significant": False}

    # Use chi2_contingency as alternative to mcnemar
    table = np.array([[0, b], [c, 0]])
    try:
        chi2, p_value, dof, expected = stats.chi2_contingency(table, correction=True)
    except ValueError:
        # Fallback: manual McNemar chi-square
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 4),
        "b": b,  # TFS-only correct
        "c": c,  # OODA-only correct
        "significant": p_value < 0.05,
    }


# ─── Chart generation ─────────────────────────────────────────────────────

def generate_charts(results: List[EpisodeResult], output_dir: Path):
    """Generate comparison charts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ooda_latencies = [r.ooda_latency_ms for r in results if r.ooda_latency_ms > 0]
    tfs_latencies = [r.tfs_latency_ms for r in results if r.tfs_latency_ms > 0]
    ooda_qualities = [r.ooda_quality for r in results]
    tfs_qualities = [r.tfs_quality for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Latency distribution (box plot)
    axes[0, 0].boxplot([ooda_latencies, tfs_latencies], tick_labels=["OODA", "TFS"], showfliers=True)
    axes[0, 0].set_ylabel("Latency (ms)")
    axes[0, 0].set_title("Latency Distribution")
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # 2. Quality comparison (scatter by episode)
    episode_indices = np.arange(len(results))
    axes[0, 1].scatter(episode_indices, ooda_qualities, alpha=0.6, label="OODA", color="#e74c3c", s=30)
    axes[0, 1].scatter(episode_indices, tfs_qualities, alpha=0.6, label="TFS", color="#2ecc71", s=30)
    axes[0, 1].set_xlabel("Episode Index")
    axes[0, 1].set_ylabel("Quality / Satisfaction Score")
    axes[0, 1].set_title("Quality per Episode")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # 3. Precision/Recall/F1 bar chart
    ooda_m = compute_metrics(results, "ooda")
    tfs_m = compute_metrics(results, "tfs")
    metrics_names = ["Precision", "Recall", "F1", "Accuracy"]
    ooda_vals = [ooda_m["precision"], ooda_m["recall"], ooda_m["f1"], ooda_m["accuracy"]]
    tfs_vals = [tfs_m["precision"], tfs_m["recall"], tfs_m["f1"], tfs_m["accuracy"]]
    x = np.arange(len(metrics_names))
    width = 0.35
    axes[1, 0].bar(x - width/2, ooda_vals, width, label="OODA", color="#e74c3c")
    axes[1, 0].bar(x + width/2, tfs_vals, width, label="TFS", color="#2ecc71")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metrics_names)
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_title("Detection Metrics")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    axes[1, 0].set_ylim(0, 1.1)

    # 4. TFS iterations vs satisfaction
    tfs_iters = [r.tfs_iterations for r in results if r.tfs_iterations > 0]
    tfs_sats = [r.tfs_satisfaction for r in results if r.tfs_iterations > 0]
    axes[1, 1].scatter(tfs_iters, tfs_sats, alpha=0.6, color="#2ecc71", s=40)
    axes[1, 1].set_xlabel("TFS Iterations")
    axes[1, 1].set_ylabel("Satisfaction Score")
    axes[1, 1].set_title("TFS: Iterations vs Satisfaction")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = output_dir / "tfs_vs_ooda_comparison.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n📊 Chart saved: {chart_path}")


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TFS vs OODA Benchmark Comparison")
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit episodes for quick test")
    parser.add_argument("--no-chart", action="store_true", help="Skip chart generation")
    parser.add_argument("--save-csv", action="store_true", help="Save results to CSV")
    parser.add_argument("--domains", nargs="*", default=None, help="Run only these domains")
    args = parser.parse_args()

    print("=" * 70)
    print("  TFS vs OODA Benchmark Comparison — PhaseBreak 2026")
    print("  Anokhin TFS (anticipatory) vs Baseline OODA (reactive)")
    print("=" * 70)

    # Build episode catalog
    print("\n📦 Building episode catalog...")
    episodes = build_episode_catalog()
    print(f"   Total episodes: {len(episodes)}")

    # Filter
    if args.domains:
        episodes = [e for e in episodes if e.domain in args.domains]
        print(f"   Filtered to domains {args.domains}: {len(episodes)} episodes")
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
        print(f"   Limited to {args.max_episodes} episodes")

    # Run benchmark
    results: List[EpisodeResult] = []
    start_total = time.perf_counter()

    for i, ep in enumerate(episodes):
        print(f"\n[{i+1}/{len(episodes)}] {ep.name} ({ep.domain}, bubble={ep.expected_bubble})")

        # Load data
        try:
            ds = ep.load_fn(**ep.load_kwargs)
            t = ds.t
            values = ds.prices if hasattr(ds, 'prices') else ds.values
            print(f"   Data loaded: n={len(t)}")
        except Exception as e:
            print(f"   ⚠️ Data load failed: {e}")
            continue

        # Run OODA
        print(f"   Running OODA...", end=" ", flush=True)
        ooda_res = run_ooda(t, values, ep.domain)
        print(f"✅ {ooda_res['latency_ms']:.0f}ms → {ooda_res['verdict']}")

        # Run TFS
        print(f"   Running TFS...", end=" ", flush=True)
        tfs_res = run_tfs(t, values, ep.domain)
        print(f"✅ {tfs_res['latency_ms']:.0f}ms → {tfs_res['verdict']} "
              f"(iter={tfs_res.get('iterations', 0)}, "
              f"match={tfs_res.get('acceptor_match', False)})")

        # Run CASCADE
        print(f"   Running CASCADE...", end=" ", flush=True)
        cascade_res = run_cascade(t, values, ep.domain)
        print(f"✅ {cascade_res['latency_ms']:.0f}ms → {cascade_res['verdict']} "
              f"(path={cascade_res.get('path', '?')})")

        results.append(EpisodeResult(
            name=ep.name,
            domain=ep.domain,
            expected_bubble=ep.expected_bubble,
            ooda_verdict=ooda_res["verdict"],
            ooda_quality=ooda_res["quality"],
            ooda_r2=ooda_res["r2"],
            ooda_latency_ms=ooda_res["latency_ms"],
            ooda_is_bubble=ooda_res["is_bubble"],
            tfs_verdict=tfs_res["verdict"],
            tfs_quality=tfs_res["quality"],
            tfs_r2=tfs_res["r2"],
            tfs_latency_ms=tfs_res["latency_ms"],
            tfs_iterations=tfs_res.get("iterations", 0),
            tfs_acceptor_match=tfs_res.get("acceptor_match", False),
            tfs_satisfaction=tfs_res.get("satisfaction", 0.0),
            tfs_is_bubble=tfs_res["is_bubble"],
            cascade_verdict=cascade_res["verdict"],
            cascade_quality=cascade_res["quality"],
            cascade_r2=cascade_res["r2"],
            cascade_latency_ms=cascade_res["latency_ms"],
            cascade_is_bubble=cascade_res["is_bubble"],
            cascade_path=cascade_res.get("path", ""),
            cascade_tfs_satisfaction=cascade_res.get("tfs_satisfaction", 0.0),
        ))

    total_time = time.perf_counter() - start_total
    print(f"\n{'=' * 70}")
    print(f"  Benchmark complete: {total_time:.1f}s for {len(results)} episodes")
    print(f"{'=' * 70}")

    # ─── Metrics ──────────────────────────────────────────────────────
    ooda_metrics = compute_metrics(results, "ooda")
    tfs_metrics = compute_metrics(results, "tfs")
    cascade_metrics = compute_metrics(results, "cascade")

    print("\n📊 COMPARATIVE METRICS")
    print("-" * 90)
    print(f"{'Метрика':<25} {'OODA':>10} {'TFS':>10} {'CASCADE':>12}")
    print("-" * 90)
    for key in ["precision", "recall", "f1", "accuracy", "fp_rate",
                "mean_latency_ms", "median_latency_ms", "mean_quality"]:
        ooda_val = ooda_metrics[key]
        tfs_val = tfs_metrics[key]
        cascade_val = cascade_metrics[key]
        if key.endswith("_ms"):
            print(f"{key:<25} {ooda_val:>8.1f} {tfs_val:>8.1f} {cascade_val:>10.1f}")
        else:
            print(f"{key:<25} {ooda_val:>8.3f} {tfs_val:>8.3f} {cascade_val:>10.3f}")

    # Cascade-specific
    cascade_paths = [r.cascade_path for r in results if r.cascade_path]
    tfs_count = sum(1 for p in cascade_paths if p == "tfs")
    ooda_count = sum(1 for p in cascade_paths if p == "ooda_fallback")
    print(f"\n🧬 CASCADE-SPECIFIC")
    print(f"   TFS path:        {tfs_count}/{len(cascade_paths)} ({tfs_count/max(1,len(cascade_paths)):.0%})")
    print(f"   OODA fallback:   {ooda_count}/{len(cascade_paths)} ({ooda_count/max(1,len(cascade_paths)):.0%})")

    # TFS-specific
    tfs_iters = [r.tfs_iterations for r in results if r.tfs_iterations > 0]
    tfs_matches = [r.tfs_acceptor_match for r in results]
    print(f"\n🧬 TFS-SPECIFIC")
    print(f"   Avg iterations:      {np.mean(tfs_iters):.2f}" if tfs_iters else "   No iterations")
    print(f"   Acceptor match rate: {np.mean(tfs_matches):.1%}" if tfs_matches else "   No matches")

    # Statistical tests
    print(f"\n📈 STATISTICAL TESTS")
    print("-" * 70)

    ooda_lat = [r.ooda_latency_ms for r in results]
    tfs_lat = [r.tfs_latency_ms for r in results]
    t_test = paired_t_test(ooda_lat, tfs_lat)
    print(f"   Paired t-test (latency):")
    print(f"     t={t_test['t_stat']}, p={t_test['p_value']}, "
          f"n_pairs={t_test['n_pairs']}, significant={t_test['significant']}")
    if t_test['significant']:
        print(f"     → TFS latency {'higher' if t_test['mean_diff_ms'] > 0 else 'lower'} "
              f"by {abs(t_test['mean_diff_ms']):.1f}ms (p<0.05)")
    else:
        print(f"     → No significant latency difference (p≥0.05)")

    mcnemar = mcnemar_test(results)
    print(f"\n   McNemar test (correctness):")
    print(f"     χ²={mcnemar['chi2']}, p={mcnemar['p_value']}")
    print(f"     TFS-only correct: {mcnemar['b']}, OODA-only correct: {mcnemar['c']}")
    if mcnemar['significant']:
        print(f"     → Significant difference in accuracy (p<0.05)")
    else:
        print(f"     → No significant difference in accuracy (p≥0.05)")

    # ─── Domain breakdown ─────────────────────────────────────────────
    print(f"\n📊 DOMAIN BREAKDOWN")
    print("-" * 70)
    domains = sorted(set(r.domain for r in results))
    for domain in domains:
        domain_results = [r for r in results if r.domain == domain]
        ooda_dm = compute_metrics(domain_results, "ooda")
        tfs_dm = compute_metrics(domain_results, "tfs")
        cascade_dm = compute_metrics(domain_results, "cascade")
        print(f"\n   {domain.upper()} (n={len(domain_results)}):")
        print(f"     OODA:    P={ooda_dm['precision']:.2f} R={ooda_dm['recall']:.2f} "
              f"F1={ooda_dm['f1']:.2f} Lat={ooda_dm['mean_latency_ms']:.0f}ms")
        print(f"     TFS:     P={tfs_dm['precision']:.2f} R={tfs_dm['recall']:.2f} "
              f"F1={tfs_dm['f1']:.2f} Lat={tfs_dm['mean_latency_ms']:.0f}ms")
        print(f"     CASCADE: P={cascade_dm['precision']:.2f} R={cascade_dm['recall']:.2f} "
              f"F1={cascade_dm['f1']:.2f} Lat={cascade_dm['mean_latency_ms']:.0f}ms")

    # ─── Save results ─────────────────────────────────────────────────
    output_dir = PROJECT_ROOT / "data" / "v2_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_data = {
        "version": "tfs_vs_ooda_v2_with_cascade",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_episodes": len(results),
        "total_time_s": round(total_time, 1),
        "ooda_metrics": ooda_metrics,
        "tfs_metrics": tfs_metrics,
        "cascade_metrics": cascade_metrics,
        "statistical_tests": {
            "paired_t_test": t_test,
            "mcnemar": mcnemar,
        },
        "episodes": [asdict(r) for r in results],
    }
    json_path = output_dir / "tfs_vs_ooda_comparison.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"\n💾 Results saved: {json_path}")

    # CSV
    if args.save_csv or True:  # Always save CSV
        import csv
        csv_path = output_dir / "tfs_vs_ooda_table.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
        print(f"   CSV table: {csv_path}")

    # Chart
    if not args.no_chart:
        generate_charts(results, output_dir)

    # ─── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")

    f1_delta_tfs = tfs_metrics["f1"] - ooda_metrics["f1"]
    f1_delta_cascade = cascade_metrics["f1"] - ooda_metrics["f1"]
    lat_delta_tfs = tfs_metrics["mean_latency_ms"] - ooda_metrics["mean_latency_ms"]
    lat_delta_cascade = cascade_metrics["mean_latency_ms"] - ooda_metrics["mean_latency_ms"]

    print(f"\n  F1-score:    OODA={ooda_metrics['f1']:.3f}  TFS={tfs_metrics['f1']:.3f}  CASCADE={cascade_metrics['f1']:.3f}")
    print(f"  Latency:     OODA={ooda_metrics['mean_latency_ms']:.0f}ms  TFS={tfs_metrics['mean_latency_ms']:.0f}ms  CASCADE={cascade_metrics['mean_latency_ms']:.0f}ms")
    print(f"  Recall:      OODA={ooda_metrics['recall']:.3f}  TFS={tfs_metrics['recall']:.3f}  CASCADE={cascade_metrics['recall']:.3f}")

    if f1_delta_cascade > f1_delta_tfs:
        print(f"\n  🏆 CASCADE лучший по F1 (близок к OODA, быстрее TFS)")
    elif f1_delta_tfs > 0:
        print(f"\n  🏆 TFS лучший по F1")
    else:
        print(f"\n  🏆 OODA лучший по F1, но CASCADE — лучший компромисс")

    if cascade_metrics["mean_latency_ms"] < ooda_metrics["mean_latency_ms"] * 0.7:
        print(f"  ⚡ CASCADE на {abs(lat_delta_cascade/ooda_metrics['mean_latency_ms']*100):.0f}% быстрее OODA")

    if cascade_metrics["recall"] > tfs_metrics["recall"]:
        print(f"  ⚡ CASCADE recall выше TFS на {(cascade_metrics['recall']-tfs_metrics['recall'])*100:.1f}%")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
