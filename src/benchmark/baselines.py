"""Baseline comparison: simple methods vs LPPLS v2 pipeline.

Baselines:
1. CAGR threshold: if annualized growth > X% → bubble
2. Z-score: if recent return z-score > 2 → bubble
3. Trend deviation: if price > 2σ above linear trend → bubble
4. Volatility spike: if recent vol > 2× historical → bubble

All baselines run on the same 44 episodes as v2 benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import structlog

from src.lppls.data import (
    KNOWN_BUBBLES,
    NEGATIVE_CONTROLS,
    FORWARD_BUBBLES,
    FORWARD_CONTROLS,
    load_yfinance,
)
from src.lppls.data_commodities import COMMODITY_BUBBLES, COMMODITY_CONTROLS, load_commodity_episode
from src.validation.adversarial_controls import generate_adversarial_series

log = structlog.get_logger()


def baseline_cagr(values: np.ndarray, threshold: float = 0.5) -> bool:
    """CAGR > threshold annualized → bubble."""
    if len(values) < 20:
        return False
    total_return = values[-1] / values[0]
    n_years = len(values) / 252  # approximate trading days
    if n_years < 0.1:
        return False
    cagr = total_return ** (1 / n_years) - 1
    return cagr > threshold


def baseline_zscore(values: np.ndarray, threshold: float = 2.0) -> bool:
    """Recent return z-score > threshold → bubble."""
    if len(values) < 60:
        return False
    log_returns = np.diff(np.log(np.clip(values, 1e-10, None)))
    recent = log_returns[-20:]
    historical = log_returns[:-20]
    if np.std(historical) < 1e-10:
        return False
    z = (np.mean(recent) - np.mean(historical)) / np.std(historical)
    return z > threshold


def baseline_trend_deviation(values: np.ndarray, sigma_threshold: float = 2.0) -> bool:
    """Price > sigma_threshold × σ above linear trend → bubble."""
    if len(values) < 30:
        return False
    t = np.arange(len(values))
    # Linear regression
    coeffs = np.polyfit(t, np.log(np.clip(values, 1e-10, None)), 1)
    trend = np.polyval(coeffs, t)
    residuals = np.log(np.clip(values, 1e-10, None)) - trend
    std_res = np.std(residuals)
    if std_res < 1e-10:
        return False
    # Is the last point > 2σ above trend?
    return residuals[-1] > sigma_threshold * std_res


def baseline_vol_spike(values: np.ndarray, ratio_threshold: float = 2.0) -> bool:
    """Recent volatility > ratio × historical → bubble."""
    if len(values) < 60:
        return False
    log_returns = np.diff(np.log(np.clip(values, 1e-10, None)))
    recent_vol = np.std(log_returns[-20:])
    hist_vol = np.std(log_returns[:-20])
    if hist_vol < 1e-10:
        return False
    return recent_vol / hist_vol > ratio_threshold


def baseline_bocpd(values: np.ndarray) -> bool:
    """Bayesian Online Changepoint Detection — detects regime shifts.

    WHY as baseline: BOCPD is a recognized method (Adams & MacKay 2007).
    Key comparison: BOCPD detects changepoints POST-HOC (at/after event).
    LPPLS predicts tc BEFORE the crash. LPPLS wins on timing.
    """
    if len(values) < 60:
        return False
    try:
        from bayesian_changepoint_detection.online_changepoint_detection import (
            online_changepoint_detection,
            constant_hazard,
            StudentT,
        )
        from functools import partial

        log_ret = np.diff(np.log(np.clip(values, 1e-10, None)))
        R, maxes = online_changepoint_detection(
            log_ret,
            partial(constant_hazard, 250),
            StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0),
        )
        # Changepoint in last 20% of data = "something happened recently"
        n = len(log_ret)
        last_20pct = maxes[-n // 5 :]
        cp_detected = np.any(np.diff(last_20pct) < -10)
        return bool(cp_detected)
    except Exception:
        return False


BASELINES = {
    "CAGR>50%": baseline_cagr,
    "Z-score>2": baseline_zscore,
    "Trend>2σ": baseline_trend_deviation,
    "Vol spike>2×": baseline_vol_spike,
    "BOCPD": baseline_bocpd,
}


def _load_all_episodes() -> list[tuple[str, str, np.ndarray, bool]]:
    """Load all benchmark episodes as (name, domain, values, expected)."""
    episodes = []

    # Finance
    for name, config in KNOWN_BUBBLES.items():
        try:
            ds = load_yfinance(**config)
            episodes.append((name, "finance", ds.prices, True))
        except Exception:
            pass
    for name, config in NEGATIVE_CONTROLS.items():
        try:
            ds = load_yfinance(**config)
            episodes.append((name, "finance", ds.prices, False))
        except Exception:
            pass

    # Commodities
    for name in COMMODITY_BUBBLES:
        try:
            ds = load_commodity_episode(name)
            episodes.append((name, "commodities", ds.prices, True))
        except Exception:
            pass
    for name in COMMODITY_CONTROLS:
        try:
            ds = load_commodity_episode(name)
            episodes.append((name, "commodities", ds.prices, False))
        except Exception:
            pass

    # Adversarial
    for name, t, values, expected in generate_adversarial_series():
        episodes.append((name, "adversarial", values, expected))

    # Forward
    for name, config in FORWARD_BUBBLES.items():
        try:
            ds = load_yfinance(**config)
            episodes.append((name, "forward", ds.prices, True))
        except Exception:
            pass
    for name, config in FORWARD_CONTROLS.items():
        try:
            ds = load_yfinance(**config)
            episodes.append((name, "forward", ds.prices, False))
        except Exception:
            pass

    return episodes


def run_baselines() -> dict:
    """Run all baselines on all episodes."""
    episodes = _load_all_episodes()
    log.info(f"Loaded {len(episodes)} episodes for baseline comparison")

    results = {}
    for bl_name, bl_fn in BASELINES.items():
        tp = fp = fn = tn = 0
        for name, domain, values, expected in episodes:
            predicted = bl_fn(values)
            if expected and predicted:
                tp += 1
            elif not expected and predicted:
                fp += 1
            elif expected and not predicted:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[bl_name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    return results


def print_baselines(results: dict, v2_metrics: dict | None = None) -> None:
    """Print baseline comparison table."""
    print("\n" + "=" * 75)
    print("Baseline Comparison (all episodes)")
    print("=" * 75)
    print(
        f"{'Method':<20} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3} {'Prec':>6} {'Rec':>6} {'F1':>5}"
    )
    print("-" * 75)

    for name, m in results.items():
        print(
            f"{name:<20} {m['tp']:>3} {m['fp']:>3} {m['fn']:>3} {m['tn']:>3} "
            f"{m['precision']:>5.1%} {m['recall']:>5.1%} {m['f1']:>4.1%}"
        )

    if v2_metrics:
        print("-" * 75)
        m = v2_metrics
        print(
            f"{'LPPLS v2 pipeline':<20} {m['tp']:>3} {m['fp']:>3} {m['fn']:>3} {m['tn']:>3} "
            f"{m['precision']:>5.1%} {m['recall']:>5.1%} {m['f1']:>4.1%}"
        )
    print()


if __name__ == "__main__":
    results = run_baselines()

    # Load v2 results for comparison
    v2_path = Path("data/v2_results/benchmark_results.json")
    v2_metrics = None
    if v2_path.exists():
        with open(v2_path) as f:
            v2 = json.load(f)
        tp = sum(d.get("tp", 0) for d in v2["domains"].values() if d.get("n", 0) > 0)
        fp = sum(d.get("fp", 0) for d in v2["domains"].values() if d.get("n", 0) > 0)
        fn = sum(d.get("fn", 0) for d in v2["domains"].values() if d.get("n", 0) > 0)
        tn = sum(d.get("tn", 0) for d in v2["domains"].values() if d.get("n", 0) > 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        v2_metrics = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    print_baselines(results, v2_metrics)

    # Save
    Path("data/v2_results").mkdir(parents=True, exist_ok=True)
    with open("data/v2_results/baselines.json", "w") as f:
        json.dump(results, f, indent=2)
