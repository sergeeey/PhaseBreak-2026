"""Cross-validate soft scoring weights on train split, evaluate on val split.

Replaces hand-tuned weights with data-driven optimal weights per domain.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import structlog

from src.lppls.data import KNOWN_BUBBLES, NEGATIVE_CONTROLS, load_yfinance
from src.lppls.optimizer import LPPLSOptimizer
from src.lppls.scoring import compute_quality_score, DOMAIN_WEIGHTS
from src.validation.splits import get_split

log = structlog.get_logger()


def _load_episodes(names: list[str], domain: str = "finance"):
    """Load episodes by name from known bubbles + controls."""
    all_eps = {**KNOWN_BUBBLES, **NEGATIVE_CONTROLS}
    episodes = []
    for name in names:
        if name not in all_eps:
            continue
        try:
            ds = load_yfinance(**all_eps[name])
            is_bubble = name in KNOWN_BUBBLES
            episodes.append((name, ds.t, ds.log_price, is_bubble))
        except Exception:
            pass
    return episodes


def _evaluate_weights(episodes: list, weights: dict[str, float], r2_threshold: float = 0.5) -> dict:
    """Run LPPLS + scoring with given weights, return metrics."""
    tp = fp = fn = tn = 0

    for name, t, lp, expected in episodes:
        opt = LPPLSOptimizer(grid_size=12, n_best=5)
        model = opt.fit(t, lp)
        r2 = model.r_squared(t, lp)
        p = model.params

        quality = compute_quality_score(p, r2, weights=weights) if p else 0.0
        predicted = p is not None and p.is_bubble and r2 > r2_threshold and quality > 0.3

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

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def grid_search_weights(domain: str = "finance") -> dict:
    """Grid search over weight combinations on train split."""
    split = get_split(domain)
    train_episodes = _load_episodes(split.train, domain)
    val_episodes = _load_episodes(split.validation, domain)

    if not train_episodes:
        log.warning("no_train_episodes", domain=domain)
        return DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS["finance"])

    # Weight grid: each component gets weight from {0.05, 0.15, 0.25, 0.35}
    # Constraint: sum = 1.0
    candidates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    components = ["m", "B", "omega", "damping", "r2"]

    best_f1 = -1
    best_weights = None
    best_metrics = None
    n_evaluated = 0

    # WHY: exhaustive grid is too large (8^5 = 32768).
    # Use coarse grid: pick 3 values per component, filter to sum≈1.0
    coarse = [0.05, 0.15, 0.25, 0.35]
    for combo in itertools.product(coarse, repeat=5):
        total = sum(combo)
        if abs(total - 1.0) > 0.15:  # allow ±0.15 tolerance
            continue

        # Normalize to sum=1
        norm = {c: v / total for c, v in zip(components, combo)}
        metrics = _evaluate_weights(train_episodes, norm)
        n_evaluated += 1

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_weights = norm
            best_metrics = metrics

    log.info(
        "weight_cv_done",
        domain=domain,
        n_evaluated=n_evaluated,
        best_f1=round(best_f1, 3),
        best_weights={k: round(v, 3) for k, v in best_weights.items()} if best_weights else None,
    )

    # Validate on val split
    if val_episodes and best_weights:
        val_metrics = _evaluate_weights(val_episodes, best_weights)
        log.info(
            "val_metrics",
            domain=domain,
            **{k: round(v, 3) if isinstance(v, float) else v for k, v in val_metrics.items()},
        )
    else:
        val_metrics = None

    return {
        "domain": domain,
        "best_weights": {k: round(v, 4) for k, v in best_weights.items()} if best_weights else None,
        "train_metrics": best_metrics,
        "val_metrics": val_metrics,
        "n_evaluated": n_evaluated,
        "current_weights": DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS["finance"]),
    }


if __name__ == "__main__":
    result = grid_search_weights("finance")
    print(f"\nFinance weights CV:")
    print(f"  Current: {result['current_weights']}")
    print(f"  Optimal: {result['best_weights']}")
    print(f"  Train:   {result['train_metrics']}")
    print(f"  Val:     {result['val_metrics']}")
