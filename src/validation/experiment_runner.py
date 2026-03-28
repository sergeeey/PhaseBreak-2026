"""Batch experiment: 100 LPPLS predictions with/without adversarial council.

Generates 50 real bubble signals + 50 noise signals.
Compares detection accuracy with and without council validation.
If council improves precision by >15% → publishable contribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray

from src.lppls.model import LPPLS, LPPLSParams
from src.lppls.optimizer import LPPLSOptimizer
from src.lppls.metrics import durbin_watson
from src.validation.council_validator import run_council, CouncilVerdict

log = structlog.get_logger()


@dataclass
class ExperimentResult:
    """Result of single prediction with/without council."""

    is_real_bubble: bool
    lppls_verdict: str  # BUBBLE / NO_BUBBLE from optimizer
    council_verdict: str  # BUBBLE / NO_BUBBLE / UNCERTAIN from council
    council_confidence: float
    r_squared: float
    params: LPPLSParams | None


@dataclass
class ExperimentSummary:
    """Summary of batch experiment."""

    n_total: int
    n_real: int
    n_noise: int

    # Without council (raw LPPLS)
    lppls_tp: int
    lppls_fp: int
    lppls_fn: int
    lppls_tn: int
    lppls_precision: float
    lppls_recall: float

    # With council
    council_tp: int
    council_fp: int
    council_fn: int
    council_tn: int
    council_precision: float
    council_recall: float

    precision_improvement: float
    precision_improvement_pct: float


def generate_synthetic_signals(
    n_real: int = 50,
    n_noise: int = 50,
    seed: int = 42,
) -> list[tuple[NDArray, NDArray, bool]]:
    """Generate synthetic LPPLS signals (real bubbles + noise).

    Returns list of (t, log_price, is_real_bubble) tuples.
    """
    rng = np.random.default_rng(seed)
    signals = []

    # Real bubbles with varying parameters
    for i in range(n_real):
        tc = rng.uniform(200, 400)
        m = rng.uniform(0.2, 0.8)
        omega = rng.uniform(6.5, 12.5)
        A = rng.uniform(8, 12)
        B = rng.uniform(-0.8, -0.05)
        C1 = rng.uniform(0.005, 0.05)
        C2 = rng.uniform(0.002, 0.03)

        n_points = int(tc * rng.uniform(0.6, 0.9))
        t = np.arange(n_points, dtype=float)
        log_price = LPPLS.func(t, tc, m, omega, A, B, C1, C2)
        noise = rng.normal(0, 0.01 * rng.uniform(0.5, 2.0), size=log_price.shape)
        log_price += noise

        signals.append((t, log_price, True))

    # Noise: random walk, linear trend, sine wave
    for i in range(n_noise):
        n_points = rng.integers(100, 300)
        t = np.arange(n_points, dtype=float)

        pattern = i % 3
        if pattern == 0:  # random walk
            log_price = 10 + np.cumsum(rng.normal(0, 0.02, size=n_points))
        elif pattern == 1:  # linear trend + noise
            log_price = 10 + 0.005 * t + rng.normal(0, 0.03, size=n_points)
        else:  # sine wave (periodic but not log-periodic)
            log_price = 10 + 0.3 * np.sin(2 * np.pi * t / 50) + rng.normal(0, 0.02, size=n_points)

        signals.append((t, log_price, False))

    rng.shuffle(signals)
    return signals


def run_experiment(
    n_real: int = 50,
    n_noise: int = 50,
    use_ollama: bool = True,
    ollama_model: str = "qwen2.5:14b",
    seed: int = 42,
) -> ExperimentSummary:
    """Run full 100-prediction experiment.

    Args:
        n_real: Number of real bubble signals
        n_noise: Number of noise signals
        use_ollama: Whether to use Ollama (False = heuristic council)
        ollama_model: Ollama model for council
        seed: Random seed

    Returns:
        ExperimentSummary with precision/recall comparison
    """
    signals = generate_synthetic_signals(n_real, n_noise, seed)
    results: list[ExperimentResult] = []

    for idx, (t, log_price, is_real) in enumerate(signals):
        # Fit LPPLS
        optimizer = LPPLSOptimizer(grid_size=10, n_best=3)
        try:
            model = optimizer.fit(t, log_price)
            r2 = model.r_squared(t, log_price)
            params = model.params
            lppls_verdict = "BUBBLE" if (params and params.is_bubble and r2 > 0.5) else "NO_BUBBLE"
            residuals = model.residuals(t, log_price) if params else np.array([])
            dw = durbin_watson(residuals) if len(residuals) > 0 else 2.0
        except Exception:
            params = None
            r2 = 0.0
            dw = 2.0
            lppls_verdict = "NO_BUBBLE"

        # Council validation
        if use_ollama:
            council = run_council(params, r2, dw, len(t), model=ollama_model)
        else:
            council = run_council(params, r2, dw, len(t), model="__heuristic__")
            # Force heuristic path by using invalid model name

        council_verdict = council.final_verdict
        if council_verdict == "UNCERTAIN":
            council_verdict = "NO_BUBBLE"  # conservative: uncertain = no bubble

        results.append(
            ExperimentResult(
                is_real_bubble=is_real,
                lppls_verdict=lppls_verdict,
                council_verdict=council_verdict,
                council_confidence=council.final_confidence,
                r_squared=r2,
                params=params,
            )
        )

        if (idx + 1) % 20 == 0:
            log.info("experiment_progress", done=idx + 1, total=len(signals))

    # Compute metrics
    lppls_tp = sum(1 for r in results if r.is_real_bubble and r.lppls_verdict == "BUBBLE")
    lppls_fp = sum(1 for r in results if not r.is_real_bubble and r.lppls_verdict == "BUBBLE")
    lppls_fn = sum(1 for r in results if r.is_real_bubble and r.lppls_verdict != "BUBBLE")
    lppls_tn = sum(1 for r in results if not r.is_real_bubble and r.lppls_verdict != "BUBBLE")

    council_tp = sum(1 for r in results if r.is_real_bubble and r.council_verdict == "BUBBLE")
    council_fp = sum(1 for r in results if not r.is_real_bubble and r.council_verdict == "BUBBLE")
    council_fn = sum(1 for r in results if r.is_real_bubble and r.council_verdict != "BUBBLE")
    council_tn = sum(1 for r in results if not r.is_real_bubble and r.council_verdict != "BUBBLE")

    def _precision(tp: int, fp: int) -> float:
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def _recall(tp: int, fn: int) -> float:
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    lppls_prec = _precision(lppls_tp, lppls_fp)
    council_prec = _precision(council_tp, council_fp)
    improvement = council_prec - lppls_prec
    improvement_pct = (improvement / lppls_prec * 100) if lppls_prec > 0 else 0

    summary = ExperimentSummary(
        n_total=len(results),
        n_real=n_real,
        n_noise=n_noise,
        lppls_tp=lppls_tp,
        lppls_fp=lppls_fp,
        lppls_fn=lppls_fn,
        lppls_tn=lppls_tn,
        lppls_precision=lppls_prec,
        lppls_recall=_recall(lppls_tp, lppls_fn),
        council_tp=council_tp,
        council_fp=council_fp,
        council_fn=council_fn,
        council_tn=council_tn,
        council_precision=council_prec,
        council_recall=_recall(council_tp, council_fn),
        precision_improvement=improvement,
        precision_improvement_pct=improvement_pct,
    )

    log.info(
        "experiment_complete",
        lppls_precision=round(lppls_prec, 3),
        council_precision=round(council_prec, 3),
        improvement_pct=round(improvement_pct, 1),
    )

    return summary
