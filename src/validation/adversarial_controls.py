"""Hard negative controls and adversarial benchmarks.

Goes beyond simple "boring periods" — includes tricky cases:
- V-shaped recovery (looks like bubble exit but isn't entry)
- Monotonic rally without crash
- Noisy oscillations with fake frequency
- Exogenous shock (not endogenous bubble)
"""

from __future__ import annotations

import numpy as np
import structlog
from numpy.typing import NDArray

from src.lppls.model import LPPLS, LPPLSParams

log = structlog.get_logger()


def generate_adversarial_series(seed: int = 42) -> list[tuple[str, NDArray, NDArray, bool]]:
    """Generate adversarial test cases for LPPLS detector.

    Returns list of (name, t, values, expected_bubble).
    """
    rng = np.random.default_rng(seed)
    cases = []

    # 1. V-shaped recovery (NOT a bubble)
    n = 200
    t = np.arange(n, dtype=float)
    v = 100 * np.exp(-0.005 * np.abs(t - 100) + rng.normal(0, 0.01, n))
    cases.append(("v_recovery", t, v, False))

    # 2. Monotonic rally without crash (NOT a bubble — no tc within horizon)
    v2 = 100 * np.exp(0.003 * t + rng.normal(0, 0.01, n))
    cases.append(("monotonic_rally", t, v2, False))

    # 3. Fake oscillations (sine wave, not log-periodic)
    v3 = 100 + 10 * np.sin(2 * np.pi * t / 30) + rng.normal(0, 1, n)
    v3 = np.clip(v3, 50, None)
    cases.append(("fake_oscillation", t, v3, False))

    # 4. Regime shift without LPPLS (sudden jump, not gradual)
    v4 = np.concatenate([100 + rng.normal(0, 1, 100), 120 + rng.normal(0, 1, 100)])
    cases.append(("regime_jump", t, v4, False))

    # 5. Real LPPLS signal (should be detected)
    params = LPPLSParams(tc=220, m=0.5, omega=8.0, A=4.6, B=-0.3, C1=0.01, C2=0.005)
    v5 = np.exp(
        LPPLS.func(t, params.tc, params.m, params.omega, params.A, params.B, params.C1, params.C2)
    )
    v5 += rng.normal(0, 0.5, n)
    v5 = np.clip(v5, 1, None)
    cases.append(("real_lppls_signal", t, v5, True))

    # 6. Volatile but no trend (high vol, flat mean)
    v6 = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    cases.append(("high_vol_flat", t, v6, False))

    return cases


def run_adversarial_benchmark(detector_fn) -> dict:
    """Run adversarial benchmark suite against a detector function.

    Args:
        detector_fn: callable(t, values) -> bool (True = bubble detected)

    Returns dict with TP, FP, FN, TN and per-case results.
    """
    cases = generate_adversarial_series()
    results = []

    for name, t, values, expected in cases:
        try:
            predicted = detector_fn(t, values)
        except Exception:
            predicted = False

        correct = predicted == expected
        results.append(
            {"name": name, "expected": expected, "predicted": predicted, "correct": correct}
        )

    tp = sum(1 for r in results if r["expected"] and r["predicted"])
    fp = sum(1 for r in results if not r["expected"] and r["predicted"])
    fn = sum(1 for r in results if r["expected"] and not r["predicted"])
    tn = sum(1 for r in results if not r["expected"] and not r["predicted"])

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "cases": results,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "adversarial_fp_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
    }
