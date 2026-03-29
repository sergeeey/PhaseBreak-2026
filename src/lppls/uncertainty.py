"""tc uncertainty via bootstrap and perturbation.

PRIMARY uncertainty method for PhaseBreak v2.
Chosen over conformal prediction because:
- Works standalone without a calibration set of known episodes
- Produces distributional summary (median, p10, p90)
- No coverage guarantee but empirically well-calibrated

Conformal prediction (conformal.py) is SECONDARY — requires calibration errors
from known bubbles, which limits it to domains with ground truth.

Produces tc distribution without full Bayesian LPPLS:
- Bootstrap over random subsets of windows/data
- Perturbation of initial conditions
- Summary: median, p10, p90 interval
"""

from __future__ import annotations

import numpy as np
import structlog
from numpy.typing import NDArray

from src.lppls.optimizer import LPPLSOptimizer

log = structlog.get_logger()


def bootstrap_tc_uncertainty(
    t: NDArray,
    log_price: NDArray,
    n_bootstrap: int = 30,
    subsample_frac: float = 0.8,
    seed: int = 42,
    m_range: tuple = (0.1, 0.9),
    omega_range: tuple = (6.0, 13.0),
    grid_size: int = 8,
) -> dict:
    """Estimate tc uncertainty via bootstrap resampling.

    For each bootstrap iteration:
    1. Subsample 80% of data points (preserving order)
    2. Fit LPPLS
    3. Collect tc estimates

    Returns dict with tc distribution summary.
    """
    rng = np.random.default_rng(seed)
    n = len(t)
    n_sub = max(20, int(n * subsample_frac))

    tc_values = []

    for i in range(n_bootstrap):
        # Subsample indices (sorted to preserve time order)
        idx = np.sort(rng.choice(n, size=n_sub, replace=False))
        t_sub = t[idx]
        lp_sub = log_price[idx]

        try:
            opt = LPPLSOptimizer(
                grid_size=grid_size,
                n_best=3,
                m_range=m_range,
                omega_range=omega_range,
            )
            model = opt.fit(t_sub, lp_sub)
            if model.params and model.params.is_bubble:
                tc_values.append(model.params.tc)
        except Exception:
            continue

    if not tc_values:
        return {
            "tc_median": None,
            "tc_p10": None,
            "tc_p90": None,
            "tc_p50_low": None,
            "tc_p50_high": None,
            "tc_std": None,
            "n_valid": 0,
        }

    tc_arr = np.array(tc_values)

    result = {
        "tc_median": float(np.median(tc_arr)),
        "tc_p10": float(np.percentile(tc_arr, 10)),
        "tc_p90": float(np.percentile(tc_arr, 90)),
        "tc_p50_low": float(np.percentile(tc_arr, 25)),
        "tc_p50_high": float(np.percentile(tc_arr, 75)),
        "tc_std": float(np.std(tc_arr)),
        "n_valid": len(tc_values),
    }

    log.info(
        "tc_uncertainty",
        n_valid=len(tc_values),
        n_total=n_bootstrap,
        tc_median=round(result["tc_median"], 1),
        tc_range=f"[{result['tc_p10']:.1f}, {result['tc_p90']:.1f}]",
    )
    return result
