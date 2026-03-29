"""Landslide displacement data for LPPLS phase transition detection.

Based on Lei & Sornette (2025): LPPLS on 52 landslide crises showed
statistical evidence of log-periodic oscillations in displacement.
Same physics as financial bubbles: hierarchical damage accumulation
follows power-law acceleration with finite-time singularity.

Real USGS data is station-specific and requires manual download.
We use synthetic displacement curves calibrated to published parameters
from Lei 2025 (m ≈ 0.2-0.4, same range as finance).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray

from src.lppls.model import LPPLS, LPPLSParams

log = structlog.get_logger()


@dataclass
class LandslideDataset:
    """Landslide displacement time series for LPPLS fitting."""

    name: str
    t: NDArray
    displacement: NDArray  # cumulative displacement (mm)
    log_displacement: NDArray
    failure_date: str | None = None


# WHY synthetic: USGS raw displacement data requires per-station download
# and processing. These are calibrated to published LPPLS parameters from
# Lei & Sornette (2025) Table 1: m ∈ [0.15, 0.45], ω ∈ [5, 12].
LANDSLIDE_EPISODES = {
    "vaiont_1963": {
        "name": "Vaiont Dam Landslide 1963 (Italy, 2000 deaths)",
        "params": LPPLSParams(tc=100, m=0.25, omega=8.0, A=3.5, B=-0.8, C1=0.05, C2=0.02),
        "n_points": 80,
        "noise": 0.03,
        "failure_date": "1963-10-09",
    },
    "randa_1991": {
        "name": "Randa Rockslide 1991 (Switzerland)",
        "params": LPPLSParams(tc=120, m=0.30, omega=9.5, A=4.0, B=-0.5, C1=0.03, C2=0.01),
        "n_points": 100,
        "noise": 0.02,
        "failure_date": "1991-04-18",
    },
    "asamushi_2015": {
        "name": "Asamushi Landslide 2015 (Japan, GPS monitored)",
        "params": LPPLSParams(tc=150, m=0.35, omega=7.5, A=3.8, B=-0.6, C1=0.04, C2=0.015),
        "n_points": 120,
        "noise": 0.025,
        "failure_date": "2015-08-20",
    },
    "xinmo_2017": {
        "name": "Xinmo Landslide 2017 (China, catastrophic)",
        "params": LPPLSParams(tc=90, m=0.20, omega=10.0, A=3.2, B=-1.0, C1=0.06, C2=0.03),
        "n_points": 70,
        "noise": 0.04,
        "failure_date": "2017-06-24",
    },
    "bingham_canyon_2013": {
        "name": "Bingham Canyon Mine Slide 2013 (USA, largest in N.America)",
        "params": LPPLSParams(tc=110, m=0.40, omega=6.5, A=4.2, B=-0.4, C1=0.02, C2=0.01),
        "n_points": 90,
        "noise": 0.02,
        "failure_date": "2013-04-10",
    },
    "stable_slope_1": {
        "name": "Stable Slope (linear creep, no failure)",
        "params": None,
        "n_points": 100,
        "noise": 0.01,
        "failure_date": None,
    },
    "stable_slope_2": {
        "name": "Stable Slope (seasonal fluctuation, no failure)",
        "params": None,
        "n_points": 120,
        "noise": 0.015,
        "failure_date": None,
    },
}


def _generate_synthetic_landslide(config: dict, seed: int = 42) -> LandslideDataset:
    """Generate synthetic landslide displacement from LPPLS params."""
    rng = np.random.default_rng(seed)
    n = config["n_points"]
    t = np.arange(n, dtype=float)

    if config["params"] is not None:
        p = config["params"]
        lp = LPPLS.func(t, p.tc, p.m, p.omega, p.A, p.B, p.C1, p.C2)
        lp += rng.normal(0, config["noise"], n)
        displacement = np.exp(lp)
    else:
        # Stable slope: linear creep + seasonal + noise
        displacement = (
            10 + 0.1 * t + 2 * np.sin(2 * np.pi * t / 60) + rng.normal(0, config["noise"] * 10, n)
        )
        displacement = np.clip(displacement, 1, None)
        lp = np.log(displacement)

    return LandslideDataset(
        name=config["name"],
        t=t,
        displacement=displacement,
        log_displacement=lp,
        failure_date=config.get("failure_date"),
    )


def load_landslide_episode(name: str, seed: int = 42) -> LandslideDataset:
    """Load a landslide episode (synthetic, calibrated to published params)."""
    if name not in LANDSLIDE_EPISODES:
        available = ", ".join(LANDSLIDE_EPISODES.keys())
        raise ValueError(f"Unknown episode '{name}'. Available: {available}")
    return _generate_synthetic_landslide(LANDSLIDE_EPISODES[name], seed=seed)
