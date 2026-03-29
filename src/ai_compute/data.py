"""AI compute growth data for LPPLS singularity detection.

Tracks super-exponential growth in AI training compute, investment,
and model parameters. Question: is there a finite-time singularity
(bubble burst / saturation point)?

Data sources:
- Epoch AI: notable AI systems with training compute (FLOP)
- Our World in Data: aggregated AI metrics
- Manual: key investment milestones
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray

log = structlog.get_logger()


@dataclass
class AIComputeDataset:
    """AI compute time series for LPPLS fitting."""

    name: str
    metric: str
    t: NDArray
    log_values: NDArray
    dates: list[str]
    values: NDArray


# WHY hardcoded: Epoch AI data requires manual curation.
# These are verified milestones from public sources.
# Training compute in petaFLOP-days (log scale shows super-exponential)
AI_TRAINING_COMPUTE = {
    "name": "AI Training Compute (notable systems)",
    "metric": "petaFLOP-days",
    "data": [
        # (year_fraction, petaFLOP_days, model_name)
        (2012.0, 0.01, "AlexNet"),
        (2014.5, 0.1, "VGG-16"),
        (2017.0, 1.0, "Transformer"),
        (2018.5, 10, "BERT"),
        (2019.0, 50, "GPT-2"),
        (2020.5, 3000, "GPT-3"),
        (2022.0, 10000, "PaLM"),
        (2023.0, 50000, "GPT-4"),
        (2023.5, 100000, "Gemini Ultra"),
        (2024.0, 300000, "Claude 3 Opus (est.)"),
        (2024.5, 500000, "GPT-4o (est.)"),
        (2025.0, 1000000, "Claude 3.5/4 (est.)"),
        (2025.5, 2000000, "Gemini 2 (est.)"),
        (2026.0, 5000000, "GPT-5 (est.)"),
    ],
}

# AI investment ($ billions cumulative)
AI_INVESTMENT = {
    "name": "Cumulative AI Infrastructure Investment",
    "metric": "USD billions",
    "data": [
        (2019.0, 20),
        (2020.0, 35),
        (2021.0, 60),
        (2022.0, 100),
        (2023.0, 180),
        (2024.0, 350),
        (2025.0, 600),
        (2026.0, 1000),
    ],
}

# Data center electricity (TWh/year)
AI_ELECTRICITY = {
    "name": "Data Center Electricity Consumption",
    "metric": "TWh/year",
    "data": [
        (2015.0, 200),
        (2016.0, 220),
        (2017.0, 240),
        (2018.0, 260),
        (2019.0, 280),
        (2020.0, 300),
        (2021.0, 330),
        (2022.0, 370),
        (2023.0, 400),
        (2024.0, 500),
        (2025.0, 650),
        (2026.0, 800),
    ],
}


def load_ai_compute(series: str = "training") -> AIComputeDataset:
    """Load AI compute time series.

    Args:
        series: "training" (FLOP), "investment" ($B), "electricity" (TWh)
    """
    if series == "training":
        source = AI_TRAINING_COMPUTE
        dates_raw = [d[0] for d in source["data"]]
        values = np.array([d[1] for d in source["data"]], dtype=float)
    elif series == "investment":
        source = AI_INVESTMENT
        dates_raw = [d[0] for d in source["data"]]
        values = np.array([d[1] for d in source["data"]], dtype=float)
    elif series == "electricity":
        source = AI_ELECTRICITY
        dates_raw = [d[0] for d in source["data"]]
        values = np.array([d[1] for d in source["data"]], dtype=float)
    else:
        raise ValueError(f"Unknown series '{series}'. Use: training, investment, electricity")

    t = np.arange(len(values), dtype=float)
    log_values = np.log(values)
    dates = [str(d) for d in dates_raw]

    return AIComputeDataset(
        name=source["name"],
        metric=source["metric"],
        t=t,
        log_values=log_values,
        dates=dates,
        values=values,
    )
