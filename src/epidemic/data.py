"""COVID-19 epidemic wave data for LPPLS phase transition detection.

Source: Our World in Data (JHU CSSE aggregated).
LPPLS applied to cumulative cases (not daily — daily is too noisy).
Super-exponential growth in cumulative cases → LPPLS fits wave acceleration.
tc = wave peak (inflection point in cumulative → peak in daily cases).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from numpy.typing import NDArray

log = structlog.get_logger()

OWID_URL = (
    "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/jhu/total_cases.csv"
)


@dataclass
class EpidemicDataset:
    """Preprocessed epidemic time series for LPPLS fitting."""

    name: str
    country: str
    t: NDArray
    log_cases: NDArray
    dates: NDArray
    cumulative_cases: NDArray
    peak_date: str | None = None
    frequency: str = "daily"


def load_covid_country(
    country: str,
    start: str,
    end: str,
    cache_dir: str = "data/epidemic",
) -> EpidemicDataset:
    """Load COVID-19 cumulative cases for a country.

    Uses cumulative cases (not daily) because:
    - Daily cases are too noisy for LPPLS
    - Cumulative cases show super-exponential growth during wave buildup
    - tc in cumulative = inflection point = peak of daily cases
    """
    cache_path = Path(cache_dir) / "owid_total_cases.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        log.info("downloading_covid_data", url=OWID_URL)
        df = pd.read_csv(OWID_URL)
        df.to_csv(cache_path, index=False)

    if country not in df.columns:
        available = [c for c in df.columns if c != "date"][:20]
        raise ValueError(f"Country '{country}' not found. Examples: {available}")

    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= start) & (df["date"] <= end)
    subset = df.loc[mask, ["date", country]].dropna().reset_index(drop=True)

    if len(subset) < 20:
        raise ValueError(f"Too few data points for {country}: {len(subset)}")

    cases = subset[country].values.astype(float)
    # WHY: LPPLS needs monotonically increasing data. Cumulative cases
    # can have reporting corrections (negative days). Fix by forward-filling.
    cases = np.maximum.accumulate(cases)
    cases = np.clip(cases, 1, None)  # avoid log(0)

    dates = subset["date"].values
    t = np.arange(len(cases), dtype=float)
    log_cases = np.log(cases)

    return EpidemicDataset(
        name=f"COVID-19 {country} ({start} to {end})",
        country=country,
        t=t,
        log_cases=log_cases,
        dates=dates,
        cumulative_cases=cases,
    )


# Known COVID waves for validation
COVID_WAVES: dict[str, dict] = {
    "us_wave1_2020": {
        "country": "United States",
        "start": "2020-02-15",
        "end": "2020-04-10",
        "peak_date": "2020-04-10",
        "name": "US Wave 1 (initial outbreak)",
    },
    "us_delta_2021": {
        "country": "United States",
        "start": "2021-06-01",
        "end": "2021-09-10",
        "peak_date": "2021-09-13",
        "name": "US Delta Wave 2021",
    },
    "us_omicron_2022": {
        "country": "United States",
        "start": "2021-11-15",
        "end": "2022-01-15",
        "peak_date": "2022-01-15",
        "name": "US Omicron Wave 2022",
    },
    "india_delta_2021": {
        "country": "India",
        "start": "2021-02-01",
        "end": "2021-05-06",
        "peak_date": "2021-05-07",
        "name": "India Delta Wave 2021 (catastrophic)",
    },
    "uk_alpha_2021": {
        "country": "United Kingdom",
        "start": "2020-11-01",
        "end": "2021-01-08",
        "peak_date": "2021-01-08",
        "name": "UK Alpha Wave (B.1.1.7)",
    },
    "brazil_gamma_2021": {
        "country": "Brazil",
        "start": "2021-01-01",
        "end": "2021-03-26",
        "peak_date": "2021-03-26",
        "name": "Brazil Gamma Wave (P.1)",
    },
}

COVID_CONTROLS: dict[str, dict] = {
    "us_summer_2020": {
        "country": "United States",
        "start": "2020-05-15",
        "end": "2020-06-15",
        "peak_date": None,
        "name": "US Summer 2020 (plateau)",
    },
    "japan_stable_2020": {
        "country": "Japan",
        "start": "2020-06-01",
        "end": "2020-09-30",
        "peak_date": None,
        "name": "Japan Stable Period 2020",
    },
}


def load_covid_wave(name: str) -> EpidemicDataset:
    """Load a predefined COVID wave episode."""
    all_episodes = {**COVID_WAVES, **COVID_CONTROLS}
    if name not in all_episodes:
        available = ", ".join(all_episodes.keys())
        raise ValueError(f"Unknown episode '{name}'. Available: {available}")

    config = all_episodes[name]
    ds = load_covid_country(
        country=config["country"],
        start=config["start"],
        end=config["end"],
    )
    ds.name = config["name"]
    ds.peak_date = config.get("peak_date")
    return ds
