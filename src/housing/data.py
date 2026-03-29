"""Housing price data loaders for LPPLS analysis.

Sources:
- FHFA House Price Index (official, monthly, by state/metro)
- Zillow ZHVI (Home Value Index, monthly, by metro/zip)

Both provide long monthly time series suitable for LPPLS fitting
on slower-moving real estate markets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from numpy.typing import NDArray

log = structlog.get_logger()

FHFA_STATE_URL = "https://www.fhfa.gov/hpi/download/monthly/hpi_master.csv"
ZILLOW_ZHVI_URL = "https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"


@dataclass
class HousingDataset:
    """Preprocessed housing time series for LPPLS fitting."""

    name: str
    region: str
    source: str  # "FHFA" or "Zillow"
    t: NDArray  # integer time index (0, 1, 2, ...)
    log_price: NDArray  # ln(price_index)
    dates: NDArray  # actual dates
    values: NDArray  # raw price index values
    peak_date: str | None = None  # known peak date if available
    frequency: str = "monthly"

    @property
    def t_last(self) -> float:
        return float(self.t[-1])

    def peak_error_quarters(self, tc: float) -> int | None:
        """Quarters between predicted tc and known peak (for quarterly FHFA data)."""
        if self.peak_date is None:
            return None
        predicted_idx = int(round(tc))
        if predicted_idx < len(self.dates):
            predicted = pd.Timestamp(self.dates[predicted_idx])
        else:
            # WHY: quarterly data → each index step = 3 months
            steps_ahead = predicted_idx - len(self.dates) + 1
            step_months = 3 if self.frequency == "quarterly" else 1
            predicted = pd.Timestamp(self.dates[-1]) + pd.DateOffset(
                months=steps_ahead * step_months
            )
        actual = pd.Timestamp(self.peak_date)
        diff_months = abs((predicted.year - actual.year) * 12 + predicted.month - actual.month)
        return diff_months // 3 if self.frequency == "quarterly" else diff_months


def load_fhfa_state(
    state: str,
    start: str | None = None,
    end: str | None = None,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Load FHFA HPI for a US state.

    Args:
        state: Two-letter state code (e.g., "AZ", "NV", "FL")
        start: Start date "YYYY-MM"
        end: End date "YYYY-MM"
        cache_dir: Optional directory to cache downloaded CSV

    Returns:
        DataFrame with columns: date, hpi
    """
    cache_path = Path(cache_dir or "data/housing") / "fhfa_master.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        log.info("downloading_fhfa", url=FHFA_STATE_URL)
        df = pd.read_csv(FHFA_STATE_URL)
        df.to_csv(cache_path, index=False)

    # Filter to state-level data
    # WHY: FHFA state data is quarterly only (no monthly). Use place_id for 2-letter codes.
    state_mask = (df["level"] == "State") & (df["place_id"] == state)
    state_df = df[state_mask].copy()

    if state_df.empty:
        # Try by place_name
        state_mask = (df["level"] == "State") & (
            df["place_name"].str.contains(state, case=False, na=False)
        )
        state_df = df[state_mask].copy()

    if state_df.empty:
        raise ValueError(f"No FHFA data for state '{state}'")

    # Build date: quarterly data has period=1-4 (quarters), monthly has period=1-12
    freq = state_df["frequency"].iloc[0] if "frequency" in state_df.columns else "quarterly"
    if freq == "quarterly":
        # Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct
        quarter_to_month = {1: 1, 2: 4, 3: 7, 4: 10}
        state_df["month"] = state_df["period"].astype(int).map(quarter_to_month)
    else:
        state_df["month"] = state_df["period"].astype(int)
    state_df["date"] = pd.to_datetime(
        state_df["yr"].astype(str) + "-" + state_df["month"].astype(str).str.zfill(2) + "-01"
    )
    # Prefer seasonally adjusted, fallback to NSA
    idx_col = "index_nsa"  # SA often empty for state quarterly
    if "index_sa" in state_df.columns:
        sa_valid = state_df["index_sa"].notna() & (state_df["index_sa"] != "")
        if sa_valid.any():
            idx_col = "index_sa"
    state_df["hpi"] = pd.to_numeric(state_df[idx_col], errors="coerce")
    state_df = (
        state_df[["date", "hpi"]].dropna(subset=["hpi"]).sort_values("date").reset_index(drop=True)
    )

    if start:
        state_df = state_df[state_df["date"] >= start]
    if end:
        state_df = state_df[state_df["date"] <= end]

    return state_df.reset_index(drop=True)


def load_zillow_metro(
    metro: str,
    start: str | None = None,
    end: str | None = None,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Load Zillow ZHVI for a metro area.

    Args:
        metro: Metro name (e.g., "Phoenix, AZ", "Las Vegas, NV")
        start: Start date "YYYY-MM"
        end: End date "YYYY-MM"

    Returns:
        DataFrame with columns: date, zhvi
    """
    cache_path = Path(cache_dir or "data/housing") / "zillow_zhvi_metro.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        log.info("downloading_zillow", url=ZILLOW_ZHVI_URL)
        df = pd.read_csv(ZILLOW_ZHVI_URL)
        df.to_csv(cache_path, index=False)

    # Find metro row
    metro_row = df[df["RegionName"].str.contains(metro, case=False, na=False)]
    if metro_row.empty:
        available = df["RegionName"].head(20).tolist()
        raise ValueError(f"Metro '{metro}' not found. Examples: {available}")

    # Melt date columns (all columns after metadata)
    meta_cols = ["RegionID", "SizeRank", "RegionName", "RegionType", "StateName"]
    date_cols = [c for c in df.columns if c not in meta_cols]

    row = metro_row.iloc[0]
    dates = []
    values = []
    for col in date_cols:
        try:
            d = pd.Timestamp(col)
            v = float(row[col])
            if pd.notna(v):
                dates.append(d)
                values.append(v)
        except (ValueError, TypeError):
            continue

    result = (
        pd.DataFrame({"date": dates, "zhvi": values}).sort_values("date").reset_index(drop=True)
    )

    if start:
        result = result[result["date"] >= start]
    if end:
        result = result[result["date"] <= end]

    return result.reset_index(drop=True)


def make_housing_dataset(
    df: pd.DataFrame,
    value_col: str,
    name: str,
    region: str,
    source: str,
    peak_date: str | None = None,
    frequency: str = "quarterly",
) -> HousingDataset:
    """Convert raw DataFrame to HousingDataset for LPPLS fitting."""
    values = df[value_col].values.astype(float)
    dates = df["date"].values
    t = np.arange(len(values), dtype=float)
    log_price = np.log(values)

    return HousingDataset(
        name=name,
        region=region,
        source=source,
        t=t,
        log_price=log_price,
        dates=dates,
        values=values,
        peak_date=peak_date,
        frequency=frequency,
    )


# ============================================================
# Known housing episodes for validation
# ============================================================

HOUSING_BUBBLES: dict[str, dict] = {
    "az_2006": {
        "loader": "fhfa",
        "region": "AZ",
        "start": "2003-01",
        "end": "2006-06",
        "peak_date": "2006-06",
        "name": "Arizona 2006 Housing Bubble",
    },
    "nv_2006": {
        "loader": "fhfa",
        "region": "NV",
        "start": "2003-01",
        "end": "2006-08",
        "peak_date": "2006-08",
        "name": "Nevada 2006 Housing Bubble",
    },
    "fl_2006": {
        "loader": "fhfa",
        "region": "FL",
        "start": "2003-01",
        "end": "2006-12",
        "peak_date": "2006-12",
        "name": "Florida 2006 Housing Bubble",
    },
    "ca_2006": {
        "loader": "fhfa",
        "region": "CA",
        "start": "2003-01",
        "end": "2006-02",
        "peak_date": "2006-02",
        "name": "California 2006 Housing Bubble",
    },
    "az_2022": {
        "loader": "fhfa",
        "region": "AZ",
        "start": "2019-01",
        "end": "2022-06",
        "peak_date": "2022-06",
        "name": "Arizona 2022 COVID Housing Boom",
    },
    "id_2022": {
        "loader": "fhfa",
        "region": "ID",
        "start": "2019-01",
        "end": "2022-05",
        "peak_date": "2022-05",
        "name": "Idaho 2022 Housing Boom (Boise)",
    },
}

HOUSING_CONTROLS: dict[str, dict] = {
    "tx_2015": {
        "loader": "fhfa",
        "region": "TX",
        "start": "2013-01",
        "end": "2015-12",
        "peak_date": None,
        "name": "Texas 2013-2015 Steady Growth",
    },
    "oh_2018": {
        "loader": "fhfa",
        "region": "OH",
        "start": "2016-01",
        "end": "2018-12",
        "peak_date": None,
        "name": "Ohio 2016-2018 Slow Growth",
    },
    "pa_2014": {
        "loader": "fhfa",
        "region": "PA",
        "start": "2012-01",
        "end": "2014-12",
        "peak_date": None,
        "name": "Pennsylvania 2012-2014 Flat",
    },
    "il_2016": {
        "loader": "fhfa",
        "region": "IL",
        "start": "2014-01",
        "end": "2016-12",
        "peak_date": None,
        "name": "Illinois 2014-2016 Recovery",
    },
}


def load_housing_episode(name: str) -> HousingDataset:
    """Load a pre-defined housing episode (bubble or control)."""
    all_episodes = {**HOUSING_BUBBLES, **HOUSING_CONTROLS}
    if name not in all_episodes:
        available = ", ".join(all_episodes.keys())
        raise ValueError(f"Unknown episode '{name}'. Available: {available}")

    config = all_episodes[name]
    df = load_fhfa_state(
        state=config["region"],
        start=config["start"],
        end=config["end"],
    )
    return make_housing_dataset(
        df=df,
        value_col="hpi",
        name=config["name"],
        region=config["region"],
        source="FHFA",
        peak_date=config.get("peak_date"),
    )
