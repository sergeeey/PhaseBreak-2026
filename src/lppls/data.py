"""Data loading utilities for LPPLS analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from numpy.typing import NDArray


@dataclass
class BubbleDataset:
    """Preprocessed time series for LPPLS fitting."""

    name: str
    t: NDArray  # integer time index (0, 1, 2, ...)
    log_price: NDArray  # ln(price)
    dates: NDArray  # actual dates
    prices: NDArray  # raw prices
    known_tc_date: str | None = None  # ground truth crash date if known

    @property
    def t_last(self) -> float:
        return float(self.t[-1])

    def tc_to_date(self, tc: float) -> str:
        """Convert tc index back to calendar date."""
        idx = int(round(tc))
        if idx < len(self.dates):
            return str(self.dates[idx])
        # Extrapolate
        days_ahead = idx - len(self.dates) + 1
        last_date = pd.Timestamp(self.dates[-1])
        return str((last_date + pd.Timedelta(days=days_ahead)).date())

    def tc_error_days(self, tc: float) -> int | None:
        """Days between predicted tc and known crash date."""
        if self.known_tc_date is None:
            return None
        predicted = pd.Timestamp(self.tc_to_date(tc))
        actual = pd.Timestamp(self.known_tc_date)
        return abs((predicted - actual).days)


def load_yfinance(
    ticker: str,
    start: str,
    end: str,
    name: str | None = None,
    known_tc_date: str | None = None,
) -> BubbleDataset:
    """Load price data from Yahoo Finance.

    Args:
        ticker: Yahoo Finance ticker (e.g., "BTC-USD", "^IXIC")
        start: Start date "YYYY-MM-DD"
        end: End date "YYYY-MM-DD"
        name: Human-readable name for this dataset
        known_tc_date: Ground truth crash date if known
    """
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker} from {start} to {end}")

    # Handle multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].iloc[:, 0].dropna()
    else:
        close = df["Close"].dropna()

    prices = close.values.astype(float)
    dates = close.index.values
    t = np.arange(len(prices), dtype=float)
    log_price = np.log(prices)

    return BubbleDataset(
        name=name or f"{ticker} ({start} to {end})",
        t=t,
        log_price=log_price,
        dates=dates,
        prices=prices,
        known_tc_date=known_tc_date,
    )


# Pre-defined known bubble datasets for validation
KNOWN_BUBBLES: dict[str, dict] = {
    "btc_2017": {
        "ticker": "BTC-USD",
        "start": "2017-01-01",
        "end": "2017-12-16",
        "known_tc_date": "2017-12-17",
        "name": "Bitcoin 2017 Bubble",
    },
    "btc_2021": {
        "ticker": "BTC-USD",
        "start": "2021-01-01",
        "end": "2021-11-09",
        "known_tc_date": "2021-11-10",
        "name": "Bitcoin 2021 Bubble",
    },
    "dotcom_2000": {
        "ticker": "^IXIC",
        "start": "1998-01-01",
        "end": "2000-03-09",
        "known_tc_date": "2000-03-10",
        "name": "Dot-com Bubble 2000",
    },
    "tesla_2021": {
        "ticker": "TSLA",
        "start": "2020-06-01",
        "end": "2021-11-03",
        "known_tc_date": "2021-11-04",
        "name": "Tesla 2021 Peak",
    },
    "china_2015": {
        "ticker": "000001.SS",
        "start": "2014-06-01",
        "end": "2015-06-11",
        "known_tc_date": "2015-06-12",
        "name": "Shanghai 2015 Bubble",
    },
    "sp500_2020": {
        "ticker": "^GSPC",
        "start": "2019-01-01",
        "end": "2020-02-19",
        "known_tc_date": "2020-02-20",
        "name": "S&P 500 Pre-COVID Peak",
    },
}


# Negative controls: periods WITHOUT bubbles (LPPLS should NOT detect)
NEGATIVE_CONTROLS: dict[str, dict] = {
    "sp500_2013_steady": {
        "ticker": "^GSPC",
        "start": "2013-01-01",
        "end": "2014-06-30",
        "known_tc_date": None,
        "name": "S&P 500 2013-14 Steady Growth (no crash)",
    },
    "btc_2019_sideways": {
        "ticker": "BTC-USD",
        "start": "2019-01-01",
        "end": "2019-09-30",
        "known_tc_date": None,
        "name": "Bitcoin 2019 Sideways (no bubble)",
    },
    "nasdaq_2016_normal": {
        "ticker": "^IXIC",
        "start": "2016-01-01",
        "end": "2016-12-31",
        "known_tc_date": None,
        "name": "NASDAQ 2016 Normal Growth",
    },
    "gold_2022_range": {
        "ticker": "GC=F",
        "start": "2022-01-01",
        "end": "2022-12-31",
        "known_tc_date": None,
        "name": "Gold 2022 Range-Bound",
    },
    "tsla_2023_recovery": {
        "ticker": "TSLA",
        "start": "2023-01-01",
        "end": "2023-09-30",
        "known_tc_date": None,
        "name": "Tesla 2023 Recovery (no bubble)",
    },
    "shanghai_2018_decline": {
        "ticker": "000001.SS",
        "start": "2018-01-01",
        "end": "2018-12-31",
        "known_tc_date": None,
        "name": "Shanghai 2018 Decline (not bubble)",
    },
}


# ============================================================
# Time-forward validation: 2024-2025 episodes
# These were NOT seen during any threshold tuning.
# ============================================================

FORWARD_BUBBLES: dict[str, dict] = {
    "nvidia_2024": {
        "ticker": "NVDA",
        "start": "2023-01-01",
        "end": "2024-06-18",
        "known_tc_date": "2024-06-20",
        "name": "Nvidia 2024 AI Bubble Peak ($140)",
    },
    "nikkei_2024": {
        "ticker": "^N225",
        "start": "2023-06-01",
        "end": "2024-07-10",
        "known_tc_date": "2024-07-11",
        "name": "Nikkei 2024 ATH Peak (42k, crashed Aug 5)",
    },
    "btc_2024": {
        "ticker": "BTC-USD",
        "start": "2023-10-01",
        "end": "2024-03-13",
        "known_tc_date": "2024-03-14",
        "name": "Bitcoin 2024 ETF Rally Peak ($73k)",
    },
}

FORWARD_CONTROLS: dict[str, dict] = {
    "sp500_2024_steady": {
        "ticker": "^GSPC",
        "start": "2024-01-01",
        "end": "2024-12-31",
        "known_tc_date": None,
        "name": "S&P 500 2024 Steady Growth (no crash)",
    },
    "gold_2024_rise": {
        "ticker": "GC=F",
        "start": "2024-01-01",
        "end": "2024-12-31",
        "known_tc_date": None,
        "name": "Gold 2024 Steady Rise (no crash)",
    },
    "msft_2024_stable": {
        "ticker": "MSFT",
        "start": "2024-01-01",
        "end": "2024-12-31",
        "known_tc_date": None,
        "name": "Microsoft 2024 Stable Growth",
    },
}


def load_forward_episode(name: str) -> BubbleDataset:
    """Load a time-forward validation episode (unseen during tuning)."""
    all_episodes = {**FORWARD_BUBBLES, **FORWARD_CONTROLS}
    if name not in all_episodes:
        available = ", ".join(all_episodes.keys())
        raise ValueError(f"Unknown forward episode '{name}'. Available: {available}")
    return load_yfinance(**all_episodes[name])


def load_known_bubble(name: str) -> BubbleDataset:
    """Load a pre-defined known bubble dataset."""
    if name not in KNOWN_BUBBLES:
        available = ", ".join(KNOWN_BUBBLES.keys())
        raise ValueError(f"Unknown bubble '{name}'. Available: {available}")
    return load_yfinance(**KNOWN_BUBBLES[name])


def load_negative_control(name: str) -> BubbleDataset:
    """Load a pre-defined negative control dataset (no bubble expected)."""
    if name not in NEGATIVE_CONTROLS:
        available = ", ".join(NEGATIVE_CONTROLS.keys())
        raise ValueError(f"Unknown control '{name}'. Available: {available}")
    return load_yfinance(**NEGATIVE_CONTROLS[name])
