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


def load_known_bubble(name: str) -> BubbleDataset:
    """Load a pre-defined known bubble dataset."""
    if name not in KNOWN_BUBBLES:
        available = ", ".join(KNOWN_BUBBLES.keys())
        raise ValueError(f"Unknown bubble '{name}'. Available: {available}")
    return load_yfinance(**KNOWN_BUBBLES[name])
