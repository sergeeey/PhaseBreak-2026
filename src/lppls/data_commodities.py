"""Commodity supercycle datasets for LPPLS analysis.

Known commodity peaks with super-exponential growth → crash:
- Oil WTI 2008: $147 → $30
- Gold 2011: $1900 → $1050
- Oil WTI 2014: $107 → $26
- Uranium 2007: spot $136 → $40
- Silver 2011: $49 → $14

Uses same yfinance loader as finance bubbles.
"""

from src.lppls.data import load_yfinance, BubbleDataset

COMMODITY_BUBBLES: dict[str, dict] = {
    "oil_2008": {
        "ticker": "CL=F",
        "start": "2007-01-01",
        "end": "2008-07-03",
        "known_tc_date": "2008-07-03",
        "name": "WTI Oil 2008 Supercycle Peak ($147)",
    },
    "gold_2011": {
        "ticker": "GC=F",
        "start": "2009-01-01",
        "end": "2011-09-05",
        "known_tc_date": "2011-09-06",
        "name": "Gold 2011 Peak ($1900)",
    },
    "oil_2014": {
        "ticker": "CL=F",
        "start": "2013-01-01",
        "end": "2014-06-19",
        "known_tc_date": "2014-06-20",
        "name": "WTI Oil 2014 Peak ($107)",
    },
    "silver_2011": {
        "ticker": "SI=F",
        "start": "2009-06-01",
        "end": "2011-04-28",
        "known_tc_date": "2011-04-28",
        "name": "Silver 2011 Peak ($49)",
    },
    "natgas_2022": {
        "ticker": "NG=F",
        "start": "2021-06-01",
        "end": "2022-08-22",
        "known_tc_date": "2022-08-22",
        "name": "Natural Gas 2022 Peak ($9.7)",
    },
    "wheat_2022": {
        "ticker": "ZW=F",
        "start": "2021-06-01",
        "end": "2022-03-07",
        "known_tc_date": "2022-03-08",
        "name": "Wheat 2022 Peak (Ukraine war)",
    },
}

COMMODITY_CONTROLS: dict[str, dict] = {
    "oil_2017_flat": {
        "ticker": "CL=F",
        "start": "2017-01-01",
        "end": "2017-12-31",
        "known_tc_date": None,
        "name": "WTI Oil 2017 Range-Bound",
    },
    "gold_2019_stable": {
        "ticker": "GC=F",
        "start": "2018-06-01",
        "end": "2019-06-01",
        "known_tc_date": None,
        "name": "Gold 2018-19 Stable",
    },
    "silver_2023_flat": {
        "ticker": "SI=F",
        "start": "2023-01-01",
        "end": "2023-12-31",
        "known_tc_date": None,
        "name": "Silver 2023 Range-Bound",
    },
    "natgas_2024_low": {
        "ticker": "NG=F",
        "start": "2024-01-01",
        "end": "2024-12-31",
        "known_tc_date": None,
        "name": "Natural Gas 2024 Low Volatility",
    },
}


def load_commodity_episode(name: str) -> BubbleDataset:
    """Load a pre-defined commodity episode."""
    all_episodes = {**COMMODITY_BUBBLES, **COMMODITY_CONTROLS}
    if name not in all_episodes:
        available = ", ".join(all_episodes.keys())
        raise ValueError(f"Unknown commodity episode '{name}'. Available: {available}")
    return load_yfinance(**all_episodes[name])
