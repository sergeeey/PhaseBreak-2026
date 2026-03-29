"""Triple split and walk-forward validation protocol.

Separates episodes into train / validation / test:
- Train: tune thresholds and scoring weights
- Validation: select best configuration
- Test: final one-time evaluation (no tuning)
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

log = structlog.get_logger()


@dataclass
class SplitProtocol:
    """Defines which episodes go into train/val/test."""

    train: list[str]
    validation: list[str]
    test: list[str]


# Finance: chronological split
FINANCE_SPLIT = SplitProtocol(
    train=["dotcom_2000", "china_2015", "sp500_2013_steady", "nasdaq_2016_normal"],
    validation=["btc_2017", "btc_2019_sideways", "gold_2022_range"],
    test=["btc_2021", "tesla_2021", "sp500_2020", "tsla_2023_recovery", "shanghai_2018_decline"],
)

# Housing: temporal split (2006 → train, 2022 → test)
HOUSING_SPLIT = SplitProtocol(
    train=["az_2006", "nv_2006", "fl_2006", "ca_2006", "oh_2018", "pa_2014"],
    validation=["tx_2015", "il_2016"],
    test=["az_2022", "id_2022"],
)

# Commodities: older → train, recent → test
COMMODITIES_SPLIT = SplitProtocol(
    train=["oil_2008", "gold_2011", "oil_2017_flat", "gold_2019_stable"],
    validation=["oil_2014", "silver_2011", "silver_2023_flat"],
    test=["natgas_2022", "wheat_2022", "natgas_2024_low"],
)


def get_split(domain: str) -> SplitProtocol:
    """Get the split protocol for a domain."""
    splits = {
        "finance": FINANCE_SPLIT,
        "housing": HOUSING_SPLIT,
        "commodities": COMMODITIES_SPLIT,
    }
    if domain not in splits:
        raise ValueError(f"No split for domain '{domain}'. Available: {list(splits.keys())}")
    return splits[domain]
