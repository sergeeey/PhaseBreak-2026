"""Strict scorecard verification for April 2026 predictions.

Checks exact binary criteria from SCORECARD_APRIL_2026.md.
No partial credit. No interpretation. Just hit/miss.

Usage:
    python -m src.benchmark.verify_scorecard
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import structlog

log = structlog.get_logger()

PREDICTION_DATE = "2026-03-28"  # last trading day before prediction
VERIFICATION_DATE = "2026-04-29"  # day after final check (yfinance end is exclusive)

# Frozen predictions with strict criteria
SCORECARD = [
    {
        "id": 1,
        "asset": "Gold",
        "ticker": "GC=F",
        "price_329": 4524.3,
        "verdict": "BUBBLE",
        "criterion": "close_428 < 4524.3",
        "threshold_type": "decline_from_prediction",
    },
    {
        "id": 2,
        "asset": "Nvidia",
        "ticker": "NVDA",
        "price_329": 167.5,
        "verdict": "NO_BUBBLE",
        "max_drawdown_pct": 15,
    },
    {
        "id": 3,
        "asset": "Bitcoin",
        "ticker": "BTC-USD",
        "price_329": 66319.7,
        "verdict": "NO_BUBBLE",
        "max_drawdown_pct": 15,
    },
    {
        "id": 4,
        "asset": "S&P 500",
        "ticker": "^GSPC",
        "price_329": 6368.9,
        "verdict": "NO_BUBBLE",
        "max_drawdown_pct": 10,
    },
    {
        "id": 5,
        "asset": "NASDAQ",
        "ticker": "^IXIC",
        "price_329": 20948.4,
        "verdict": "NO_BUBBLE",
        "max_drawdown_pct": 10,
    },
    {
        "id": 6,
        "asset": "Tesla",
        "ticker": "TSLA",
        "price_329": 361.8,
        "verdict": "NO_BUBBLE",
        "max_drawdown_pct": 15,
    },
    {
        "id": 7,
        "asset": "Ethereum",
        "ticker": "ETH-USD",
        "price_329": 1992.7,
        "verdict": "NO_BUBBLE",
        "max_drawdown_pct": 15,
    },
    {
        "id": 8,
        "asset": "Oil WTI",
        "ticker": "CL=F",
        "price_329": 99.6,
        "verdict": "NO_BUBBLE",
        "max_drawdown_pct": 15,
    },
    {
        "id": 9,
        "asset": "Nikkei",
        "ticker": "^N225",
        "price_329": 53373.1,
        "verdict": "NO_BUBBLE",
        "max_drawdown_pct": 10,
    },
    {
        "id": 10,
        "asset": "ARM",
        "ticker": "ARM",
        "price_329": 144.1,
        "verdict": "NO_BUBBLE",
        "max_drawdown_pct": 15,
    },
    {
        "id": 11,
        "asset": "Broadcom",
        "ticker": "AVGO",
        "price_329": 300.7,
        "verdict": "NO_BUBBLE",
        "max_drawdown_pct": 15,
    },
    {
        "id": 12,
        "asset": "Palantir",
        "ticker": "PLTR",
        "price_329": 143.1,
        "verdict": "NO_BUBBLE",
        "max_drawdown_pct": 15,
    },
    {
        "id": 13,
        "asset": "Coinbase",
        "ticker": "COIN",
        "price_329": 161.1,
        "verdict": "NO_BUBBLE",
        "max_drawdown_pct": 15,
    },
]


def _get_price_data(ticker: str, start: str, end: str) -> pd.Series | None:
    """Get close prices for a ticker."""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            return df["Close"].iloc[:, 0]
        return df["Close"]
    except Exception:
        return None


def verify_scorecard() -> dict:
    """Run strict binary verification on all 13 predictions."""
    today = datetime.now().strftime("%Y-%m-%d")
    end_date = min(today, VERIFICATION_DATE)

    print(f"\n{'=' * 85}")
    print(f"PhaseBreak STRICT Scorecard — checked {today}")
    print(f"Predictions from: {PREDICTION_DATE}  |  Final check: {VERIFICATION_DATE}")
    print(f"{'=' * 85}\n")

    bubble_hits = bubble_total = 0
    nobubble_hits = nobubble_total = 0
    results = []

    print(
        f"{'#':<3} {'Asset':<12} {'Verdict':<11} {'Price 3/29':<11} {'Now/Min':<11} {'Draw%':<7} {'Result'}"
    )
    print("-" * 85)

    for pred in SCORECARD:
        ticker = pred["ticker"]
        prices = _get_price_data(ticker, PREDICTION_DATE, end_date)

        if prices is None or len(prices) < 1:
            print(
                f"{pred['id']:<3} {pred['asset']:<12} {pred['verdict']:<11} {'N/A':<11} {'N/A':<11} {'N/A':<7} SKIP"
            )
            results.append({**pred, "result": "SKIP"})
            continue

        current_price = float(prices.iloc[-1])
        min_price = float(prices.min())
        price_329 = pred["price_329"]
        drawdown_pct = (price_329 - min_price) / price_329 * 100

        if pred["verdict"] == "BUBBLE":
            # HIT if close on last available day < prediction price
            hit = current_price < price_329
            bubble_total += 1
            if hit:
                bubble_hits += 1
            change = (current_price - price_329) / price_329 * 100
            display_price = f"${current_price:.1f}"
            display_draw = f"{change:+.1f}%"
        else:
            # HIT if max drawdown within threshold
            threshold = pred["max_drawdown_pct"]
            hit = drawdown_pct <= threshold
            nobubble_total += 1
            if hit:
                nobubble_hits += 1
            display_price = f"${min_price:.1f}"
            display_draw = f"-{drawdown_pct:.1f}%"

        result_str = "HIT" if hit else "MISS"
        results.append(
            {
                **pred,
                "current_price": current_price,
                "min_price": min_price,
                "drawdown_pct": round(drawdown_pct, 1),
                "result": result_str,
            }
        )

        print(
            f"{pred['id']:<3} {pred['asset']:<12} {pred['verdict']:<11} "
            f"${price_329:<10.1f} {display_price:<11} {display_draw:<7} {result_str}"
        )

    print("-" * 85)
    total_hits = bubble_hits + nobubble_hits
    total = bubble_total + nobubble_total

    print(f"\nBUBBLE:    {bubble_hits}/{bubble_total}")
    print(f"NO_BUBBLE: {nobubble_hits}/{nobubble_total}")
    print(f"OVERALL:   {total_hits}/{total} ({total_hits / total * 100:.0f}%)" if total > 0 else "")

    # False negatives: NO_BUBBLE that crashed
    fn = nobubble_total - nobubble_hits
    print(f"False negatives (missed crashes): {fn}/{nobubble_total}")

    output = {
        "check_date": today,
        "bubble_hits": bubble_hits,
        "bubble_total": bubble_total,
        "nobubble_hits": nobubble_hits,
        "nobubble_total": nobubble_total,
        "overall_hits": total_hits,
        "overall_total": total,
        "false_negatives": fn,
        "results": results,
    }

    out_path = Path("data/predictions") / f"scorecard_{today}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    return output


if __name__ == "__main__":
    verify_scorecard()
