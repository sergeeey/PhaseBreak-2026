"""Verify April 2026 predictions against actual market data.

Run this script after April 30, 2026 to check how predictions held up.

Usage:
    python -m src.benchmark.verify_predictions
    python -m src.benchmark.verify_predictions --date 2026-04-15  # mid-month check
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yfinance as yf
import structlog

log = structlog.get_logger()

PREDICTIONS_PATH = Path("data/predictions/april_2026_predictions.json")

TICKER_MAP = {
    "Gold (GC=F)": "GC=F",
    "Solana (SOL-USD)": "SOL-USD",
    "MicroStrategy (MSTR)": "MSTR",
    "Nvidia (NVDA)": "NVDA",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "S&P 500 (^GSPC)": "^GSPC",
    "NASDAQ (^IXIC)": "^IXIC",
    "Tesla (TSLA)": "TSLA",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Oil WTI (CL=F)": "CL=F",
    "Nikkei 225 (^N225)": "^N225",
    "ARM Holdings (ARM)": "ARM",
    "Broadcom (AVGO)": "AVGO",
    "Palantir (PLTR)": "PLTR",
    "Coinbase (COIN)": "COIN",
}


def load_predictions() -> dict:
    with open(PREDICTIONS_PATH) as f:
        return json.load(f)


def get_current_price(ticker: str) -> float | None:
    try:
        df = yf.download(ticker, period="5d", progress=False)
        if df.empty:
            return None
        if hasattr(df.columns, "levels"):
            return float(df["Close"].iloc[-1, 0])
        return float(df["Close"].iloc[-1])
    except Exception:
        return None


def verify(check_date: str | None = None) -> dict:
    """Verify predictions against current prices."""
    predictions = load_predictions()
    check = check_date or datetime.now().strftime("%Y-%m-%d")

    results = []
    correct = 0
    total = 0

    print(f"\n{'=' * 80}")
    print(f"PhaseBreak Prediction Verification — {check}")
    print(f"Predictions made: {predictions['prediction_date']}")
    print(f"{'=' * 80}\n")
    print(f"{'Asset':<25} {'Verdict':<12} {'Pred $':<10} {'Now $':<10} {'Change':<10} {'Status'}")
    print("-" * 80)

    for pred in predictions["predictions"]:
        asset = pred["asset"]
        ticker = TICKER_MAP.get(asset)
        if not ticker:
            continue

        price_then = pred.get("price_at_prediction", 0)
        price_now = get_current_price(ticker)

        if price_now is None or price_then == 0:
            print(f"{asset:<25} {pred['verdict']:<12} {'N/A':<10} {'N/A':<10} {'N/A':<10} SKIP")
            continue

        change_pct = (price_now - price_then) / price_then * 100
        verdict = pred["verdict"]

        # Evaluate
        if verdict == "BUBBLE":
            # Bubble = expect crash. Confirmed if dropped >10%
            if change_pct < -10:
                status = "CONFIRMED (crashed)"
            elif change_pct > 10:
                status = "REFUTED (still rising)"
            else:
                status = "PENDING (±10%)"
        elif verdict == "POSSIBLE":
            if change_pct < -15:
                status = "CONFIRMED (crashed)"
            elif change_pct > 15:
                status = "REFUTED (rising)"
            else:
                status = "PENDING"
        else:  # NO_BUBBLE
            if abs(change_pct) < 20:
                status = "CONFIRMED (stable)"
                correct += 1
            else:
                status = "REFUTED (big move)"
            total += 1

        if verdict in ("BUBBLE", "POSSIBLE"):
            if "CONFIRMED" in status:
                correct += 1
            total += 1

        results.append(
            {
                "asset": asset,
                "verdict": verdict,
                "price_prediction": price_then,
                "price_current": round(price_now, 2),
                "change_pct": round(change_pct, 1),
                "status": status,
            }
        )

        print(
            f"{asset:<25} {verdict:<12} {price_then:<10.1f} {price_now:<10.1f} "
            f"{change_pct:>+8.1f}%  {status}"
        )

    print("-" * 80)
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\nScoreable: {total}, Correct: {correct}, Accuracy: {accuracy:.0f}%")

    # Save verification
    output = {
        "verification_date": check,
        "prediction_date": predictions["prediction_date"],
        "results": results,
        "scoreable": total,
        "correct": correct,
        "accuracy": round(accuracy, 1),
    }

    out_path = Path("data/predictions") / f"verification_{check}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    return output


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith("--date") else None
    if date and "=" in date:
        date = date.split("=")[1]
    elif len(sys.argv) > 2:
        date = sys.argv[2]
    verify(date)
