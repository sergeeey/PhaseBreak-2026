"""Lock live predictions — immutable record for prospective validation."""

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")

import sys
import json
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from predict_all import run_lppls_with_hmm, load_yfinance_data

WATCHLIST = [
    "NVDA",
    "AAPL",
    "MSFT",
    "GOOGL",
    "META",
    "AMZN",
    "TSLA",
    "PLTR",
    "MSTR",
    "ARM",
    "AVGO",
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "GC=F",
    "SI=F",
    "CL=F",
    "^GSPC",
    "^IXIC",
    "^DJI",
]

results = []
header = f"{'Ticker':<10} {'Price':>10} {'Verdict':<12} {'Q':>5} {'HMM':<8} {'H':>6} {'tc':>12}"
print(header)
print("-" * 70)

for ticker in WATCHLIST:
    try:
        t, prices, dates, price = load_yfinance_data(ticker)
        r = run_lppls_with_hmm(t, prices)
        r["ticker"] = ticker
        r["price"] = price
        results.append(r)
        v = r["verdict"]
        tc = r.get("tc_date_str", "") if v != "NO_BUBBLE" else ""
        H = r.get("hurst", 0)
        print(
            f"{ticker:<10} {price:>10,.2f} {v:<12} {r['quality_score']:>5.2f} {r['hmm_regime']:<8} {H:>6.3f} {tc:>12}"
        )
    except Exception as e:
        print(f"{ticker:<10} ERROR: {str(e)[:40]}")

signals = [r for r in results if r["verdict"] in ("BUBBLE", "POSSIBLE")]
print(f"\n{'=' * 70}")
print(f"SIGNALS: {len(signals)} of {len(results)}")
for s in signals:
    print(
        f"  {s['ticker']} — {s['verdict']} Q={s['quality_score']:.2f} H={s.get('hurst', 0):.3f} tc={s.get('tc_date_str', '?')}"
    )

# Lock predictions
prediction_record = {
    "prediction_date": time.strftime("%Y-%m-%d %H:%M"),
    "model": "PhaseBreak Hybrid 2.0 (LPPLS + HMM + Hurst)",
    "signals": [
        {
            "ticker": s["ticker"],
            "verdict": s["verdict"],
            "quality": s["quality_score"],
            "hurst": s.get("hurst", 0),
            "hmm_regime": s.get("hmm_regime", ""),
            "tc_date": s.get("tc_date_str", ""),
            "price_at_prediction": s.get("price", 0),
        }
        for s in signals
    ],
    "all_results": [
        {
            "ticker": r["ticker"],
            "verdict": r["verdict"],
            "quality": r["quality_score"],
            "hurst": r.get("hurst", 0),
            "hmm_regime": r.get("hmm_regime", ""),
            "price": r.get("price", 0),
        }
        for r in results
    ],
    "no_signal_count": len(results) - len(signals),
}

output_path = Path(__file__).parent / "data" / "LOCKED_PREDICTIONS_20260406.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(prediction_record, f, indent=2, ensure_ascii=False, default=str)
print(f"\nPredictions locked to {output_path}")
