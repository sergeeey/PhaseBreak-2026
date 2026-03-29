"""PhaseBreak CLI — scan assets for bubble signals.

Usage:
    python -m src.cli scan NVDA                    # single asset
    python -m src.cli scan NVDA TSLA BTC-USD       # multiple assets
    python -m src.cli scan NVDA --window 12m       # custom window
    python -m src.cli scan NVDA --domain commodities
    python -m src.cli verify                       # run scorecard
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta

import numpy as np
import yfinance as yf

from src.pipeline.stages import run_full_pipeline


def scan(tickers: list[str], window_months: int = 12, domain: str = "finance") -> None:
    """Scan assets for bubble signals."""
    end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=window_months * 30)).strftime("%Y-%m-%d")

    print(f"\nPhaseBreak v2 Scan — {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Window: {window_months}m ({start} → {end}) | Domain: {domain}")
    print(f"{'Ticker':<10} {'Verdict':<12} {'Quality':<8} {'tc':<12} {'HMM':<10}")
    print("-" * 55)

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty or len(df) < 20:
                print(f"{ticker:<10} INSUFFICIENT_DATA (n={len(df)})")
                continue

            if hasattr(df.columns, "levels"):
                prices = df["Close"].iloc[:, 0].dropna().values.astype(float)
            else:
                prices = df["Close"].dropna().values.astype(float)

            if len(prices) < 20:
                print(f"{ticker:<10} INSUFFICIENT_DATA (n={len(prices)})")
                continue

            t = np.arange(len(prices), dtype=float)
            result = run_full_pipeline(t, prices, domain=domain, n_bootstrap=20)

            q = result.fit.quality_score if result.fit else 0.0
            tc = ""
            if result.fit and result.fit.tc_estimate:
                tc_idx = int(result.fit.tc_estimate)
                if tc_idx < len(df.index):
                    tc = str(df.index[tc_idx].date())
                else:
                    days_ahead = tc_idx - len(df.index) + 1
                    tc = str((df.index[-1] + timedelta(days=days_ahead)).date())

            hmm = result.screening.hmm_regime or "skip"
            print(f"{ticker:<10} {result.final_verdict:<12} {q:<8.3f} {tc:<12} {hmm:<10}")

        except Exception as e:
            print(f"{ticker:<10} ERROR: {str(e)[:40]}")


def verify() -> None:
    """Run scorecard verification."""
    from src.benchmark.verify_scorecard import verify_scorecard

    verify_scorecard()


def main() -> None:
    parser = argparse.ArgumentParser(description="PhaseBreak — phase transition detector")
    sub = parser.add_subparsers(dest="command")

    scan_parser = sub.add_parser("scan", help="Scan assets for bubble signals")
    scan_parser.add_argument("tickers", nargs="+", help="Ticker symbols (e.g. NVDA BTC-USD)")
    scan_parser.add_argument("--window", default="12m", help="Data window (e.g. 6m, 12m, 18m)")
    scan_parser.add_argument(
        "--domain", default="finance", help="Domain (finance, commodities, housing)"
    )

    sub.add_parser("verify", help="Run prediction scorecard")

    args = parser.parse_args()

    if args.command == "scan":
        months = int(args.window.replace("m", ""))
        scan(args.tickers, window_months=months, domain=args.domain)
    elif args.command == "verify":
        verify()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
