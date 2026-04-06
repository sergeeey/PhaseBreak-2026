"""Scheduled monitor for PhaseBreak Hybrid 2.0.

Runs LPPLS predictions + sector scan on schedule, sends Telegram alerts.
Register in Windows Task Scheduler or Unix cron.

Usage:
    python monitor_cron.py                      # run once, send alerts
    python monitor_cron.py --loop --interval 6  # run every 6 hours
    python monitor_cron.py --dry-run             # run without sending alerts
    python monitor_cron.py --sectors-only        # only sector scan
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import structlog
from rich.console import Console

from pipeline.alerter import (
    AlertPayload,
    TelegramAlerter,
    build_alert_payload,
    should_alert,
)
from pipeline.persistence import classify_all_signals
from pipeline.sector_scan import SectorScanner

log = structlog.get_logger()
console = Console()

LOG_PATH = Path(__file__).parent / "data" / "monitor_log.json"


def run_predictions() -> list[dict]:
    """Run LPPLS predictions on tracked assets. Returns list of result dicts."""
    try:
        from lppls.ds_filters import fit_lppls_simple
        import numpy as np
        import yfinance as yf

        # Core watchlist — high-profile assets
        watchlist = [
            "NVDA",
            "TSLA",
            "AAPL",
            "MSFT",
            "GOOGL",
            "META",
            "AMZN",
            "BTC-USD",
            "ETH-USD",
            "SOL-USD",
            "GC=F",
            "SI=F",
            "CL=F",
            "GME",
            "PLTR",
            "MSTR",
            "ARM",
            "AVGO",
        ]

        results = []
        for ticker in watchlist:
            try:
                df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
                if df.empty or len(df) < 30:
                    continue

                prices = df["Close"].dropna().values.flatten()
                t = np.arange(len(prices), dtype=float)
                result = fit_lppls_simple(t, prices)
                result["ticker"] = ticker
                result["price"] = float(prices[-1])
                results.append(result)
            except Exception as e:
                log.debug("prediction_failed", ticker=ticker, error=str(e))

        return results
    except ImportError as e:
        log.error("import_failed", error=str(e))
        return []


def run_monitor(
    loop: bool = False,
    interval_hours: float = 6.0,
    dry_run: bool = False,
    sectors_only: bool = False,
) -> None:
    """Main monitor loop."""
    alerter = TelegramAlerter() if not dry_run else None

    while True:
        run_start = time.time()
        all_alerts: list[AlertPayload] = []

        console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
        console.print(f"[bold]PhaseBreak Monitor — {time.strftime('%Y-%m-%d %H:%M')}[/bold]")
        console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")

        # Phase 1: Individual predictions + persistence
        if not sectors_only:
            console.print("\n[bold]Phase 1: Individual Predictions[/bold]")
            results = run_predictions()
            console.print(f"  Analyzed {len(results)} assets")

            # Persistence: track signal history, classify CONFIRMED/TENTATIVE
            enriched = classify_all_signals(results)

            for r in enriched:
                status = r.get("status", "NO_SIGNAL")
                if status == "CONFIRMED" and should_alert(r):
                    payload = build_alert_payload(r)
                    payload.top_factors.insert(
                        0, f"CONFIRMED ({r.get('consecutive_count', 0)} scans)"
                    )
                    all_alerts.append(payload)
                    console.print(
                        f"  [red]  CONFIRMED: {r['ticker']} — {r['verdict']} "
                        f"(Q={r['quality_score']:.2f}, {r.get('consecutive_count', 0)} scans)[/red]"
                    )
                elif status == "TENTATIVE" and should_alert(r):
                    console.print(
                        f"  [yellow]  TENTATIVE: {r['ticker']} — {r['verdict']} "
                        f"(Q={r['quality_score']:.2f}, {r.get('consecutive_count', 0)} scans)[/yellow]"
                    )
                elif status == "DROPPED":
                    console.print(f"  [dim]  DROPPED: {r['ticker']} — signal disappeared[/dim]")

        # Phase 2: Sector scan
        console.print("\n[bold]Phase 2: Sector Scan[/bold]")
        try:
            scanner = SectorScanner()
            report = scanner.scan_all_sectors()
            report.print_console_table()

            for sector in report.sectors:
                if sector.is_sector_alert:
                    all_alerts.append(
                        AlertPayload(
                            ticker=sector.sector_id,
                            verdict="SECTOR_ALERT",
                            quality_score=sector.heat_score,
                            domain="sector",
                            sector_name=sector.sector_name,
                            sector_tickers=sector.alert_tickers,
                        )
                    )
        except Exception as e:
            log.warning("sector_scan_failed", error=str(e))

        # Phase 3: Send alerts
        console.print(f"\n[bold]Phase 3: Alerts ({len(all_alerts)} total)[/bold]")
        if all_alerts and alerter:
            sent = alerter.send_batch(all_alerts)
            console.print(f"  Sent {sent}/{len(all_alerts)} Telegram alerts")
        elif all_alerts and dry_run:
            console.print("  [dim]DRY RUN — alerts not sent[/dim]")
            for a in all_alerts:
                console.print(f"  [dim]{a.format_telegram()[:100]}...[/dim]")
        else:
            console.print("  No alerts to send")

        # Log run
        elapsed = time.time() - run_start
        _append_log(len(all_alerts), elapsed, dry_run)
        console.print(f"\n[dim]Run complete in {elapsed:.1f}s[/dim]")

        if not loop:
            break

        console.print(f"[dim]Next run in {interval_hours}h...[/dim]")
        time.sleep(interval_hours * 3600)


def _append_log(alert_count: int, elapsed: float, dry_run: bool) -> None:
    """Append run record to monitor_log.json."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        logs = []
        if LOG_PATH.exists():
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                logs = json.load(f)

        logs.append(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "alerts": alert_count,
                "elapsed_sec": round(elapsed, 1),
                "dry_run": dry_run,
            }
        )

        # Keep last 500 entries
        logs = logs[-500:]

        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log.debug("log_write_failed", error=str(e))


def main():
    parser = argparse.ArgumentParser(description="PhaseBreak Monitor")
    parser.add_argument("--loop", action="store_true", help="Run in loop mode")
    parser.add_argument(
        "--interval", type=float, default=6.0, help="Interval in hours (default: 6)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't send alerts")
    parser.add_argument("--sectors-only", action="store_true", help="Only run sector scan")
    args = parser.parse_args()

    run_monitor(
        loop=args.loop,
        interval_hours=args.interval,
        dry_run=args.dry_run,
        sectors_only=args.sectors_only,
    )


if __name__ == "__main__":
    main()
