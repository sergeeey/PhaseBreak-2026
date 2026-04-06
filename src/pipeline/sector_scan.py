"""Sector Scan — parallel LPPLS analysis across sectors.

Runs fit_lppls_simple on all tickers in defined sectors using ThreadPoolExecutor.
Detects sector-wide contagion when 3+ tickers show bubble signals.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import yaml
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

log = structlog.get_logger()
console = Console()


class TickerResult(BaseModel):
    """Result of LPPLS fit for a single ticker."""

    ticker: str
    verdict: str = "NO_BUBBLE"
    quality_score: float = 0.0
    r_squared: float = 0.0
    tc_date_str: str | None = None
    m: float = 0.0
    omega: float = 0.0
    error: str | None = None


class SectorResult(BaseModel):
    """Aggregated result for one sector."""

    sector_id: str
    sector_name: str
    total_tickers: int = 0
    bubble_count: int = 0
    possible_count: int = 0
    no_bubble_count: int = 0
    failed_count: int = 0
    is_sector_alert: bool = False
    alert_tickers: list[str] = Field(default_factory=list)
    ticker_results: list[TickerResult] = Field(default_factory=list)

    @property
    def heat_score(self) -> float:
        """0.0-1.0 sector heat. BUBBLE=1.0, POSSIBLE=0.5."""
        if self.total_tickers == 0:
            return 0.0
        return (self.bubble_count * 1.0 + self.possible_count * 0.5) / self.total_tickers


class ScanReport(BaseModel):
    """Full sector scan report."""

    scan_date: str = ""
    elapsed_seconds: float = 0.0
    sectors: list[SectorResult] = Field(default_factory=list)
    sector_alerts: list[str] = Field(default_factory=list)

    def to_heatmap_json(self) -> str:
        """Export as JSON for visualization."""
        import json

        data = {
            "scan_date": self.scan_date,
            "elapsed_seconds": self.elapsed_seconds,
            "sectors": [
                {
                    "id": s.sector_id,
                    "name": s.sector_name,
                    "heat_score": round(s.heat_score, 3),
                    "bubble": s.bubble_count,
                    "possible": s.possible_count,
                    "total": s.total_tickers,
                    "is_alert": s.is_sector_alert,
                    "alert_tickers": s.alert_tickers,
                }
                for s in self.sectors
            ],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def print_console_table(self) -> None:
        """Print Rich table to console."""
        table = Table(title="Sector Scan Results", show_header=True, header_style="bold cyan")
        table.add_column("Sector", style="white", width=18)
        table.add_column("Heat", justify="right", width=6)
        table.add_column("BUBBLE", justify="right", style="red", width=8)
        table.add_column("POSSIBLE", justify="right", style="yellow", width=10)
        table.add_column("NO_BUBBLE", justify="right", style="green", width=11)
        table.add_column("Failed", justify="right", style="dim", width=8)
        table.add_column("Alert", justify="center", width=7)

        for s in self.sectors:
            alert_icon = "\U0001f6a8" if s.is_sector_alert else ""
            heat = f"{s.heat_score:.0%}"
            table.add_row(
                s.sector_name,
                heat,
                str(s.bubble_count),
                str(s.possible_count),
                str(s.no_bubble_count),
                str(s.failed_count),
                alert_icon,
            )

        console.print()
        console.print(table)

        if self.sector_alerts:
            console.print(f"\n[bold red]SECTOR ALERTS: {', '.join(self.sector_alerts)}[/bold red]")
        console.print(f"[dim]Scan time: {self.elapsed_seconds:.1f}s[/dim]\n")


class SectorScanner:
    """Parallel sector scan with LPPLS analysis."""

    def __init__(
        self,
        sectors_config_path: str = "configs/sectors.yaml",
        alert_threshold: int = 3,
        max_workers: int = 8,
        period: str = "2y",
    ):
        config_path = Path(__file__).parent.parent.parent / sectors_config_path
        if not config_path.exists():
            raise FileNotFoundError(f"Sector config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self._sectors = config.get("sectors", {})
        self._alert_threshold = config.get("alert_threshold", alert_threshold)
        self._max_workers = max_workers
        self._period = period

        total_tickers = sum(len(s["tickers"]) for s in self._sectors.values())
        log.info(
            "sector_scanner_initialized",
            sectors=len(self._sectors),
            tickers=total_tickers,
            threshold=self._alert_threshold,
        )

    def scan_all_sectors(self) -> ScanReport:
        """Run full sector scan. Main entry point."""
        start = time.time()
        results = []
        alerts = []

        for sector_id, sector_data in self._sectors.items():
            sector_result = self.scan_sector(
                sector_id=sector_id,
                sector_name=sector_data["name"],
                tickers=sector_data["tickers"],
            )
            results.append(sector_result)
            if sector_result.is_sector_alert:
                alerts.append(sector_result.sector_name)

        elapsed = time.time() - start

        report = ScanReport(
            scan_date=time.strftime("%Y-%m-%d %H:%M"),
            elapsed_seconds=round(elapsed, 1),
            sectors=results,
            sector_alerts=alerts,
        )

        log.info(
            "sector_scan_complete",
            sectors=len(results),
            alerts=len(alerts),
            elapsed=round(elapsed, 1),
        )
        return report

    def scan_sector(
        self,
        sector_id: str,
        sector_name: str,
        tickers: list[str],
    ) -> SectorResult:
        """Scan a single sector (all tickers in parallel)."""
        console.print(f"  Scanning [bold]{sector_name}[/bold] ({len(tickers)} tickers)...")

        ticker_results = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {executor.submit(self._analyze_ticker, t): t for t in tickers}

            for future in futures:
                ticker = futures[future]
                try:
                    result = future.result(timeout=30)
                    ticker_results.append(result)
                except FuturesTimeout:
                    ticker_results.append(TickerResult(ticker=ticker, error="timeout"))
                except Exception as e:
                    ticker_results.append(TickerResult(ticker=ticker, error=str(e)))

        bubble = sum(1 for r in ticker_results if r.verdict == "BUBBLE")
        possible = sum(1 for r in ticker_results if r.verdict == "POSSIBLE")
        no_bubble = sum(1 for r in ticker_results if r.verdict == "NO_BUBBLE")
        failed = sum(1 for r in ticker_results if r.error)

        alert_tickers = [r.ticker for r in ticker_results if r.verdict in ("BUBBLE", "POSSIBLE")]
        is_alert = (bubble + possible) >= self._alert_threshold

        return SectorResult(
            sector_id=sector_id,
            sector_name=sector_name,
            total_tickers=len(tickers),
            bubble_count=bubble,
            possible_count=possible,
            no_bubble_count=no_bubble,
            failed_count=failed,
            is_sector_alert=is_alert,
            alert_tickers=alert_tickers,
            ticker_results=ticker_results,
        )

    def _analyze_ticker(self, ticker: str) -> TickerResult:
        """Analyze single ticker with LPPLS. Catches all exceptions."""
        try:
            import yfinance as yf
            from lppls.ds_filters import fit_lppls_simple

            df = yf.download(ticker, period=self._period, progress=False, auto_adjust=True)
            if df.empty or len(df) < 30:
                return TickerResult(ticker=ticker, error="insufficient_data")

            prices = df["Close"].dropna().values.flatten()
            t = np.arange(len(prices), dtype=float)

            result = fit_lppls_simple(t, prices)

            return TickerResult(
                ticker=ticker,
                verdict=result.get("verdict", "NO_BUBBLE"),
                quality_score=result.get("quality_score", 0.0),
                r_squared=result.get("r_squared", 0.0),
                tc_date_str=result.get("tc_date_str"),
                m=result.get("m", 0.0),
                omega=result.get("omega", 0.0),
            )
        except Exception as e:
            log.debug("ticker_analysis_failed", ticker=ticker, error=str(e))
            return TickerResult(ticker=ticker, error=str(e))


def main():
    """CLI entry point for sector scan."""
    import argparse

    parser = argparse.ArgumentParser(description="PhaseBreak Sector Scan")
    parser.add_argument("--period", default="2y", help="yfinance period (default: 2y)")
    parser.add_argument("--workers", type=int, default=8, help="Max parallel workers")
    parser.add_argument("--output", type=str, default=None, help="Save heatmap JSON to file")
    args = parser.parse_args()

    scanner = SectorScanner(max_workers=args.workers, period=args.period)
    report = scanner.scan_all_sectors()
    report.print_console_table()

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report.to_heatmap_json())
        console.print(f"[dim]Heatmap saved to: {args.output}[/dim]")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    main()
