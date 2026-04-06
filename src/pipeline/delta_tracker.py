"""Delta Tracker — Continuous Intelligence for PhaseBreak predictions.

Compares current predictions with previous runs to detect:
- New bubble signals
- Dropped bubble signals
- tc date shifts (with stability metric)
- Quality score changes

Usage:
    tracker = DeltaTracker()
    delta = tracker.compare_and_save(current_predictions)
    tracker.print_delta(delta)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel

log = structlog.get_logger()
console = Console()


@dataclass
class DeltaReport:
    """Report of changes between prediction runs."""

    date: str
    new_bubbles: list[str] = field(default_factory=list)
    dropped_bubbles: list[str] = field(default_factory=list)
    tc_shifts: list[dict] = field(default_factory=list)
    quality_changes: list[dict] = field(default_factory=list)
    stability_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "new_bubbles": self.new_bubbles,
            "dropped_bubbles": self.dropped_bubbles,
            "tc_shifts": self.tc_shifts,
            "quality_changes": self.quality_changes,
            "stability_summary": self.stability_summary,
        }


class DeltaTracker:
    """Tracks prediction changes over time.

    Saves prediction history and compares new runs with previous ones
    to identify emerging bubbles, dropped signals, and tc stability.
    """

    def __init__(self, history_file: str = "data/predictions_history.json"):
        """
        Args:
            history_file: Path to JSON file storing prediction history
        """
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history = self._load_history()

    def _load_history(self) -> dict:
        """Load prediction history from JSON file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                log.warning("history_load_failed", error=str(e))
                return {}
        return {}

    def compare_and_save(
        self,
        current_predictions: list[dict[str, Any]],
    ) -> DeltaReport:
        """Compare current predictions with last run and save.

        Args:
            current_predictions: List of prediction dicts with keys:
                - ticker: Asset ticker
                - verdict: BUBBLE / POSSIBLE / NO_BUBBLE
                - tc_date_str: Predicted crash date
                - quality_score: 0-1
                - ds_confidence: DS-LPPLS confidence (optional)

        Returns:
            DeltaReport with all changes
        """
        today = datetime.now().strftime("%Y-%m-%d")
        report = DeltaReport(date=today)

        # Get last prediction slice
        last_date = sorted(self.history.keys())[-1] if self.history else None
        last_predictions = {}

        if last_date:
            last_predictions = {
                p["ticker"]: p for p in self.history[last_date].get("predictions", [])
            }

        # Index current predictions
        current_indexed = {p["ticker"]: p for p in current_predictions}

        # Stability counters
        stable_count = 0
        drifting_count = 0
        unstable_count = 0

        for ticker, current in current_indexed.items():
            past = last_predictions.get(ticker)

            if not past:
                # New asset — not a change, just first observation
                continue

            # 1. New bubble signal
            if current["verdict"] == "BUBBLE" and past.get("verdict") != "BUBBLE":
                report.new_bubbles.append(
                    f"🚨 {ticker}: {past.get('verdict', 'N/A')} → BUBBLE "
                    f"(tc: {current.get('tc_date_str', 'N/A')}, "
                    f"Q: {current.get('quality_score', 0):.3f})"
                )

            # 2. Dropped bubble signal
            elif current["verdict"] != "BUBBLE" and past.get("verdict") == "BUBBLE":
                report.dropped_bubbles.append(
                    f"📉 {ticker}: BUBBLE → {current['verdict']} "
                    f"(сигнал сдулся)"
                )

            # 3. tc shift (only for BUBBLE signals)
            elif current["verdict"] == "BUBBLE" and past.get("verdict") == "BUBBLE":
                current_tc = current.get("tc_date_str")
                past_tc = past.get("tc_date_str")

                if current_tc and past_tc and current_tc != past_tc:
                    # Calculate shift in days
                    try:
                        current_dt = datetime.strptime(current_tc, "%Y-%m-%d")
                        past_dt = datetime.strptime(past_tc, "%Y-%m-%d")
                        shift_days = (current_dt - past_dt).days

                        # Stability classification
                        if abs(shift_days) <= 7:
                            stability = "STABLE"
                            stable_count += 1
                            icon = "✅"
                        elif abs(shift_days) <= 21:
                            stability = "DRIFTING"
                            drifting_count += 1
                            icon = "⚠️"
                        else:
                            stability = "UNSTABLE"
                            unstable_count += 1
                            icon = "❌"

                        report.tc_shifts.append({
                            "ticker": ticker,
                            "past_tc": past_tc,
                            "current_tc": current_tc,
                            "shift_days": shift_days,
                            "stability": stability,
                            "icon": icon,
                        })

                    except ValueError:
                        pass

            # 4. Quality score changes
            current_q = current.get("quality_score", 0)
            past_q = past.get("quality_score", 0)

            if abs(current_q - past_q) > 0.1:
                direction = "↑" if current_q > past_q else "↓"
                report.quality_changes.append({
                    "ticker": ticker,
                    "past_quality": round(past_q, 3),
                    "current_quality": round(current_q, 3),
                    "direction": direction,
                })

        # Stability summary
        total_bubble_tc = stable_count + drifting_count + unstable_count
        report.stability_summary = {
            "stable": stable_count,
            "drifting": drifting_count,
            "unstable": unstable_count,
            "total_tc_comparisons": total_bubble_tc,
            "stability_ratio": round(stable_count / total_bubble_tc, 2) if total_bubble_tc > 0 else None,
        }

        # Save to history
        self.history[today] = {
            "predictions": current_predictions,
            "delta_report": report.to_dict(),
        }

        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

        log.info(
            "delta_saved",
            date=today,
            new_bubbles=len(report.new_bubbles),
            dropped_bubbles=len(report.dropped_bubbles),
            tc_shifts=len(report.tc_shifts),
        )

        return report

    def print_delta(self, report: DeltaReport) -> None:
        """Print delta report to console with Rich formatting."""
        console.print()
        console.print(
            Panel(
                f"[bold]Delta Report — {report.date}[/bold]\n"
                f"Comparing with previous run",
                title="📊 DELTA REPORT",
                border_style="cyan",
            )
        )

        # New bubbles
        if report.new_bubbles:
            console.print("\n[bold red]🚨 NEW BUBBLE SIGNALS:[/bold red]")
            for item in report.new_bubbles:
                console.print(f"  {item}")
        else:
            console.print("\n[dim]✅ No new bubble signals[/dim]")

        # Dropped bubbles
        if report.dropped_bubbles:
            console.print("\n[bold green]📉 DROPPED BUBBLE SIGNALS:[/bold green]")
            for item in report.dropped_bubbles:
                console.print(f"  {item}")
        else:
            console.print("\n[dim]✅ No dropped bubble signals[/dim]")

        # tc shifts
        if report.tc_shifts:
            console.print("\n[bold yellow]⏱️ TC DATE SHIFTS:[/bold yellow]")
            for shift in report.tc_shifts:
                console.print(
                    f"  {shift['icon']} {shift['ticker']}: "
                    f"{shift['past_tc']} → {shift['current_tc']} "
                    f"({shift['shift_days']:+d} days) "
                    f"[{shift['stability']}]"
                )
        else:
            console.print("\n[dim]✅ No tc shifts (all stable)[/dim]")

        # Quality changes
        if report.quality_changes:
            console.print("\n[bold]📈 QUALITY SCORE CHANGES:[/bold]")
            for change in report.quality_changes:
                color = "green" if change["direction"] == "↑" else "red"
                console.print(
                    f"  {change['ticker']}: "
                    f"[{color}]{change['direction']}[/] "
                    f"{change['past_quality']:.3f} → {change['current_quality']:.3f}"
                )

        # Stability summary
        if report.stability_summary["total_tc_comparisons"] > 0:
            console.print("\n[bold]🎯 STABILITY SUMMARY:[/bold]")
            s = report.stability_summary
            console.print(
                f"  ✅ Stable: {s['stable']} | "
                f"⚠️ Drifting: {s['drifting']} | "
                f"❌ Unstable: {s['unstable']}"
            )
            if s["stability_ratio"]:
                console.print(f"  Stability ratio: {s['stability_ratio']:.0%}")
                if s["stability_ratio"] >= 0.8:
                    console.print("  [green]🟢 Signals are STABLE — crash risk is high[/green]")
                elif s["stability_ratio"] >= 0.5:
                    console.print("  [yellow]🟡 Some signals drifting — monitor closely[/yellow]")
                else:
                    console.print("  [red]🔴 Many unstable signals — may be noise[/red]")

        # No changes
        if not any([
            report.new_bubbles,
            report.dropped_bubbles,
            report.tc_shifts,
            report.quality_changes,
        ]):
            console.print("\n[bold green]✅ No changes detected. Trends are stable.[/bold green]")

        console.print()

    def get_ticker_history(self, ticker: str) -> list[dict]:
        """Get prediction history for a specific ticker."""
        history = []
        for date, data in sorted(self.history.items()):
            for pred in data.get("predictions", []):
                if pred.get("ticker") == ticker:
                    history.append({"date": date, **pred})
        return history

    def get_stability_trend(self, ticker: str) -> str:
        """Get stability assessment for a ticker's tc over time."""
        history = self.get_ticker_history(ticker)
        bubble_history = [h for h in history if h.get("verdict") == "BUBBLE" and h.get("tc_date_str")]

        if len(bubble_history) < 2:
            return "INSUFFICIENT_DATA"

        # Check tc consistency
        tc_dates = []
        for h in bubble_history:
            try:
                tc_dates.append(datetime.strptime(h["tc_date_str"], "%Y-%m-%d"))
            except ValueError:
                continue

        if len(tc_dates) < 2:
            return "INSUFFICIENT_DATA"

        # Calculate variance
        tc_days = [(d - tc_dates[0]).days for d in tc_dates]
        tc_range = max(tc_days) - min(tc_days)

        if tc_range <= 7:
            return "VERY_STABLE"
        elif tc_range <= 21:
            return "STABLE"
        elif tc_range <= 45:
            return "DRIFTING"
        else:
            return "UNSTABLE"
