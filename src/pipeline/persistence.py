"""Signal Persistence Filter — confirms signals only after N consecutive scans.

Problem: LPPLS signals flip-flop between scans (5/5 BUBBLE signals from March 29
disappeared by April 5). Users lose trust when alerts come and go.

Solution: track signal history per ticker. Only CONFIRMED after 3+ consecutive
scans with BUBBLE/POSSIBLE. TENTATIVE for 1-2 scans. DROPPED when signal disappears.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

HISTORY_PATH = Path(__file__).parent.parent.parent / "data" / "signal_history.json"
CONFIRM_THRESHOLD = 3  # consecutive scans needed for CONFIRMED


def load_history() -> dict[str, list[dict]]:
    """Load signal history. Returns {ticker: [{date, verdict, quality}, ...]}."""
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_history(history: dict[str, list[dict]]) -> None:
    """Save signal history. Keeps last 30 entries per ticker."""
    trimmed = {}
    for ticker, entries in history.items():
        trimmed[ticker] = entries[-30:]

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, indent=2, ensure_ascii=False)


def update_history(
    scan_results: list[dict[str, Any]],
    history: dict[str, list[dict]] | None = None,
) -> dict[str, list[dict]]:
    """Add current scan results to history."""
    if history is None:
        history = load_history()

    scan_date = time.strftime("%Y-%m-%d %H:%M")

    for result in scan_results:
        ticker = result.get("ticker", "")
        if not ticker:
            continue

        entry = {
            "date": scan_date,
            "verdict": result.get("verdict", "NO_BUBBLE"),
            "quality": result.get("quality_score", 0.0),
        }

        if ticker not in history:
            history[ticker] = []
        history[ticker].append(entry)

    save_history(history)
    return history


def get_signal_status(
    ticker: str,
    history: dict[str, list[dict]] | None = None,
) -> dict[str, Any]:
    """Get persistence status for a ticker.

    Returns:
        status: CONFIRMED | TENTATIVE | DROPPED | NO_SIGNAL
        consecutive_count: number of consecutive BUBBLE/POSSIBLE scans
        first_seen: date of first signal in current streak
        history_length: total entries for this ticker
    """
    if history is None:
        history = load_history()

    entries = history.get(ticker, [])
    if not entries:
        return {
            "status": "NO_SIGNAL",
            "consecutive_count": 0,
            "first_seen": None,
            "history_length": 0,
        }

    # Count consecutive BUBBLE/POSSIBLE from the end
    consecutive = 0
    for entry in reversed(entries):
        if entry["verdict"] in ("BUBBLE", "POSSIBLE"):
            consecutive += 1
        else:
            break

    # Determine status
    if consecutive == 0:
        # Check if there WAS a signal before (DROPPED)
        had_signal = (
            any(e["verdict"] in ("BUBBLE", "POSSIBLE") for e in entries[:-1])
            if len(entries) > 1
            else False
        )
        status = "DROPPED" if had_signal else "NO_SIGNAL"
    elif consecutive >= CONFIRM_THRESHOLD:
        status = "CONFIRMED"
    else:
        status = "TENTATIVE"

    # First seen in current streak
    first_seen = None
    if consecutive > 0:
        streak_start = len(entries) - consecutive
        first_seen = entries[streak_start]["date"]

    return {
        "status": status,
        "consecutive_count": consecutive,
        "first_seen": first_seen,
        "history_length": len(entries),
    }


def classify_all_signals(
    scan_results: list[dict[str, Any]],
    auto_save: bool = True,
) -> list[dict[str, Any]]:
    """Update history and classify all signals from a scan.

    Returns enriched results with persistence status.
    """
    history = load_history()
    history = update_history(scan_results, history)

    enriched = []
    confirmed = []
    tentative = []
    dropped = []

    for result in scan_results:
        ticker = result.get("ticker", "")
        status = get_signal_status(ticker, history)

        enriched_result = {**result, **status}
        enriched.append(enriched_result)

        if status["status"] == "CONFIRMED":
            confirmed.append(ticker)
        elif status["status"] == "TENTATIVE":
            tentative.append(ticker)
        elif status["status"] == "DROPPED":
            dropped.append(ticker)

    log.info(
        "persistence_classified",
        total=len(scan_results),
        confirmed=len(confirmed),
        tentative=len(tentative),
        dropped=len(dropped),
    )

    if confirmed:
        log.info("confirmed_signals", tickers=confirmed)
    if dropped:
        log.info("dropped_signals", tickers=dropped)

    return enriched
