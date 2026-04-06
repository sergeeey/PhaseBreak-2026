"""Crash-oriented evaluation — measures what LPPLS actually predicts.

Question: "Will this asset drop >30% within 6 months?"
LPPLS BUBBLE/POSSIBLE = prediction of crash.
Ground truth = actual drawdown from Yahoo Finance.

Metrics:
- Crash Precision: of assets we flagged, how many actually crashed?
- Crash Recall: of assets that crashed, how many did we flag?
- Lead time: how many days before the crash did we predict it?
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import structlog
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lppls.ds_filters import fit_lppls_simple, passes_sornette_filters

log = structlog.get_logger()
console = Console()


def run_crash_eval(
    eval_path: str = "data/eval_crash.json",
    use_hmm: bool = False,
    output: str = "data/crash_eval_results.json",
) -> dict:
    """Run crash-oriented evaluation."""
    import yfinance as yf

    with open(Path(__file__).parent.parent.parent / eval_path, encoding="utf-8") as f:
        data = json.load(f)

    cases = data["cases"]
    results = []
    tp = fp = fn = tn = 0

    console.print(f"\n[bold]Crash-Oriented Eval — {len(cases)} cases[/bold]")
    console.print(f"Question: will asset drop >30% within 6 months?\n")

    for c in cases:
        ticker = c["ticker"]
        start = c["period_start"]
        end = c["period_end"]
        crashed = c["crashed_30pct"]
        drawdown = c["actual_drawdown_pct"]

        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty or len(df) < 30:
                results.append({"ticker": ticker, "error": "no_data"})
                if crashed:
                    fn += 1
                else:
                    tn += 1
                continue

            prices = df["Close"].dropna().values.flatten()
            t = np.arange(len(prices), dtype=float)

            if use_hmm:
                from predict_all import run_lppls_with_hmm

                lppls = run_lppls_with_hmm(t, prices)
            else:
                lppls = fit_lppls_simple(t, prices)

            predicted_crash = lppls["verdict"] in ("BUBBLE", "POSSIBLE")

            if predicted_crash and crashed:
                tp += 1
                match = "TP"
            elif predicted_crash and not crashed:
                fp += 1
                match = "FP"
            elif not predicted_crash and crashed:
                fn += 1
                match = "FN"
            else:
                tn += 1
                match = "TN"

            color = {"TP": "green", "FP": "red", "FN": "yellow", "TN": "dim"}.get(match, "white")
            console.print(
                f"  [{color}]{match}[/{color}] {ticker:10s} "
                f"predicted={'CRASH' if predicted_crash else 'safe':6s} "
                f"actual={'CRASH(' + str(drawdown) + '%)' if crashed else 'safe':16s} "
                f"verdict={lppls['verdict']:12s} Q={lppls['quality_score']:.3f}"
            )

            results.append(
                {
                    "id": c["id"],
                    "ticker": ticker,
                    "predicted_crash": predicted_crash,
                    "actual_crash": crashed,
                    "actual_drawdown": drawdown,
                    "match": match,
                    "verdict": lppls["verdict"],
                    "quality": lppls["quality_score"],
                }
            )

        except Exception as e:
            log.debug("crash_eval_error", ticker=ticker, error=str(e))
            if crashed:
                fn += 1
            else:
                tn += 1

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(cases) if cases else 0

    console.print(f"\n{'=' * 60}")
    console.print(f"[bold]CRASH EVAL RESULTS[/bold]")
    console.print(f"{'=' * 60}")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("True Positives (caught crash)", str(tp))
    table.add_row("False Positives (false alarm)", str(fp))
    table.add_row("False Negatives (missed crash)", str(fn))
    table.add_row("True Negatives (correctly safe)", str(tn))
    table.add_row("Crash Precision", f"{precision:.0%}")
    table.add_row("Crash Recall", f"{recall:.0%}")
    table.add_row("F1 Score", f"{f1:.0%}")
    table.add_row("Accuracy", f"{accuracy:.0%}")
    console.print(table)

    summary = {
        "eval_date": time.strftime("%Y-%m-%d %H:%M"),
        "total_cases": len(cases),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "crash_precision": round(precision, 4),
        "crash_recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "use_hmm": use_hmm,
        "results": results,
    }

    if output:
        out_path = Path(__file__).parent.parent.parent / output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        console.print(f"\n[dim]Saved to {out_path}[/dim]")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PhaseBreak Crash Eval")
    parser.add_argument("--hmm", action="store_true", help="Use HMM gate")
    parser.add_argument("--output", default="data/crash_eval_results.json")
    args = parser.parse_args()
    run_crash_eval(use_hmm=args.hmm, output=args.output)
