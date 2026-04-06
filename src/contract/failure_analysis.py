"""Failure Analysis — understand WHY the model misses crashes.

For each missed crash (FN), checks what each module showed
before the crash. Identifies blind spots and patterns.
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

from src.lppls.ds_filters import fit_lppls_simple
from src.lppls.hmm_gate import hmm_prescreen
from src.lppls.hurst_signal import compute_hurst

log = structlog.get_logger()
console = Console()


def analyze_failures(eval_path: str = "data/eval_crash.json") -> dict:
    """For each crash we missed, analyze what signals showed."""
    import yfinance as yf

    with open(Path(__file__).parent.parent.parent / eval_path, encoding="utf-8") as f:
        data = json.load(f)

    failures = []
    successes = []

    console.print("\n[bold]FAILURE ANALYSIS — Why did we miss these crashes?[/bold]\n")

    for c in data["cases"]:
        if not c["crashed_30pct"]:
            continue

        ticker = c["ticker"]
        start = c["period_start"]
        end = c["period_end"]
        drawdown = c["actual_drawdown_pct"]

        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty or len(df) < 30:
                continue

            prices = df["Close"].dropna().values.flatten()
            t = np.arange(len(prices), dtype=float)

            lppls = fit_lppls_simple(t, prices)
            hmm = hmm_prescreen(prices)
            H = compute_hurst(prices)

            predicted = lppls["verdict"] in ("BUBBLE", "POSSIBLE")
            entry = {
                "ticker": ticker,
                "name": c["name"],
                "drawdown": drawdown,
                "predicted": predicted,
                "verdict": lppls["verdict"],
                "quality": lppls["quality_score"],
                "r_squared": lppls["r_squared"],
                "m": lppls["m"],
                "omega": lppls["omega"],
                "oscillations": lppls["oscillations"],
                "damping": lppls["damping"],
                "hmm_regime": hmm["regime"],
                "hmm_bubble_prob": hmm["bubble_prob"],
                "hurst": H,
                "B": lppls["B"],
            }

            if predicted:
                successes.append(entry)
            else:
                failures.append(entry)
                # Diagnose WHY it failed
                reasons = []
                if lppls["m"] <= 0.1:
                    reasons.append("m=0.1 (optimizer floor — too vertical)")
                if lppls["oscillations"] < 2.0:
                    reasons.append(f"osc={lppls['oscillations']:.1f} < 2.0 (too few oscillations)")
                if lppls["r_squared"] < 0.75:
                    reasons.append(f"R²={lppls['r_squared']:.2f} < 0.75 (poor fit)")
                if lppls["quality_score"] < 0.75:
                    reasons.append(f"Q={lppls['quality_score']:.2f} < 0.75")
                if lppls["B"] >= 0:
                    reasons.append("B >= 0 (no super-exponential growth)")
                entry["failure_reasons"] = reasons

        except Exception as e:
            log.debug("failure_analysis_error", ticker=ticker, error=str(e))

    # Print results
    table = Table(
        title="Missed Crashes (False Negatives)", show_header=True, header_style="bold red"
    )
    table.add_column("Ticker", width=10)
    table.add_column("Crash", justify="right", width=7)
    table.add_column("Q", justify="right", width=5)
    table.add_column("m", justify="right", width=6)
    table.add_column("osc", justify="right", width=5)
    table.add_column("R²", justify="right", width=6)
    table.add_column("HMM", width=8)
    table.add_column("H", justify="right", width=6)
    table.add_column("Why missed", width=40)

    for f in failures:
        table.add_row(
            f["ticker"],
            f"{f['drawdown']}%",
            f"{f['quality']:.2f}",
            f"{f['m']:.3f}",
            f"{f['oscillations']:.1f}",
            f"{f['r_squared']:.2f}",
            f["hmm_regime"],
            f"{f['hurst']:.3f}",
            "; ".join(f.get("failure_reasons", []))[:40],
        )
    console.print(table)

    # Pattern analysis
    console.print("\n[bold]PATTERN ANALYSIS[/bold]")
    if failures:
        m_floor = sum(1 for f in failures if f["m"] <= 0.1)
        low_osc = sum(1 for f in failures if f["oscillations"] < 2.0)
        low_r2 = sum(1 for f in failures if f["r_squared"] < 0.75)
        low_q = sum(1 for f in failures if f["quality"] < 0.75)

        console.print(f"  Total missed crashes: {len(failures)}")
        console.print(
            f"  m=0.1 (too vertical): {m_floor}/{len(failures)} ({m_floor / len(failures) * 100:.0f}%)"
        )
        console.print(
            f"  osc < 2.0 (too few): {low_osc}/{len(failures)} ({low_osc / len(failures) * 100:.0f}%)"
        )
        console.print(
            f"  R² < 0.75 (poor fit): {low_r2}/{len(failures)} ({low_r2 / len(failures) * 100:.0f}%)"
        )
        console.print(f"  Q < 0.75: {low_q}/{len(failures)} ({low_q / len(failures) * 100:.0f}%)")

    console.print(f"\n  Caught crashes: {len(successes)}")
    for s in successes:
        console.print(
            f"    {s['ticker']:10s} dd={s['drawdown']}% Q={s['quality']:.2f} m={s['m']:.3f} osc={s['oscillations']:.1f}"
        )

    result = {
        "analysis_date": time.strftime("%Y-%m-%d %H:%M"),
        "total_crashes": len(failures) + len(successes),
        "caught": len(successes),
        "missed": len(failures),
        "recall": len(successes) / (len(failures) + len(successes))
        if (failures or successes)
        else 0,
        "failures": failures,
        "successes": successes,
    }

    out_path = Path(__file__).parent.parent.parent / "data" / "failure_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f_out:
        json.dump(result, f_out, indent=2, ensure_ascii=False, default=str)
    console.print(f"\n[dim]Saved to {out_path}[/dim]")

    return result


if __name__ == "__main__":
    analyze_failures()
