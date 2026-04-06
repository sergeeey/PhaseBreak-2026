"""Eval Runner — run evaluation set and check verdict contract compliance.

Usage:
    python -m src.contract.eval_runner              # run full eval set
    python -m src.contract.eval_runner --ticker NVDA # run single case
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Load .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.contract.verdict_contract import compute_final_verdict, VerdictResult, check_contract
from src.lppls.ds_filters import fit_lppls_simple, passes_sornette_filters
from src.hybrid.rag_engine import RAGEngine

log = structlog.get_logger()
console = Console()


def load_eval_set(path: str = "data/eval_set.json") -> list[dict]:
    """Load evaluation set from JSON file."""
    eval_path = Path(__file__).parent.parent.parent / path
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval set not found: {eval_path}")

    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("cases", [])
    anti_bubble = data.get("anti_bubble_cases", [])

    return cases + anti_bubble


def run_lppls_for_case(case: dict) -> dict:
    """Run LPPLS analysis for a single eval case."""
    import yfinance as yf

    ticker = case["ticker"]
    start = case["period_start"]
    end = case["period_end"]

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return {"verdict": "NO_BUBBLE", "quality_score": 0.0, "r_squared": 0.0, "error": "no_data"}

    prices = df["Close"].dropna().values.flatten()
    if len(prices) < 30:
        return {"verdict": "NO_BUBBLE", "quality_score": 0.0, "r_squared": 0.0, "error": "too_few_points"}

    t = np.arange(len(prices), dtype=float)
    result = fit_lppls_simple(t, prices)

    # Add Sornette filter check
    result["sornette_pass"] = passes_sornette_filters(
        tc=result["tc_estimate"],
        m=result["m"],
        omega=result["omega"],
        a=result.get("A", 0),
        b=result["B"],
        c=result.get("C", 0),
        t_start=0,
        t_end=t[-1],
        domain=case.get("domain", "finance"),
    )

    return result


def run_rag_for_case(case: dict, rag: RAGEngine, blind: bool = False) -> dict:
    """Run RAG analysis for a single eval case.
    
    Args:
        case: Eval case dict
        rag: RAG engine
        blind: If True, do NOT pass known_outcome to LLM (fair test)
    """
    ticker = case["ticker"]
    start = case["period_start"]
    end = case["period_end"]

    try:
        context = rag.collect_context(ticker, start, end, max_results=6)
        briefing = rag.analyze_bubble(
            ticker=ticker,
            context=context,
            tc_date=case.get("tc_actual"),
            quality_score=0.5,
            hmm_regime=None,
            price=None,
            known_outcome=None if blind else case.get("label"),
        )
        return briefing.to_dict()
    except Exception as e:
        log.warning("rag_failed", ticker=ticker, error=str(e))
        return {"conclusion": "UNKNOWN", "confidence_in_conclusion": 0.0, "error": str(e)}


def run_eval_set(
    eval_set: list[dict] | None = None,
    run_rag: bool = False,
    blind: bool = True,
    output: str = "data/eval_results.json",
) -> dict:
    """Run full evaluation set and check contract compliance.

    Args:
        eval_set: List of eval cases (loads from file if None)
        run_rag: Whether to run RAG analysis (slower)
        blind: If True (default), do NOT pass known_outcome to LLM — fair test
        output: Path to save results

    Returns:
        Eval results dict
    """
    if eval_set is None:
        eval_set = load_eval_set()

    start = time.time()
    results = []
    rag = RAGEngine() if run_rag else None

    console.print()
    console.print(
        Panel(
            f"[bold]PhaseBreak Eval Set[/bold]\n"
            f"Cases: {len(eval_set)}\n"
            f"RAG: {'ON' if run_rag else 'OFF'}\n"
            f"Mode: {'BLIND (fair test)' if blind else 'VALIDATION (with answers)'}\n"
            f"Speculative: {sum(1 for c in eval_set if c['label'] == 'SPECULATIVE_BUBBLE')}\n"
            f"Fundamental: {sum(1 for c in eval_set if c['label'] == 'FUNDAMENTAL_GROWTH')}\n"
            f"Mixed: {sum(1 for c in eval_set if c['label'] == 'MIXED')}",
            title="Evaluation Setup",
            border_style="cyan",
        )
    )

    for i, case in enumerate(eval_set, 1):
        console.print(f"\n[bold cyan][{i}/{len(eval_set)}] {case['ticker']} ({case['name']})[/bold cyan]")
        console.print(f"  Expected: {case['label']}")

        # LPPLS
        lppls = run_lppls_for_case(case)
        console.print(f"  LPPLS: {lppls.get('verdict', 'N/A')} (Q={lppls.get('quality_score', 0):.3f}, R²={lppls.get('r_squared', 0):.3f})")

        # RAG
        rag_result = None
        if run_rag and rag:
            rag_result = run_rag_for_case(case, rag, blind=blind)
            console.print(f"  RAG: {rag_result.get('conclusion', 'N/A')} (conf={rag_result.get('confidence_in_conclusion', 0):.2f})")

        # Contract
        final = compute_final_verdict(lppls, rag_result)
        violations = check_contract(lppls, rag_result, final)

        # Match logic: LPPLS detects bubble patterns, not fundamental vs speculative
        # SPECULATIVE_BUBBLE expected → LPPLS should say BUBBLE/POSSIBLE
        # FUNDAMENTAL_GROWTH expected → LPPLS should say NO_BUBBLE (no bubble pattern)
        # MIXED expected → LPPLS can say anything
        # NO_BUBBLE (anti-bubble) → LPPLS should say NO_BUBBLE
        expected = case["label"]
        actual = final.verdict
        if expected == "SPECULATIVE_BUBBLE":
            match = actual in ("BUBBLE", "POSSIBLE")
        elif expected == "FUNDAMENTAL_GROWTH":
            match = actual == "NO_BUBBLE"
        elif expected == "NO_BUBBLE":
            match = actual == "NO_BUBBLE"
        else:  # MIXED
            match = True  # Any verdict is acceptable for mixed

        result = {
            "id": case.get("id"),
            "ticker": case["ticker"],
            "name": case["name"],
            "expected": case["label"],
            "actual": final.verdict,
            "match": match,
            "reason": final.reason,
            "confidence": final.confidence,
            "contract_violations": violations,
            "lppls": {
                "verdict": lppls.get("verdict"),
                "quality_score": lppls.get("quality_score"),
                "r_squared": lppls.get("r_squared"),
                "sornette_pass": lppls.get("sornette_pass"),
            },
        }
        if rag_result:
            result["rag"] = {
                "conclusion": rag_result.get("conclusion"),
                "confidence": rag_result.get("confidence_in_conclusion"),
            }

        results.append(result)

        # Display
        match_icon = "✅" if match else "❌"
        violation_icon = "⚠️" if violations else ""
        console.print(f"  Final: {final.verdict} {match_icon} {violation_icon}")
        if violations:
            for v in violations:
                console.print(f"    ⚠️ Contract violation: {v}")

    # Summary
    elapsed = time.time() - start
    matches = sum(1 for r in results if r["match"])
    total = len(results)
    accuracy = matches / total if total > 0 else 0

    total_violations = sum(len(r["contract_violations"]) for r in results)

    console.print()
    console.print("=" * 80)
    console.print("[bold]EVAL SET RESULTS[/bold]")
    console.print("=" * 80)

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Ticker", style="white")
    table.add_column("Expected", style="white")
    table.add_column("Actual", style="white")
    table.add_column("Match", style="white", justify="center")
    table.add_column("Violations", style="white", justify="right")
    table.add_column("Confidence", style="white", justify="right")

    for r in results:
        match_icon = "✅" if r["match"] else "❌"
        table.add_row(
            r["ticker"],
            r["expected"],
            r["actual"],
            match_icon,
            str(len(r["contract_violations"])),
            f"{r['confidence']:.3f}",
        )

    console.print(table)

    console.print(f"\n[bold]Accuracy: {matches}/{total} ({accuracy:.0%})[/bold]")
    console.print(f"[bold]Contract violations: {total_violations}[/bold]")
    console.print(f"[bold]Time: {elapsed:.1f}s[/bold]")

    # Save results
    if output:
        output_data = {
            "eval_date": time.strftime("%Y-%m-%d %H:%M"),
            "eval_set_version": "1.0",
            "total_cases": total,
            "matches": matches,
            "accuracy": round(accuracy, 4),
            "contract_violations": total_violations,
            "elapsed_seconds": round(elapsed, 1),
            "results": results,
        }
        output_path = Path(__file__).parent.parent.parent / output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        console.print(f"\n[dim]Results saved to: {output_path}[/dim]")

    return output_data if output else {"results": results, "accuracy": accuracy}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PhaseBreak Eval Set Runner")
    parser.add_argument("--ticker", type=str, default=None, help="Run single case")
    parser.add_argument("--no-rag", action="store_true", help="Skip RAG analysis")
    parser.add_argument("--no-blind", action="store_true", help="Pass known_outcome to LLM (validation mode, not fair test)")
    parser.add_argument("--output", type=str, default="data/eval_results.json")
    args = parser.parse_args()

    eval_set = load_eval_set()

    if args.ticker:
        eval_set = [c for c in eval_set if c["ticker"] == args.ticker]
        if not eval_set:
            console.print(f"[red]No eval case for ticker: {args.ticker}[/red]")
            return

    run_eval_set(
        eval_set=eval_set,
        run_rag=not args.no_rag,
        blind=not args.no_blind,
        output=args.output,
    )


if __name__ == "__main__":
    main()
