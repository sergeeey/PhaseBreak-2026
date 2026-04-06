"""PhaseBreak Deep Analysis — uses ALL system capabilities on every asset.

Unlike predict_all.py (simplified 3-signal pipeline), this script runs
the FULL academic pipeline + hybrid enrichment for maximum accuracy.

Pipeline per asset:
1. run_full_pipeline() — 15-module academic LPPLS (optimizer, confidence, scoring, uncertainty)
2. fit_lppls_simple() — standalone strict fitter (Sornette filters)
3. DS-LPPLS multi-window confidence
4. HMM regime detection
5. Hurst exponent (R/S analysis)
6. MFDFA multifractal width (if available)
7. Critical Slowing Down (variance + AC1 trends)
8. RAG analysis (RSS + LLM) — only for signals
9. Verdict contract (5 rules)
10. Sector context
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from src.lppls.ds_filters import fit_lppls_simple, fit_lppls_with_ds, passes_sornette_filters
from src.lppls.hmm_gate import hmm_prescreen
from src.lppls.hurst_signal import compute_hurst, hurst_verdict, hurst_supports_bubble
from src.contract.verdict_contract import compute_final_verdict, check_contract

log = structlog.get_logger()
console = Console()

WATCHLIST = {
    "Finance": [
        "NVDA",
        "AAPL",
        "MSFT",
        "GOOGL",
        "META",
        "AMZN",
        "TSLA",
        "ARM",
        "AVGO",
        "PLTR",
        "MSTR",
    ],
    "Indices": ["^GSPC", "^IXIC", "^DJI"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD"],
    "Commodities": ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "ZW=F", "PL=F"],
    "Sector ETFs": ["XLF", "XLE", "XLK", "XLV", "GDX", "SLV", "DBA", "TLT", "ARKK"],
}


def load_prices(ticker: str, period: str = "2y"):
    """Load price data from yfinance."""
    import yfinance as yf

    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    prices = df["Close"].dropna().values.flatten()
    if len(prices) < 60:
        raise ValueError(f"Too few points: {len(prices)}")
    dates = df.index[: len(prices)]
    return np.arange(len(prices), dtype=float), prices, dates


def run_academic_pipeline(ticker: str, t, prices):
    """Run the full 15-module academic pipeline."""
    try:
        from src.pipeline.stages import run_full_pipeline

        # Need start/end dates for yfinance re-download inside pipeline
        # Pipeline downloads its own data, so we pass ticker + date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=len(prices) + 30)).strftime("%Y-%m-%d")
        result = run_full_pipeline(ticker, start_date, end_date)
        return {
            "verdict": result.final_verdict,
            "confidence": result.final_confidence,
            "path": result.path,
            "fit_m": result.fit.m if result.fit and hasattr(result.fit, "m") else None,
            "fit_omega": result.fit.omega if result.fit and hasattr(result.fit, "omega") else None,
            "fit_quality": result.fit.quality
            if result.fit and hasattr(result.fit, "quality")
            else None,
            "screening_regime": result.screening.regime if result.screening else None,
            "available": True,
        }
    except Exception as e:
        return {"verdict": "UNAVAILABLE", "error": str(e)[:80], "available": False}


def run_ds_lppls(t, prices, domain="finance"):
    """Run DS-LPPLS multi-window confidence."""
    try:
        result = fit_lppls_with_ds(t, prices, domain=domain)
        return {
            "ds_confidence": result.get("ds_confidence", 0),
            "ds_verdict": result.get("ds_verdict", "NO_BUBBLE"),
            "valid_windows": result.get("ds_valid_windows", 0),
            "total_windows": result.get("ds_total_windows", 0),
            "ds_median_tc": result.get("ds_median_tc"),
            "ds_tc_std": result.get("ds_tc_std"),
            "sornette_pass": result.get("sornette_filters_pass", False),
        }
    except Exception as e:
        return {"ds_confidence": 0, "ds_verdict": "ERROR", "error": str(e)[:50]}


def run_mfdfa(prices):
    """Run MFDFA multifractal analysis."""
    try:
        from MFDFA import MFDFA as mfdfa_func

        lag = np.unique(np.logspace(0.7, 3, 50).astype(int))
        lag = lag[lag < len(prices) // 4]
        if len(lag) < 5:
            return {"delta_alpha": None, "available": False}
        _, fluct = mfdfa_func(prices, lag=lag, q=[-5, -3, -1, 0, 1, 3, 5], order=1)
        # Compute multifractal width
        from numpy.polynomial import polynomial as P

        alphas = []
        for i in range(fluct.shape[1]):
            valid = fluct[:, i] > 0
            if valid.sum() > 3:
                c = P.polyfit(np.log(lag[valid]), np.log(fluct[valid, i]), 1)
                alphas.append(c[1])
        if len(alphas) >= 2:
            delta_alpha = max(alphas) - min(alphas)
        else:
            delta_alpha = 0
        return {"delta_alpha": round(float(delta_alpha), 4), "available": True}
    except Exception as e:
        return {"delta_alpha": None, "available": False, "error": str(e)[:50]}


def run_ews(prices):
    """Critical Slowing Down — variance and autocorrelation trends."""
    try:
        returns = np.diff(np.log(prices))
        window = min(50, len(returns) // 4)
        if window < 10:
            return {"var_trend": None, "ac1_trend": None}

        # Rolling variance
        roll_var = [np.var(returns[max(0, i - window) : i + 1]) for i in range(len(returns))]
        # Rolling AC1
        roll_ac1 = []
        for i in range(len(returns)):
            chunk = returns[max(0, i - window) : i + 1]
            if len(chunk) > 5:
                ac1 = np.corrcoef(chunk[:-1], chunk[1:])[0, 1] if len(chunk) > 2 else 0
                roll_ac1.append(ac1)
            else:
                roll_ac1.append(0)

        # Trend: linear regression on last half
        half = len(roll_var) // 2
        x = np.arange(half, dtype=float)

        var_slope = np.polyfit(x, roll_var[-half:], 1)[0] if half > 5 else 0
        ac1_slope = np.polyfit(x, roll_ac1[-half:], 1)[0] if half > 5 else 0

        return {
            "var_trend": "INCREASING"
            if var_slope > 1e-6
            else ("DECREASING" if var_slope < -1e-6 else "STABLE"),
            "ac1_trend": "INCREASING"
            if ac1_slope > 1e-4
            else ("DECREASING" if ac1_slope < -1e-4 else "STABLE"),
            "var_slope": round(float(var_slope), 8),
            "ac1_slope": round(float(ac1_slope), 6),
        }
    except Exception:
        return {"var_trend": None, "ac1_trend": None}


def run_rag_analysis(ticker: str, price: float):
    """RAG analysis — RSS + LLM."""
    try:
        from src.hybrid.rag_engine import RAGEngine

        rag = RAGEngine()
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        ctx = rag.collect_context(ticker, start, end, max_results=8)
        briefing = rag.analyze_bubble(
            ticker=ticker,
            context=ctx,
            quality_score=0.8,
            hmm_regime=None,
            price=price,
        )
        return {
            "conclusion": briefing.conclusion,
            "confidence": briefing.confidence_in_conclusion,
            "key_evidence": briefing.key_evidence[:150],
            "fundamental": briefing.fundamental_factors[:2],
            "speculative": briefing.speculative_factors[:2],
            "recommendation": briefing.recommendation,
            "sources": len(ctx.search_results),
            "available": True,
        }
    except Exception as e:
        return {"conclusion": "UNAVAILABLE", "error": str(e)[:80], "available": False}


def deep_analyze(ticker: str, domain: str, run_rag: bool = False) -> dict:
    """Run ALL analysis modules on a single ticker."""
    result = {"ticker": ticker, "domain": domain, "timestamp": time.strftime("%Y-%m-%d %H:%M")}

    # Load data
    try:
        t, prices, dates = load_prices(ticker)
        result["price"] = round(float(prices[-1]), 2)
        result["data_points"] = len(prices)
    except Exception as e:
        result["error"] = str(e)[:80]
        return result

    # 1. Simple LPPLS (strict filters)
    simple = fit_lppls_simple(t, prices)
    result["lppls_simple"] = {
        "verdict": simple["verdict"],
        "quality": simple["quality_score"],
        "r_squared": simple["r_squared"],
        "m": simple["m"],
        "omega": simple["omega"],
        "damping": simple["damping"],
        "oscillations": simple["oscillations"],
        "tc_date": simple.get("tc_date_str"),
        "B": simple["B"],
    }

    # 2. Academic pipeline (15 modules)
    result["academic"] = run_academic_pipeline(ticker, t, prices)

    # 3. DS-LPPLS multi-window
    result["ds_lppls"] = run_ds_lppls(
        t, prices, domain="commodities" if domain == "Commodities" else "finance"
    )

    # 4. HMM regime
    hmm = hmm_prescreen(prices)
    result["hmm"] = hmm

    # 5. Hurst
    H = compute_hurst(prices)
    H_3m = compute_hurst(prices[-63:]) if len(prices) >= 63 else H
    result["hurst"] = {
        "H_2y": H,
        "H_3m": H_3m,
        "verdict": hurst_verdict(H),
        "supports_bubble": hurst_supports_bubble(H),
    }

    # 6. MFDFA
    result["mfdfa"] = run_mfdfa(prices)

    # 7. Critical Slowing Down
    result["ews"] = run_ews(prices)

    # 8. Sornette filters
    result["sornette_pass"] = passes_sornette_filters(
        tc=simple["tc_estimate"],
        m=simple["m"],
        omega=simple["omega"],
        a=simple.get("A", 0),
        b=simple["B"],
        c=simple.get("C", 0),
        t_start=0,
        t_end=t[-1],
        domain="commodities" if domain == "Commodities" else "finance",
    )

    # 9. Verdict contract
    rag_result = None
    if run_rag and simple["verdict"] in ("BUBBLE", "POSSIBLE"):
        rag = run_rag_analysis(ticker, result["price"])
        result["rag"] = rag
        if rag.get("available"):
            rag_result = {
                "conclusion": rag["conclusion"],
                "confidence_in_conclusion": rag["confidence"],
            }

    final = compute_final_verdict(
        {
            **simple,
            "ds_verdict": result["ds_lppls"]["ds_verdict"],
            "ds_confidence": result["ds_lppls"]["ds_confidence"],
            "sornette_pass": result["sornette_pass"],
        },
        rag_result,
    )
    result["final_verdict"] = final.verdict
    result["final_reason"] = final.reason
    result["contract_violations"] = final.contract_violations

    # 10. Signal strength (composite)
    signals_yes = 0
    signals_total = 0
    if simple["verdict"] in ("BUBBLE", "POSSIBLE"):
        signals_yes += 1
    signals_total += 1
    if hmm["regime"] == "BUBBLE":
        signals_yes += 1
    signals_total += 1
    if H > 0.65:
        signals_yes += 1
    signals_total += 1
    if result["ds_lppls"]["ds_verdict"] in ("BUBBLE", "POSSIBLE"):
        signals_yes += 1
    signals_total += 1
    if result["ews"]["var_trend"] == "INCREASING":
        signals_yes += 1
    signals_total += 1

    result["signal_strength"] = f"{signals_yes}/{signals_total}"
    result["signal_score"] = round(signals_yes / signals_total, 2) if signals_total > 0 else 0

    return result


def print_card(r: dict) -> None:
    """Print rich analysis card for one asset."""
    ticker = r["ticker"]
    price = r.get("price", 0)
    verdict = r.get("final_verdict", "ERROR")
    color = {"BUBBLE": "red", "POSSIBLE": "yellow", "NO_BUBBLE": "green"}.get(verdict, "white")

    simple = r.get("lppls_simple", {})
    acad = r.get("academic", {})
    ds = r.get("ds_lppls", {})
    hmm = r.get("hmm", {})
    hurst = r.get("hurst", {})
    mfdfa = r.get("mfdfa", {})
    ews = r.get("ews", {})
    rag = r.get("rag", {})

    lines = [
        f"[bold {color}]{verdict}[/bold {color}] — Signal strength: {r.get('signal_strength', '?')}",
        "",
        f"LPPLS simple:  {simple.get('verdict', '?'):12s} Q={simple.get('quality', 0):.2f} R2={simple.get('r_squared', 0):.3f} m={simple.get('m', 0):.3f} w={simple.get('omega', 0):.3f} osc={simple.get('oscillations', 0):.2f} tc={simple.get('tc_date', '')}",
        f"Academic pipe:  {acad.get('verdict', '?'):12s} conf={acad.get('confidence', '?')} path={acad.get('path', '')}",
        f"DS-LPPLS:      {ds.get('ds_verdict', '?'):12s} conf={ds.get('ds_confidence', 0):.3f} windows={ds.get('valid_windows', 0)}/{ds.get('total_windows', 0)} sornette={r.get('sornette_pass', '')}",
        f"HMM:           {hmm.get('regime', '?'):12s} bubble_prob={hmm.get('bubble_prob', 0):.2f}",
        f"Hurst:         H(2y)={hurst.get('H_2y', 0):.3f} H(3m)={hurst.get('H_3m', 0):.3f} → {hurst.get('verdict', '?')} supports_bubble={hurst.get('supports_bubble', '')}",
    ]

    if mfdfa.get("available"):
        lines.append(f"MFDFA:         da={mfdfa.get('delta_alpha', '?')}")
    if ews.get("var_trend"):
        lines.append(f"EWS:           variance={ews['var_trend']} AC1={ews['ac1_trend']}")
    if rag.get("available"):
        lines.append(
            f"RAG:           {rag['conclusion']} conf={rag['confidence']:.2f} → {rag['recommendation']}"
        )
        lines.append(f"               {rag['key_evidence'][:100]}")
    if r.get("contract_violations"):
        lines.append(f"Violations:    {r['contract_violations']}")

    console.print(
        Panel(
            "\n".join(lines),
            title=f"{ticker} — ${price:,.2f} ({r['domain']})",
            border_style=color,
        )
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PhaseBreak Deep Analysis")
    parser.add_argument(
        "--rag", action="store_true", help="Run RAG on signals (requires OpenAI key)"
    )
    parser.add_argument("--ticker", type=str, help="Analyze single ticker")
    parser.add_argument("--output", default="data/DEEP_ANALYSIS_latest.json")
    args = parser.parse_args()

    start = time.time()
    all_results = []

    console.print(
        Panel(
            f"[bold]PhaseBreak Deep Analysis[/bold]\n"
            f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n"
            f"Pipeline: LPPLS + Academic + DS-LPPLS + HMM + Hurst + MFDFA + EWS + Verdict Contract"
            + (" + RAG" if args.rag else ""),
            border_style="cyan",
        )
    )

    if args.ticker:
        watchlist = {"Custom": [args.ticker]}
    else:
        watchlist = WATCHLIST

    for domain, tickers in watchlist.items():
        console.print(f"\n[bold cyan]=== {domain} ({len(tickers)} assets) ===[/bold cyan]")
        for ticker in tickers:
            r = deep_analyze(ticker, domain, run_rag=args.rag)
            if "error" in r:
                console.print(f"  [dim]{ticker}: {r['error']}[/dim]")
                continue
            all_results.append(r)
            print_card(r)

    # Summary
    elapsed = time.time() - start
    signals = [r for r in all_results if r.get("final_verdict") in ("BUBBLE", "POSSIBLE")]
    hmm_bubbles = [r for r in all_results if r.get("hmm", {}).get("regime") == "BUBBLE"]

    console.print(f"\n[bold]{'=' * 70}[/bold]")
    console.print(f"[bold]SUMMARY — {len(all_results)} assets in {elapsed:.0f}s[/bold]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Ticker")
    table.add_column("Verdict", justify="center")
    table.add_column("Strength")
    table.add_column("Q")
    table.add_column("HMM")
    table.add_column("Hurst")
    table.add_column("EWS var")
    table.add_column("tc")
    table.add_column("Price", justify="right")

    for r in sorted(all_results, key=lambda x: x.get("signal_score", 0), reverse=True):
        v = r.get("final_verdict", "?")
        color = {"BUBBLE": "red", "POSSIBLE": "yellow"}.get(v, "dim")
        table.add_row(
            f"[{color}]{r['ticker']}[/{color}]",
            f"[{color}]{v}[/{color}]",
            r.get("signal_strength", ""),
            f"{r.get('lppls_simple', {}).get('quality', 0):.2f}",
            r.get("hmm", {}).get("regime", ""),
            f"{r.get('hurst', {}).get('H_2y', 0):.3f}",
            r.get("ews", {}).get("var_trend", ""),
            r.get("lppls_simple", {}).get("tc_date", "") if v != "NO_BUBBLE" else "",
            f"${r.get('price', 0):,.2f}",
        )
    console.print(table)

    if signals:
        console.print(f"\n[bold red]ACTIVE SIGNALS ({len(signals)}):[/bold red]")
        for s in signals:
            console.print(
                f"  {s['ticker']:10s} {s['final_verdict']:12s} strength={s['signal_strength']} tc={s.get('lppls_simple', {}).get('tc_date', '')}"
            )

    if hmm_bubbles:
        console.print(
            f"\n[bold yellow]HMM BUBBLE REGIME ({len(hmm_bubbles)}, LPPLS may not confirm):[/bold yellow]"
        )
        for h in hmm_bubbles:
            if h.get("final_verdict") not in ("BUBBLE", "POSSIBLE"):
                console.print(
                    f"  {h['ticker']:10s} HMM=BUBBLE prob={h['hmm']['bubble_prob']:.2f} H={h['hurst']['H_2y']:.3f}"
                )

    # Save
    output = {
        "scan_date": time.strftime("%Y-%m-%d %H:%M"),
        "model": "PhaseBreak Deep Analysis (all modules)",
        "elapsed_seconds": round(elapsed, 1),
        "total_scanned": len(all_results),
        "signals": len(signals),
        "results": all_results,
    }
    out_path = Path(__file__).parent / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    console.print(f"\n[dim]Saved to {out_path}[/dim]")


if __name__ == "__main__":
    main()
