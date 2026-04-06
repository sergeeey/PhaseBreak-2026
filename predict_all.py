"""PhaseBreak Hybrid 2.0 — ALL PREDICTIONS.

Comprehensive prediction across ALL domains:
- Finance (stocks, indices)
- Commodities (oil, gold, silver, etc.)
- Crypto (BTC, ETH, SOL, etc.)
- Housing (10 metro markets)

Usage:
    python predict_all.py                    # run all predictions
    python predict_all.py --finance-only     # just finance
    python predict_all.py --housing-only     # just housing
    python predict_all.py --no-rag           # skip RAG analysis
    python predict_all.py --output results.json
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

# Load .env
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from src.hybrid.housing import HousingDataLoader, HousingEpisode, HOT_MARKETS
from src.hybrid.rag_engine import RAGEngine, ContextBundle, HybridBriefing
from src.hybrid.orchestrator import HybridOrchestrator
from src.lppls.ds_filters import fit_lppls_with_ds, fit_lppls_simple, passes_sornette_filters
from src.lppls.hmm_gate import hmm_prescreen
from src.lppls.hurst_signal import compute_hurst, hurst_supports_bubble
from src.pipeline.delta_tracker import DeltaTracker
from src.contract.verdict_contract import compute_final_verdict, check_contract
from src.contract.domain_status import get_domain_status, can_claim_validated, get_status_label

log = structlog.get_logger()
console = Console()

# ─── Asset Universe ───────────────────────────────────────────────────

FINANCE_ASSETS = [
    # US Stocks
    {"ticker": "NVDA", "name": "Nvidia", "category": "AI/Tech"},
    {"ticker": "TSLA", "name": "Tesla", "category": "EV/Auto"},
    {"ticker": "AAPL", "name": "Apple", "category": "Tech"},
    {"ticker": "MSFT", "name": "Microsoft", "category": "Tech"},
    {"ticker": "AMZN", "name": "Amazon", "category": "Tech/Retail"},
    {"ticker": "META", "name": "Meta", "category": "Tech/Social"},
    {"ticker": "GOOGL", "name": "Google", "category": "Tech"},
    {"ticker": "ARM", "name": "ARM Holdings", "category": "Semiconductors"},
    {"ticker": "AVGO", "name": "Broadcom", "category": "Semiconductors"},
    {"ticker": "PLTR", "name": "Palantir", "category": "AI/Data"},
    {"ticker": "MSTR", "name": "MicroStrategy", "category": "Bitcoin Proxy"},
    {"ticker": "COIN", "name": "Coinbase", "category": "Crypto Exchange"},
    # Indices
    {"ticker": "^GSPC", "name": "S&P 500", "category": "Index"},
    {"ticker": "^IXIC", "name": "NASDAQ", "category": "Index"},
    {"ticker": "^DJI", "name": "Dow Jones", "category": "Index"},
    {"ticker": "^N225", "name": "Nikkei 225", "category": "Index"},
    # Safe havens
    {"ticker": "GLD", "name": "Gold ETF", "category": "Commodity ETF"},
    {"ticker": "TLT", "name": "20+ Year Treasury", "category": "Bond ETF"},
]

COMMODITY_ASSETS = [
    {"ticker": "GC=F", "name": "Gold Futures", "category": "Precious Metal"},
    {"ticker": "SI=F", "name": "Silver Futures", "category": "Precious Metal"},
    {"ticker": "CL=F", "name": "Crude Oil WTI", "category": "Energy"},
    {"ticker": "BZ=F", "name": "Brent Oil", "category": "Energy"},
    {"ticker": "NG=F", "name": "Natural Gas", "category": "Energy"},
    {"ticker": "HG=F", "name": "Copper", "category": "Industrial Metal"},
    {"ticker": "PL=F", "name": "Platinum", "category": "Precious Metal"},
]

CRYPTO_ASSETS = [
    {"ticker": "BTC-USD", "name": "Bitcoin", "category": "Crypto"},
    {"ticker": "ETH-USD", "name": "Ethereum", "category": "Crypto"},
    {"ticker": "SOL-USD", "name": "Solana", "category": "Crypto"},
    {"ticker": "XRP-USD", "name": "Ripple", "category": "Crypto"},
    {"ticker": "ADA-USD", "name": "Cardano", "category": "Crypto"},
    {"ticker": "DOGE-USD", "name": "Dogecoin", "category": "Meme Crypto"},
    {"ticker": "AVAX-USD", "name": "Avalanche", "category": "Crypto"},
    {"ticker": "DOT-USD", "name": "Polkadot", "category": "Crypto"},
]


# ─── HMM-gated LPPLS ─────────────────────────────────────────────────


def run_lppls_with_hmm(t: np.ndarray, prices: np.ndarray, domain: str = "finance") -> dict:
    """Run HMM pre-screening then LPPLS.

    HMM is advisory, not a blocker:
    - BUBBLE/GROWTH → run LPPLS, trust verdict
    - NORMAL → run LPPLS anyway, but downgrade BUBBLE→POSSIBLE
    This preserves recall while reducing FP on truly flat assets.
    """
    hmm = hmm_prescreen(prices)
    lppls = fit_lppls_simple(t, prices)
    lppls["hmm_regime"] = hmm["regime"]
    lppls["hmm_bubble_prob"] = hmm["bubble_prob"]
    lppls["hmm_skipped"] = False

    # Hurst exponent — independent persistence signal
    H = compute_hurst(prices)
    lppls["hurst"] = H
    lppls["hurst_supports_bubble"] = hurst_supports_bubble(H)

    # HMM NORMAL + LPPLS BUBBLE → downgrade to POSSIBLE (conflicting signals)
    if not hmm["should_fit"] and lppls["verdict"] == "BUBBLE":
        lppls["verdict"] = "POSSIBLE"
        lppls["hmm_downgraded"] = True

    # HMM NORMAL + LPPLS POSSIBLE + low quality → downgrade to NO_BUBBLE
    if not hmm["should_fit"] and lppls["verdict"] == "POSSIBLE" and lppls["quality_score"] < 0.8:
        lppls["verdict"] = "NO_BUBBLE"
        lppls["hmm_downgraded"] = True

    return lppls


# ─── Data Loading ─────────────────────────────────────────────────────


def load_yfinance_data(
    ticker: str, period: str = "2y"
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    """Load price data from Yahoo Finance."""
    import yfinance as yf

    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {ticker}")

    close = df["Close"].dropna().values.flatten()
    if len(close) < 30:
        raise ValueError(f"Too few data points for {ticker}: {len(close)}")

    dates = df.index[: len(close)].strftime("%Y-%m-%d").tolist()
    t = np.arange(len(close), dtype=float)
    price = float(close[-1])

    return t, close, dates, price


# ─── Main Prediction Pipeline ────────────────────────────────────────


def run_all_predictions(
    finance: bool = True,
    commodities: bool = True,
    crypto: bool = True,
    housing: bool = True,
    run_rag: bool = True,
    output: str = "data/all_predictions.json",
) -> dict:
    """Run predictions across ALL domains."""

    start = time.time()
    all_results = []
    bubble_signals = []

    # ─── 1. Finance ──────────────────────────────────────────────────
    if finance:
        console.print()
        console.print(
            Panel(
                f"[bold]PHASE 1: Finance ({len(FINANCE_ASSETS)} assets)[/bold]\n"
                f"Stocks, Indices, ETFs",
                title="Finance Predictions",
                border_style="blue",
            )
        )

        for asset in FINANCE_ASSETS:
            try:
                t, prices, dates, price = load_yfinance_data(asset["ticker"], period="2y")
                lppls = run_lppls_with_hmm(t, prices)
                lppls["sornette_pass"] = passes_sornette_filters(
                    tc=lppls["tc_estimate"],
                    m=lppls["m"],
                    omega=lppls["omega"],
                    a=lppls.get("A", 0),
                    b=lppls["B"],
                    c=lppls.get("C", 0),
                    t_start=0,
                    t_end=t[-1],
                    domain="finance",
                )
                if not lppls["sornette_pass"] and lppls["verdict"] == "BUBBLE":
                    lppls["verdict"] = "POSSIBLE"

                result = {
                    "domain": "finance",
                    "ticker": asset["ticker"],
                    "name": asset["name"],
                    "category": asset["category"],
                    "current_price": price,
                    "data_points": len(prices),
                    "period": f"{dates[0]} to {dates[-1]}",
                    "domain_status": get_status_label("finance"),
                    **lppls,
                }

                all_results.append(result)
                if lppls["verdict"] in ("BUBBLE", "POSSIBLE"):
                    bubble_signals.append(result)

                verdict_color = {"BUBBLE": "red", "POSSIBLE": "yellow", "NO_BUBBLE": "green"}.get(
                    lppls["verdict"], "white"
                )
                console.print(
                    f"  {asset['ticker']:8s} | ${price:>10,.2f} | "
                    f"R²={lppls['r_squared']:.3f} | Q={lppls['quality_score']:.3f} | "
                    f"tc={lppls.get('tc_date_str', '—'):12s} | "
                    f"[{verdict_color}]{lppls['verdict']}[/{verdict_color}]"
                )

            except Exception as e:
                console.print(f"  [red]{asset['ticker']:8s} | FAILED: {str(e)[:50]}[/red]")

    # ─── 2. Commodities ──────────────────────────────────────────────
    if commodities:
        console.print()
        console.print(
            Panel(
                f"[bold]PHASE 2: Commodities ({len(COMMODITY_ASSETS)} assets)[/bold]\n"
                f"Gold, Silver, Oil, Gas, Copper",
                title="Commodity Predictions",
                border_style="yellow",
            )
        )

        for asset in COMMODITY_ASSETS:
            try:
                t, prices, dates, price = load_yfinance_data(asset["ticker"], period="2y")
                lppls = run_lppls_with_hmm(t, prices)
                lppls["sornette_pass"] = passes_sornette_filters(
                    tc=lppls["tc_estimate"],
                    m=lppls["m"],
                    omega=lppls["omega"],
                    a=lppls.get("A", 0),
                    b=lppls["B"],
                    c=lppls.get("C", 0),
                    t_start=0,
                    t_end=t[-1],
                    domain="finance",
                )
                if not lppls["sornette_pass"] and lppls["verdict"] == "BUBBLE":
                    lppls["verdict"] = "POSSIBLE"

                result = {
                    "domain": "commodities",
                    "ticker": asset["ticker"],
                    "name": asset["name"],
                    "category": asset["category"],
                    "current_price": price,
                    "data_points": len(prices),
                    "period": f"{dates[0]} to {dates[-1]}",
                    "domain_status": get_status_label("commodities"),
                    **lppls,
                }

                all_results.append(result)
                if lppls["verdict"] in ("BUBBLE", "POSSIBLE"):
                    bubble_signals.append(result)

                verdict_color = {"BUBBLE": "red", "POSSIBLE": "yellow", "NO_BUBBLE": "green"}.get(
                    lppls["verdict"], "white"
                )
                console.print(
                    f"  {asset['ticker']:8s} | ${price:>10,.2f} | "
                    f"R²={lppls['r_squared']:.3f} | Q={lppls['quality_score']:.3f} | "
                    f"tc={lppls.get('tc_date_str', '—'):12s} | "
                    f"[{verdict_color}]{lppls['verdict']}[/{verdict_color}]"
                )

            except Exception as e:
                console.print(f"  [red]{asset['ticker']:8s} | FAILED: {str(e)[:50]}[/red]")

    # ─── 3. Crypto ───────────────────────────────────────────────────
    if crypto:
        console.print()
        console.print(
            Panel(
                f"[bold]PHASE 3: Crypto ({len(CRYPTO_ASSETS)} assets)[/bold]\n"
                f"BTC, ETH, SOL, XRP, ADA, DOGE, AVAX, DOT",
                title="Crypto Predictions",
                border_style="magenta",
            )
        )

        for asset in CRYPTO_ASSETS:
            try:
                t, prices, dates, price = load_yfinance_data(asset["ticker"], period="2y")
                lppls = run_lppls_with_hmm(t, prices)
                lppls["sornette_pass"] = passes_sornette_filters(
                    tc=lppls["tc_estimate"],
                    m=lppls["m"],
                    omega=lppls["omega"],
                    a=lppls.get("A", 0),
                    b=lppls["B"],
                    c=lppls.get("C", 0),
                    t_start=0,
                    t_end=t[-1],
                    domain="finance",
                )
                if not lppls["sornette_pass"] and lppls["verdict"] == "BUBBLE":
                    lppls["verdict"] = "POSSIBLE"

                result = {
                    "domain": "crypto",
                    "ticker": asset["ticker"],
                    "name": asset["name"],
                    "category": asset["category"],
                    "current_price": price,
                    "data_points": len(prices),
                    "period": f"{dates[0]} to {dates[-1]}",
                    "domain_status": get_status_label("crypto"),
                    **lppls,
                }

                all_results.append(result)
                if lppls["verdict"] in ("BUBBLE", "POSSIBLE"):
                    bubble_signals.append(result)

                verdict_color = {"BUBBLE": "red", "POSSIBLE": "yellow", "NO_BUBBLE": "green"}.get(
                    lppls["verdict"], "white"
                )
                console.print(
                    f"  {asset['ticker']:10s} | ${price:>12,.2f} | "
                    f"R²={lppls['r_squared']:.3f} | Q={lppls['quality_score']:.3f} | "
                    f"tc={lppls.get('tc_date_str', '—'):12s} | "
                    f"[{verdict_color}]{lppls['verdict']}[/{verdict_color}]"
                )

            except Exception as e:
                console.print(f"  [red]{asset['ticker']:10s} | FAILED: {str(e)[:50]}[/red]")

    # ─── 4. Housing ──────────────────────────────────────────────────
    if housing:
        console.print()
        console.print(
            Panel(
                f"[bold]PHASE 4: Housing ({len(HOT_MARKETS)} markets)[/bold]\n"
                f"Zillow ZHVI, deseasonalized, monthly data",
                title="Housing Predictions",
                border_style="cyan",
            )
        )

        loader = HousingDataLoader()
        housing_episodes = loader.scan_hot_markets(years_back=5, deseasonalize=True)

        for ep in housing_episodes:
            prices = np.array(ep.prices_deseasonalized)
            t = np.arange(len(prices), dtype=float)
            lppls = fit_lppls_with_ds(t, prices, domain="housing")

            ep.tc_date_str = lppls.get("tc_date_str")
            ep.quality_score = lppls.get("quality_score", 0.0)
            ep.r_squared = lppls.get("r_squared", 0.0)
            ep.m_param = lppls.get("m")
            ep.omega_param = lppls.get("omega")
            ep.verdict = lppls.get("verdict", "UNKNOWN")

            result = {
                "domain": "housing",
                "ticker": ep.region,
                "name": ep.region,
                "category": f"Housing ({ep.region_type})",
                "current_price": ep.prices[-1],
                "data_points": ep.n_points,
                "period": f"{ep.start_date} to {ep.end_date}",
                "domain_status": get_status_label("housing"),
                **lppls,
            }

            all_results.append(result)
            if lppls["verdict"] in ("BUBBLE", "POSSIBLE"):
                bubble_signals.append(result)

            verdict_color = {"BUBBLE": "red", "POSSIBLE": "yellow", "NO_BUBBLE": "green"}.get(
                lppls["verdict"], "white"
            )
            console.print(
                f"  {ep.region:20s} | ${ep.prices[-1]:>10,.0f} | "
                f"R²={lppls['r_squared']:.3f} | Q={lppls['quality_score']:.3f} | "
                f"tc={lppls.get('tc_date_str', '—'):12s} | "
                f"[{verdict_color}]{lppls['verdict']}[/{verdict_color}]"
            )

    # ─── 5. RAG Analysis for Bubble Signals ──────────────────────────
    rag_briefings = []
    if run_rag and bubble_signals:
        console.print()
        console.print(
            Panel(
                f"[bold]PHASE 5: RAG Analysis ({len(bubble_signals)} bubble signals)[/bold]\n"
                f"Collecting news and fundamentals...",
                title="RAG Semantic Validation",
                border_style="magenta",
            )
        )

        rag = RAGEngine()

        for signal in bubble_signals:
            if signal["verdict"] not in ("BUBBLE", "POSSIBLE"):
                continue

            console.print(f"\n  Analyzing: {signal['ticker']} ({signal['name']})")

            try:
                context = rag.collect_context(
                    ticker=signal["ticker"],
                    start_date=signal["period"].split(" to ")[0],
                    end_date=signal["period"].split(" to ")[1],
                    max_results=6,
                )

                briefing = rag.analyze_bubble(
                    ticker=signal["ticker"],
                    context=context,
                    tc_date=signal.get("tc_date_str"),
                    quality_score=signal["quality_score"],
                    hmm_regime=None,
                    price=signal["current_price"],
                )

                rag_briefings.append(
                    {
                        "ticker": signal["ticker"],
                        "domain": signal["domain"],
                        **briefing.to_dict(),
                    }
                )

                conclusion_color = {
                    "FUNDAMENTAL_GROWTH": "green",
                    "SPECULATIVE_BUBBLE": "red",
                    "MIXED": "yellow",
                }.get(briefing.conclusion, "white")

                console.print(
                    f"    Conclusion: [{conclusion_color}]{briefing.conclusion}[/{conclusion_color}] "
                    f"(confidence: {briefing.confidence_in_conclusion:.0%})"
                )
                console.print(f"    Key evidence: {briefing.key_evidence[:100]}")

            except Exception as e:
                console.print(f"    [red]RAG failed: {str(e)[:80]}[/red]")

    # ─── Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - start

    bubbles = [r for r in all_results if r["verdict"] == "BUBBLE"]
    possibles = [r for r in all_results if r["verdict"] == "POSSIBLE"]
    no_bubbles = [r for r in all_results if r["verdict"] == "NO_BUBBLE"]

    console.print()
    console.print("=" * 100)
    console.print("[bold]ALL PREDICTIONS SUMMARY[/bold]")
    console.print("=" * 100)

    # By domain
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Domain", style="white")
    table.add_column("Status", style="white")
    table.add_column("Total", style="white", justify="right")
    table.add_column("BUBBLE", style="red", justify="right")
    table.add_column("POSSIBLE", style="yellow", justify="right")
    table.add_column("NO_BUBBLE", style="green", justify="right")

    for domain in ["finance", "commodities", "crypto", "housing"]:
        domain_results = [r for r in all_results if r["domain"] == domain]
        if not domain_results:
            continue
        b = len([r for r in domain_results if r["verdict"] == "BUBBLE"])
        p = len([r for r in domain_results if r["verdict"] == "POSSIBLE"])
        n = len([r for r in domain_results if r["verdict"] == "NO_BUBBLE"])
        status = get_status_label(domain)
        status_icon = {
            "production-like": "✅",
            "experimental": "⚠️",
            "research-only": "🔬",
            "synthetic-only": "🧪",
        }.get(status, "❓")
        table.add_row(
            domain.upper(),
            f"{status_icon} {status}",
            str(len(domain_results)),
            str(b),
            str(p),
            str(n),
        )

    console.print(table)

    # Top bubble signals
    if bubbles:
        console.print("\n[bold red]🔴 BUBBLE SIGNALS (immediate attention):[/bold red]")
        for b in sorted(bubbles, key=lambda x: x["quality_score"], reverse=True):
            console.print(
                f"  • {b['ticker']:12s} | {b['name']:20s} | tc={b.get('tc_date_str', '—'):12s} | Q={b['quality_score']:.3f}"
            )

    if possibles:
        console.print("\n[bold yellow]🟡 POSSIBLE SIGNALS (monitor closely):[/bold yellow]")
        for p in sorted(possibles, key=lambda x: x["quality_score"], reverse=True):
            console.print(
                f"  • {p['ticker']:12s} | {p['name']:20s} | tc={p.get('tc_date_str', '—'):12s} | Q={p['quality_score']:.3f}"
            )

    console.print(f"\n[bold]Total assets analyzed: {len(all_results)}[/bold]")
    console.print(
        f"[bold red]BUBBLE: {len(bubbles)}[/bold red] | "
        f"[bold yellow]POSSIBLE: {len(possibles)}[/bold yellow] | "
        f"[bold green]NO_BUBBLE: {len(no_bubbles)}[/bold green]"
    )
    console.print(f"[bold]Total time: {elapsed:.1f}s[/bold]")

    # ─── 6. Eval Set Check ───────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            "[bold]PHASE 6: Eval Set Check[/bold]\nRunning locked evaluation set...",
            title="Eval Set",
            border_style="magenta",
        )
    )

    try:
        from src.contract.eval_runner import load_eval_set, run_lppls_for_case

        eval_set = load_eval_set()
        eval_results = []
        for case in eval_set[:5]:  # Quick check first 5
            lppls = run_lppls_for_case(case)
            eval_results.append(
                {
                    "ticker": case["ticker"],
                    "expected": case["label"],
                    "lppls_verdict": lppls.get("verdict"),
                    "quality": lppls.get("quality_score"),
                }
            )
        console.print(f"[dim]Quick eval check: {len(eval_results)} cases processed[/dim]")
        for er in eval_results:
            match = (
                "✅"
                if er["lppls_verdict"] in ("BUBBLE", "POSSIBLE")
                and er["expected"] == "SPECULATIVE_BUBBLE"
                else "✅"
                if er["lppls_verdict"] == "NO_BUBBLE" and er["expected"] == "FUNDAMENTAL_GROWTH"
                else "⚠️"
            )
            console.print(
                f"  {match} {er['ticker']}: expected={er['expected']}, lppls={er['lppls_verdict']}"
            )
    except Exception as e:
        console.print(f"[dim]Eval set check skipped: {str(e)[:80]}[/dim]")

    # ─── 7. Delta Tracker ────────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            "[bold]PHASE 7: Delta Tracking[/bold]\nComparing with previous predictions...",
            title="Delta Report",
            border_style="green",
        )
    )

    tracker = DeltaTracker()
    delta_report = tracker.compare_and_save(all_results)
    tracker.print_delta(delta_report)

    # Save results
    output_data = {
        "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "elapsed_seconds": round(elapsed, 1),
        "total_assets": len(all_results),
        "bubble_count": len(bubbles),
        "possible_count": len(possibles),
        "no_bubble_count": len(no_bubbles),
        "results": all_results,
        "rag_briefings": rag_briefings,
        "delta_report": delta_report.to_dict(),
    }

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        console.print(f"\n[dim]Full results saved to: {output}[/dim]")

    return output_data


# ─── CLI ─────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PhaseBreak Hybrid 2.0 — ALL PREDICTIONS")
    parser.add_argument("--finance-only", action="store_true")
    parser.add_argument("--commodities-only", action="store_true")
    parser.add_argument("--crypto-only", action="store_true")
    parser.add_argument("--housing-only", action="store_true")
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--output", type=str, default="data/all_predictions.json")
    args = parser.parse_args()

    run_all_predictions(
        finance=not args.commodities_only and not args.crypto_only and not args.housing_only,
        commodities=not args.finance_only and not args.crypto_only and not args.housing_only,
        crypto=not args.finance_only and not args.commodities_only and not args.housing_only,
        housing=not args.finance_only and not args.commodities_only and not args.crypto_only,
        run_rag=not args.no_rag,
        output=args.output,
    )


if __name__ == "__main__":
    main()
