"""HybridOrchestrator — coordinates LPPLS math engine + RAG semantic analysis.

This is the main entry point for PhaseBreak Hybrid 2.0.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.hybrid.rag_engine import ContextBundle, HybridBriefing, RAGEngine

log = structlog.get_logger()
console = Console()


@dataclass
class LPPLSInput:
    """Input for LPPLS analysis (from PhaseBreak core or manual)."""

    ticker: str
    start_date: str
    end_date: str
    frequency: str = "daily"  # daily, monthly, quarterly


@dataclass
class LPPLSResult:
    """Simplified LPPLS fit result."""

    ticker: str
    tc_estimate: float | None  # critical time (index or date)
    tc_date_str: str | None  # human-readable date
    quality_score: float  # 0-1
    r_squared: float
    hmm_regime: str | None  # NORMAL, GROWTH, BUBBLE
    hmm_bubble_prob: float  # 0-1
    verdict: str  # BUBBLE, POSSIBLE, NO_BUBBLE
    price: float | None
    params: dict[str, float] | None = None  # m, omega, B, etc.


@dataclass
class HybridResult:
    """Complete hybrid analysis result."""

    lppls: LPPLSResult | None
    briefing: HybridBriefing
    context: ContextBundle | None = None
    total_time_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = {
            "briefing": self.briefing.to_dict(),
            "total_time_sec": self.total_time_sec,
        }
        if self.lppls:
            d["lppls"] = {
                "ticker": self.lppls.ticker,
                "tc_date": self.lppls.tc_date_str,
                "quality_score": self.lppls.quality_score,
                "hmm_regime": self.lppls.hmm_regime,
                "verdict": self.lppls.verdict,
                "price": self.lppls.price,
            }
        if self.context:
            d["context_sources"] = len(self.context.search_results)
            d["context_time_sec"] = self.context.collection_time_sec
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class HybridOrchestrator:
    """Coordinates LPPLS math engine + RAG semantic analysis.

    Usage:
        orchestrator = HybridOrchestrator()
        result = orchestrator.run_hybrid_analysis(
            ticker="NVDA",
            start_date="2023-01-01",
            end_date="2024-06-20",
            known_outcome="NVDA continued rising on AI demand, no crash",
        )
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        model: str = "gpt-4o",
        use_lppls_core: bool = True,
        skip_llm: bool = False,  # For mock testing
    ):
        """
        Args:
            openai_api_key: OpenAI API key
            openai_base_url: Custom API URL (Ollama, Together, etc.)
            model: LLM model name
            use_lppls_core: If True, try to import PhaseBreak LPPLS core
        """
        self.rag = RAGEngine(
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            model=model,
            skip_llm=skip_llm,
        )

        # Try to import PhaseBreak LPPLS core
        self.lppls_core = None
        if use_lppls_core:
            self.lppls_core = self._try_import_lppls()
            if self.lppls_core:
                log.info("lppls_core_loaded", source="PhaseBreak 2026")
            else:
                log.warning("lppls_core_not_found", fallback="manual_input")

    def _try_import_lppls(self):
        """Try to import PhaseBreak LPPLS core from sibling directory."""
        # Try multiple paths
        paths_to_try = [
            # From PhaseBreak Hybrid 2.0 directory
            Path(__file__).parent.parent.parent / "PhaseBreak 2026" / "src",
            # From working directory
            Path("D:\\ВСЕ ПРОЕКТЫ ОПТИМЕЗАЦИЯ\\PhaseBreak 2026\\src"),
            # From sys.path
            Path("PhaseBreak 2026") / "src",
        ]

        for p in paths_to_try:
            if (p / "lppls").exists():
                if str(p) not in sys.path:
                    sys.path.insert(0, str(p))
                try:
                    from lppls.ensemble import HMMLPPLSEnsemble
                    from lppls.data import load_yfinance

                    return {"ensemble": HMMLPPLSEnsemble, "loader": load_yfinance}
                except ImportError:
                    continue

        return None

    # ─── Main Entry Point ──────────────────────────────────────────────

    def run_hybrid_analysis(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        tc_date: str | None = None,
        quality_score: float | None = None,
        hmm_regime: str | None = None,
        price: float | None = None,
        known_outcome: str | None = None,
        run_lppls: bool = True,
    ) -> HybridResult:
        """Run full hybrid analysis: LPPLS + RAG.

        Args:
            ticker: Asset ticker/symbol
            start_date: Analysis period start
            end_date: Analysis period end
            tc_date: Pre-computed LPPLS tc date (or None to compute)
            quality_score: Pre-computed LPPLS quality (or None to compute)
            hmm_regime: Pre-computed HMM regime (or None to compute)
            price: Current price (or None to fetch)
            known_outcome: For historical validation — what actually happened
            run_lppls: If True, try to run LPPLS analysis

        Returns:
            HybridResult with LPPLS + RAG analysis
        """
        start = time.time()

        # Step 1: LPPLS analysis (if enabled and available)
        lppls_result = None
        if run_lppls and self.lppls_core:
            lppls_result = self._run_lppls_analysis(ticker, start_date, end_date)
            # Use computed values if not provided
            if lppls_result:
                tc_date = tc_date or lppls_result.tc_date_str
                quality_score = quality_score if quality_score is not None else lppls_result.quality_score
                hmm_regime = hmm_regime or lppls_result.hmm_regime
                price = price or lppls_result.price
        elif run_lppls and not self.lppls_core:
            log.warning("lppls_not_available", fallback="manual_parameters")

        # Ensure we have at least some quality score
        if quality_score is None:
            quality_score = 0.5  # Default neutral

        # Step 2: RAG analysis
        briefing = self.rag.run_full_analysis(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            tc_date=tc_date,
            quality_score=quality_score,
            hmm_regime=hmm_regime,
            price=price,
            known_outcome=known_outcome,
        )

        total_time = time.time() - start

        return HybridResult(
            lppls=lppls_result,
            briefing=briefing,
            total_time_sec=round(total_time, 1),
        )

    def _run_lppls_analysis(
        self, ticker: str, start_date: str, end_date: str
    ) -> LPPLSResult | None:
        """Run LPPLS analysis using PhaseBreak core."""
        if not self.lppls_core:
            return None

        try:
            log.info("running_lppls", ticker=ticker, period=f"{start_date} to {end_date}")

            # Load data
            loader = self.lppls_core["loader"]
            ds = loader(
                ticker=ticker,
                start=start_date,
                end=end_date,
            )

            # Run ensemble analysis
            ensemble = self.lppls_core["ensemble"]()
            result = ensemble.analyze(ds.t, ds.log_price)

            return LPPLSResult(
                ticker=ticker,
                tc_estimate=result.tc_estimate if hasattr(result, "tc_estimate") else None,
                tc_date_str=None,  # Would need date conversion
                quality_score=result.quality_score if hasattr(result, "quality_score") else 0.5,
                r_squared=result.r_squared if hasattr(result, "r_squared") else 0.0,
                hmm_regime=result.hmm_regime if hasattr(result, "hmm_regime") else None,
                hmm_bubble_prob=result.hmm_bubble_prob if hasattr(result, "hmm_bubble_prob") else 0.0,
                verdict=result.final_verdict if hasattr(result, "final_verdict") else "UNKNOWN",
                price=float(ds.prices[-1]) if hasattr(ds, "prices") else None,
            )

        except Exception as e:
            log.error("lppls_analysis_failed", ticker=ticker, error=str(e))
            return None

    # ─── Display ───────────────────────────────────────────────────────

    @staticmethod
    def print_briefing(result: HybridResult) -> None:
        """Print formatted briefing to console using Rich."""
        b = result.briefing

        # Header
        console.print()
        console.print(
            Panel(
                f"[bold]{b.ticker}[/bold] — Hybrid Analysis\n"
                f"tc: {b.tc_date or 'N/A'} | Quality: {b.quality_score:.2f} | "
                f"HMM: {b.hmm_regime or 'N/A'}",
                title="PhaseBreak Hybrid 2.0",
                border_style="blue",
            )
        )

        # Conclusion
        conclusion_color = {
            "FUNDAMENTAL_GROWTH": "green",
            "SPECULATIVE_BUBBLE": "red",
            "MIXED": "yellow",
        }.get(b.conclusion, "white")

        console.print()
        console.print(
            Panel(
                f"[bold {conclusion_color}]{b.conclusion}[/bold {conclusion_color}]\n"
                f"Confidence: {b.confidence_in_conclusion:.0%}\n"
                f"Recommendation: {b.recommendation}\n\n"
                f"[dim]{b.key_evidence}[/dim]",
                title="Conclusion",
                border_style=conclusion_color,
            )
        )

        # Factors table
        table = Table(title="Factor Analysis", show_header=True, header_style="bold magenta")
        table.add_column("Type", style="cyan", width=12)
        table.add_column("Factor", style="white")

        for f in b.fundamental_factors:
            table.add_row("[green]Fundamental[/green]", f)
        for f in b.speculative_factors:
            table.add_row("[red]Speculative[/red]", f)

        if not b.fundamental_factors and not b.speculative_factors:
            table.add_row("N/A", "No factors identified")

        console.print()
        console.print(table)

        # Sentiment
        console.print()
        console.print(f"[dim]Sentiment:[/dim] {b.sentiment_summary}")

        # Historical validation
        if b.matches_known_outcome is not None:
            match_icon = "✅" if b.matches_known_outcome else "❌"
            console.print()
            console.print(
                f"[dim]Historical Validation:[/dim] {match_icon} "
                f"{'Matches' if b.matches_known_outcome else 'Does not match'} known outcome"
            )
            if b.explanation:
                console.print(f"[dim]Explanation:[/dim] {b.explanation}")

        # Timing
        console.print()
        console.print(
            f"[dim]Analysis time: {result.total_time_sec:.1f}s | "
            f"Context: {len(b.raw_llm_response)} chars[/dim]"
        )
        console.print()

    # ─── Batch Analysis ────────────────────────────────────────────────

    def run_batch_analysis(
        self,
        assets: list[dict[str, Any]],
        output_path: str | None = None,
    ) -> list[HybridResult]:
        """Run analysis on multiple assets.

        Args:
            assets: List of dicts with keys:
                - ticker (required)
                - start_date, end_date (required)
                - tc_date, quality_score, hmm_regime, price (optional)
                - known_outcome (optional, for validation)
            output_path: Save results to JSON file

        Returns:
            List of HybridResult
        """
        results = []
        for i, asset in enumerate(assets, 1):
            console.print(f"\n[bold cyan]=== [{i}/{len(assets)}] {asset['ticker']} ===[/bold cyan]")

            result = self.run_hybrid_analysis(
                ticker=asset["ticker"],
                start_date=asset["start_date"],
                end_date=asset["end_date"],
                tc_date=asset.get("tc_date"),
                quality_score=asset.get("quality_score"),
                hmm_regime=asset.get("hmm_regime"),
                price=asset.get("price"),
                known_outcome=asset.get("known_outcome"),
            )

            self.print_briefing(result)
            results.append(result)

        # Save results
        if output_path:
            output = {
                "analysis_date": time.strftime("%Y-%m-%d"),
                "model_version": "hybrid-v1",
                "results": [r.to_dict() for r in results],
            }
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            log.info("results_saved", path=output_path)

        return results
