"""RAG Engine for PhaseBreak Hybrid.

Collects context via web search + LLM analysis, returns structured briefing.
No external vector DB needed for Phase 1 — direct LLM with search context.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import structlog
from openai import OpenAI
from pydantic import BaseModel, Field

from src.hybrid.prompts import (
    CONTEXT_QUERY_PROMPT,
    HISTORICAL_VALIDATION_PROMPT,
    HYBRID_BRIEFING_PROMPT,
    SYSTEM_PROMPT,
)

log = structlog.get_logger()


@dataclass
class SearchResult:
    """Single search result snippet."""

    title: str
    url: str
    snippet: str
    source: str  # e.g., "google", "news", "wikipedia"
    date: str | None = None


@dataclass
class ContextBundle:
    """Collected context for a single asset analysis."""

    ticker: str
    query: str
    search_results: list[SearchResult] = field(default_factory=list)
    raw_context: str = ""
    collection_time_sec: float = 0.0


@dataclass
class HybridBriefing:
    """Final structured briefing from RAG analysis."""

    ticker: str
    tc_date: str | None
    quality_score: float
    hmm_regime: str | None
    price: float | None

    fundamental_factors: list[str] = field(default_factory=list)
    speculative_factors: list[str] = field(default_factory=list)
    sentiment_summary: str = ""
    conclusion: str = "UNKNOWN"  # FUNDAMENTAL_GROWTH | SPECULATIVE_BUBBLE | MIXED
    confidence_in_conclusion: float = 0.0
    key_evidence: str = ""
    recommendation: str = "CAUTION"  # HOLD | CAUTION | AVOID

    # Metadata
    matches_known_outcome: bool | None = None
    explanation: str = ""
    raw_llm_response: str = ""
    analysis_time_sec: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "ticker": self.ticker,
            "tc_date": self.tc_date,
            "quality_score": self.quality_score,
            "hmm_regime": self.hmm_regime,
            "price": self.price,
            "fundamental_factors": self.fundamental_factors,
            "speculative_factors": self.speculative_factors,
            "sentiment_summary": self.sentiment_summary,
            "conclusion": self.conclusion,
            "confidence_in_conclusion": self.confidence_in_conclusion,
            "key_evidence": self.key_evidence,
            "recommendation": self.recommendation,
            "matches_known_outcome": self.matches_known_outcome,
            "explanation": self.explanation,
            "analysis_time_sec": self.analysis_time_sec,
            "error": self.error,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class RAGEngine:
    """Retrieval-Augmented Generation engine for financial analysis.

    Phase 2 approach: RSS feeds (primary) → DuckDuckGo (fallback) → LLM analysis.
    Includes 24h response cache to avoid repeated LLM calls for same ticker.
    """

    _llm_cache: dict[str, tuple[float, dict]] = {}  # {cache_key: (timestamp, response_dict)}
    _CACHE_TTL = 86400  # 24 hours

    # Default LLM routing: Ollama (free) for drafts, OpenAI for polish
    OLLAMA_URL = "http://localhost:11434/v1"
    OLLAMA_MODELS = {
        "draft": "gemma4:26b",  # Best for structured JSON analysis
        "fast": "qwen2.5:14b",  # Quick context summaries
        "reasoning": "qwq:32b",  # Complex chain-of-thought
    }

    def __init__(
        self,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        model: str | None = None,
        use_free_search: bool = True,
        skip_llm: bool = False,
        rss_provider: Any | None = None,
        use_ollama: bool = True,  # Default: use local Ollama
        ollama_model: str = "draft",  # draft/fast/reasoning
    ):
        """
        Args:
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            openai_base_url: Custom base URL (for compatible APIs)
            model: LLM model name (overrides auto-selection)
            use_free_search: If True, use free web search for context collection
            skip_llm: If True, skip LLM calls (for mock testing)
            rss_provider: RSS feed provider
            use_ollama: If True, route to local Ollama (free). False = OpenAI API.
            ollama_model: Which Ollama model: "draft" (gemma4:26b), "fast" (qwen2.5:14b), "reasoning" (qwq:32b)
        """
        self.use_ollama = use_ollama
        self.use_free_search = use_free_search
        self.skip_llm = skip_llm

        # Model selection
        if model:
            self.model = model
        elif use_ollama:
            self.model = self.OLLAMA_MODELS.get(ollama_model, self.OLLAMA_MODELS["draft"])
        else:
            self.model = "gpt-4o"

        # RSS provider (primary search)
        self._rss = rss_provider
        if self._rss is None:
            try:
                from src.hybrid.rss_provider import RSSFeedProvider

                self._rss = RSSFeedProvider()
            except Exception as e:
                log.info("rss_provider_not_available", reason=str(e))
                self._rss = None

        # LLM client — Ollama or OpenAI, both via OpenAI SDK
        self.client = None
        if not skip_llm:
            if use_ollama:
                # Ollama exposes OpenAI-compatible API at /v1
                try:
                    self.client = OpenAI(
                        base_url=self.OLLAMA_URL,
                        api_key="ollama",  # Ollama doesn't need real key
                    )
                    log.info("ollama_client_initialized", model=self.model, url=self.OLLAMA_URL)
                except Exception as e:
                    log.warning("ollama_init_failed", error=str(e), fallback="openai")
                    self.client = OpenAI()
                    self.model = "gpt-4o"
                    self.use_ollama = False
            else:
                client_kwargs: dict[str, Any] = {}
                if openai_api_key:
                    client_kwargs["api_key"] = openai_api_key
                if openai_base_url:
                    client_kwargs["base_url"] = openai_base_url
                self.client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()

        log.info(
            "rag_engine_initialized",
            model=self.model,
            backend="ollama" if self.use_ollama else "openai",
            search_enabled=use_free_search,
            rss_enabled=self._rss is not None,
            skip_llm=skip_llm,
        )

    # ─── Context Collection ─────────────────────────────────────────────

    def collect_context(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        max_results: int = 10,
    ) -> ContextBundle:
        """Collect context via web search for a given ticker and time period.

        Phase 1: Uses free web search (DuckDuckGo/Google via httpx).
        Future: Can integrate paid APIs (Bloomberg, Reuters).

        Args:
            ticker: Asset ticker/symbol
            start_date: Period start (YYYY-MM-DD)
            end_date: Period end (YYYY-MM-DD)
            max_results: Max search results to collect

        Returns:
            ContextBundle with search results
        """
        start = time.time()
        query = CONTEXT_QUERY_PROMPT.format(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )

        log.info("collecting_context", ticker=ticker, period=f"{start_date} to {end_date}")

        results: list[SearchResult] = []

        # Primary: RSS feeds (structured, reliable)
        if self._rss:
            try:
                rss_hits = self._rss.search(
                    query=f"{ticker} {start_date}",
                    ticker=ticker,
                    max_results=max_results,
                )
                results = [
                    SearchResult(
                        title=h.get("title", ""),
                        url=h.get("url", ""),
                        snippet=h.get("snippet", ""),
                        source=h.get("source", "rss"),
                        date=h.get("date"),
                    )
                    for h in rss_hits
                ]
                log.info("rss_results", ticker=ticker, count=len(results))
            except Exception as e:
                log.warning("rss_search_failed", error=str(e))

        # Fallback: DuckDuckGo if RSS returned < 3 results
        if len(results) < 3:
            log.info("ddg_fallback", ticker=ticker, rss_count=len(results))
            ddg = self._duckduckgo_search(
                f"{ticker} {start_date} {end_date} news analysis", max_results
            )
            ddg += self._duckduckgo_search(
                f"{ticker} fundamentals earnings revenue {start_date}", max_results // 2
            )
            ddg += self._duckduckgo_search(
                f"{ticker} bubble hype retail speculation {start_date}", max_results // 2
            )
            results.extend(ddg)

        # Deduplicate by URL
        seen = set()
        unique_results = []
        for r in results:
            if r.url not in seen:
                seen.add(r.url)
                unique_results.append(r)

        raw_context = self._format_context(unique_results)
        elapsed = time.time() - start

        bundle = ContextBundle(
            ticker=ticker,
            query=query,
            search_results=unique_results[:max_results],
            raw_context=raw_context,
            collection_time_sec=round(elapsed, 1),
        )

        log.info(
            "context_collected",
            ticker=ticker,
            results=len(bundle.search_results),
            time_sec=bundle.collection_time_sec,
        )

        return bundle

    def _duckduckgo_search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Simple DuckDuckGo HTML search (no API key needed)."""
        results = []
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            params = {"q": query, "kl": "en-us"}
            resp = httpx.get(
                "https://html.duckduckgo.com/html/",
                headers=headers,
                params=params,
                timeout=10.0,
            )

            if resp.status_code == 200:
                # Simple HTML parsing — extract results
                text = resp.text
                # DuckDuckGo HTML format: class="result__snippet"
                import re

                # Extract titles and snippets
                snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', text, re.DOTALL)
                links = re.findall(r'class="result__a"[^>]*>(.*?)</a>', text, re.DOTALL)
                urls = re.findall(r"result__url[^>]*>(.*?)</a>", text, re.DOTALL)

                for i in range(min(len(snippets), max_results)):
                    # Clean HTML tags
                    snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()
                    title = re.sub(r"<[^>]+>", "", links[i]).strip() if i < len(links) else ""
                    url = re.sub(r"<[^>]+>", "", urls[i]).strip() if i < len(urls) else ""

                    results.append(
                        SearchResult(
                            title=title[:100],
                            url=url[:200],
                            snippet=snippet[:300],
                            source="duckduckgo",
                        )
                    )
        except Exception as e:
            log.warning("duckduckgo_search_failed", error=str(e))

        return results

    def _format_context(self, results: list[SearchResult]) -> str:
        """Format search results into readable context for LLM."""
        if not results:
            return "No search results available."

        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[{i}] {r.title}")
            if r.date:
                parts.append(f"    Date: {r.date}")
            parts.append(f"    Source: {r.url}")
            parts.append(f"    {r.snippet}")
            parts.append("")

        return "\n".join(parts)

    # ─── LLM Analysis ──────────────────────────────────────────────────

    def analyze_bubble(
        self,
        ticker: str,
        context: ContextBundle,
        tc_date: str | None = None,
        quality_score: float = 0.0,
        hmm_regime: str | None = None,
        price: float | None = None,
        known_outcome: str | None = None,
    ) -> HybridBriefing:
        """Run LLM analysis on collected context.

        Args:
            ticker: Asset ticker
            context: Collected context bundle
            tc_date: Predicted crash date from LPPLS
            quality_score: LPPLS quality score
            hmm_regime: HMM market regime
            price: Current price
            known_outcome: For historical validation — what actually happened

        Returns:
            HybridBriefing with structured analysis
        """
        start = time.time()

        # Cache check — avoid repeated LLM calls for same ticker within 24h
        cache_key = f"{ticker}:{tc_date}:{known_outcome is not None}"
        cached = self._llm_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < self._CACHE_TTL:
            log.info("rag_cache_hit", ticker=ticker)
            briefing = self._parse_response(
                cached[1].get("raw", "{}"),
                hmm_regime=hmm_regime,
                price=price,
                tc_date=tc_date,
                quality_score=quality_score,
            )
            briefing.ticker = ticker
            briefing.analysis_time_sec = 0.0
            return briefing

        # Mock mode: return synthetic briefing without LLM
        if self.skip_llm:
            return self._mock_briefing(
                ticker=ticker,
                tc_date=tc_date,
                quality_score=quality_score,
                hmm_regime=hmm_regime,
                price=price,
                context=context,
                known_outcome=known_outcome,
            )

        # Choose prompt based on whether we know the outcome
        if known_outcome:
            prompt = HISTORICAL_VALIDATION_PROMPT.format(
                ticker=ticker,
                tc_date=tc_date or "unknown",
                quality_score=quality_score,
                model_verdict="BUBBLE" if quality_score > 0.5 else "NO_BUBBLE",
                retrieved_context=context.raw_context,
                known_outcome=known_outcome,
            )
        else:
            prompt = HYBRID_BRIEFING_PROMPT.format(
                ticker=ticker,
                tc_date=tc_date or "unknown",
                quality_score=quality_score,
                hmm_regime=hmm_regime or "unknown",
                price=price or "unknown",
                retrieved_context=context.raw_context,
            )

        try:
            log.info(
                "calling_llm",
                model=self.model,
                backend="ollama" if self.use_ollama else "openai",
                prompt_length=len(prompt),
            )
            create_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
            }
            # json_object mode: only for OpenAI API (Ollama models may not support it)
            if not self.use_ollama:
                create_kwargs["response_format"] = {"type": "json_object"}
            response = self.client.chat.completions.create(**create_kwargs)

            raw = response.choices[0].message.content or ""
            log.info("llm_response_received", response_length=len(raw))

            # Cache the raw response for 24h
            self._llm_cache[cache_key] = (time.time(), {"raw": raw})

            briefing = self._parse_response(
                raw,
                known_outcome=bool(known_outcome),
                hmm_regime=hmm_regime,
                price=price,
                tc_date=tc_date,
                quality_score=quality_score,
            )

        except Exception as e:
            log.error("llm_analysis_failed", error=str(e), error_type=type(e).__name__)
            briefing = HybridBriefing(
                ticker=ticker,
                tc_date=tc_date,
                quality_score=quality_score,
                hmm_regime=hmm_regime,
                price=price,
                error=f"LLM analysis failed: {str(e)}",
            )

        briefing.analysis_time_sec = round(time.time() - start, 1)
        return briefing

    # ─── Mock Briefing (for offline testing) ───────────────────────────

    def _mock_briefing(
        self,
        ticker: str,
        tc_date: str | None,
        quality_score: float,
        hmm_regime: str | None,
        price: float | None,
        context: ContextBundle,
        known_outcome: str | None,
    ) -> HybridBriefing:
        """Generate synthetic briefing for mock/offline testing.

        If known_outcome is provided (validation mode), use it.
        Otherwise return generic MIXED briefing — no hardcoded answers.
        """
        if known_outcome == "SPECULATIVE_BUBBLE":
            return HybridBriefing(
                ticker=ticker,
                tc_date=tc_date,
                quality_score=quality_score,
                hmm_regime=hmm_regime,
                price=price,
                fundamental_factors=[],
                speculative_factors=[
                    f"Context collected for {ticker} ({len(context.search_results)} sources)",
                    f"Multiple sources indicate speculative activity",
                ],
                sentiment_summary="Speculative signals detected based on available context",
                conclusion="SPECULATIVE_BUBBLE",
                confidence_in_conclusion=0.6,
                key_evidence="Speculative activity detected in available sources",
                recommendation="AVOID",
                matches_known_outcome=True,
                explanation="Mock mode — outcome known from eval set",
                raw_llm_response="[MOCK MODE — validation]",
                analysis_time_sec=0.1,
            )
        elif known_outcome == "FUNDAMENTAL_GROWTH":
            return HybridBriefing(
                ticker=ticker,
                tc_date=tc_date,
                quality_score=quality_score,
                hmm_regime=hmm_regime,
                price=price,
                fundamental_factors=[
                    f"Context collected for {ticker} ({len(context.search_results)} sources)",
                    f"Sources indicate fundamental growth drivers",
                ],
                speculative_factors=[],
                sentiment_summary="Fundamental growth detected based on available context",
                conclusion="FUNDAMENTAL_GROWTH",
                confidence_in_conclusion=0.6,
                key_evidence="Fundamental growth drivers identified in available sources",
                recommendation="HOLD",
                matches_known_outcome=True,
                explanation="Mock mode — outcome known from eval set",
                raw_llm_response="[MOCK MODE — validation]",
                analysis_time_sec=0.1,
            )
        else:
            # Generic fallback — no hardcoded ticker-specific answers
            return HybridBriefing(
                ticker=ticker,
                tc_date=tc_date,
                quality_score=quality_score,
                hmm_regime=hmm_regime,
                price=price,
                fundamental_factors=[
                    f"Context collected for {ticker} ({len(context.search_results)} sources)",
                ],
                speculative_factors=[],
                sentiment_summary=f"Analysis of {ticker} based on {len(context.search_results)} sources",
                conclusion="MIXED",
                confidence_in_conclusion=0.3,
                key_evidence="Insufficient context for definitive conclusion",
                recommendation="CAUTION",
                matches_known_outcome=None,
                explanation="Mock mode — no specific analysis available",
                raw_llm_response="[MOCK MODE — blind test]",
                analysis_time_sec=0.1,
            )

    def _parse_response(
        self,
        raw: str,
        known_outcome: bool = False,
        hmm_regime: str | None = None,
        price: float | None = None,
        tc_date: str | None = None,
        quality_score: float = 0.0,
    ) -> HybridBriefing:
        """Parse LLM JSON response into HybridBriefing."""
        try:
            # Try to extract JSON from response (handle markdown code blocks)
            if "```" in raw:
                # Extract JSON from code block
                import re

                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
                if match:
                    raw = match.group(1)

            data = json.loads(raw)

            return HybridBriefing(
                ticker=data.get("ticker", ""),
                tc_date=tc_date or data.get("tc_date"),
                quality_score=quality_score or data.get("quality_score", 0.0),
                hmm_regime=hmm_regime,
                price=price,
                fundamental_factors=data.get("fundamental_factors", []),
                speculative_factors=data.get("speculative_factors", []),
                sentiment_summary=data.get("sentiment_summary", ""),
                conclusion=data.get("conclusion", "UNKNOWN"),
                confidence_in_conclusion=data.get("confidence_in_conclusion", 0.0),
                key_evidence=data.get("key_evidence", ""),
                recommendation=data.get("recommendation", "CAUTION"),
                matches_known_outcome=data.get("matches_known_outcome"),
                explanation=data.get("explanation", ""),
                raw_llm_response=raw,
            )

        except json.JSONDecodeError as e:
            log.error("json_parse_failed", error=str(e), raw=raw[:200])
            return HybridBriefing(
                ticker="",
                error=f"Failed to parse LLM response as JSON: {str(e)}",
                raw_llm_response=raw,
            )

    # ─── Convenience: Full Pipeline ────────────────────────────────────

    def run_full_analysis(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        tc_date: str | None = None,
        quality_score: float = 0.0,
        hmm_regime: str | None = None,
        price: float | None = None,
        known_outcome: str | None = None,
    ) -> HybridBriefing:
        """End-to-end: collect context → analyze → return briefing.

        This is the main entry point for Phase 1.

        Args:
            ticker: Asset ticker/symbol
            start_date: Context period start
            end_date: Context period end
            tc_date: LPPLS predicted crash date
            quality_score: LPPLS quality score
            hmm_regime: HMM market regime
            price: Current price
            known_outcome: For historical validation

        Returns:
            HybridBriefing with full analysis
        """
        log.info(
            "starting_full_analysis",
            ticker=ticker,
            period=f"{start_date} to {end_date}",
            known_outcome=bool(known_outcome),
        )

        # Step 1: Collect context
        context = self.collect_context(ticker, start_date, end_date)

        # Step 2: Analyze
        briefing = self.analyze_bubble(
            ticker=ticker,
            context=context,
            tc_date=tc_date,
            quality_score=quality_score,
            hmm_regime=hmm_regime,
            price=price,
            known_outcome=known_outcome,
        )

        # Enrich with context metadata
        briefing.ticker = ticker
        briefing.tc_date = tc_date
        briefing.quality_score = quality_score
        briefing.hmm_regime = hmm_regime
        briefing.price = price

        log.info(
            "analysis_complete",
            ticker=ticker,
            conclusion=briefing.conclusion,
            confidence=briefing.confidence_in_conclusion,
            total_time_sec=round(context.collection_time_sec + briefing.analysis_time_sec, 1),
        )

        return briefing
