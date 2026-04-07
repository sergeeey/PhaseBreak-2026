"""PhaseBreak FastAPI Server — API for LPPLS bubble detection.

Usage:
    uvicorn service.server.main:app --reload --port 8000
    # or
    python -m service.server.main

API Endpoints:
    GET  /api/v1/health                 — Health check
    POST /api/v1/scan                   — Scan multiple tickers
    GET  /api/v1/scan/{ticker}          — Scan single ticker
    GET  /api/v1/scorecard              — Get current scorecard
    GET  /api/v1/domains                — List available domains
    GET  /api/v1/domains/{domain}/episodes — Get domain episodes
    GET  /api/v1/history/{ticker}       — Get signal history for ticker
    GET  /api/v1/signals                — Get latest signals (all tickers)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.stages import run_full_pipeline  # noqa: E402

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PhaseBreak API",
    description="LPPLS Phase Transition Detection — REST API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ScanRequest(BaseModel):
    """Request model for batch scanning."""

    tickers: list[str] = Field(..., min_length=1, max_length=50, description="Ticker symbols")
    window_months: int = Field(12, ge=3, le=36, description="Data window in months")
    domain: str = Field("finance", description="Domain: finance, commodities, housing")


class ScanResult(BaseModel):
    """Result for a single ticker scan."""

    ticker: str
    name: str | None
    price: float | None
    verdict: str
    quality_score: float
    tc_date: str | None
    tc_uncertainty: dict[str, float] | None
    hmm_regime: str | None
    r_squared: float
    data_points: int
    scan_date: str


class ScanResponse(BaseModel):
    """Response for batch scan."""

    scan_date: str
    window_months: int
    domain: str
    results: list[ScanResult]
    summary: dict[str, list[str]]


class ScorecardResponse(BaseModel):
    """Scorecard with live predictions."""

    published_date: str
    verification_deadline: str
    bubble_signals: list[dict[str, Any]]
    no_bubble: list[dict[str, Any]]
    full_scan_summary: dict[str, Any]


class SignalHistoryEntry(BaseModel):
    """Single entry in signal history."""

    date: str
    quality: float
    verdict: str
    tc_date: str | None
    r_squared: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "PhaseBreak API",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/v1/scan", response_model=ScanResponse)
def scan_assets(request: ScanRequest):
    """Scan multiple assets for bubble signals.

    This endpoint runs the full PhaseBreak pipeline (HMM screening → LPPLS fit
    → soft scoring → tc uncertainty) on each requested ticker.
    """
    end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=request.window_months * 30)).strftime("%Y-%m-%d")

    results = []
    bubble_tickers = []
    possible_tickers = []
    no_bubble_tickers = []

    for ticker in request.tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty or len(df) < 20:
                no_bubble_tickers.append(ticker)
                results.append(
                    ScanResult(
                        ticker=ticker,
                        name=None,
                        price=None,
                        verdict="INSUFFICIENT_DATA",
                        quality_score=0.0,
                        tc_date=None,
                        tc_uncertainty=None,
                        hmm_regime=None,
                        r_squared=0.0,
                        data_points=len(df),
                        scan_date=end,
                    )
                )
                continue

            # Extract close prices
            if hasattr(df.columns, "levels"):
                prices = df["Close"].iloc[:, 0].dropna().values.astype(float)
            else:
                prices = df["Close"].dropna().values.astype(float)

            if len(prices) < 20:
                no_bubble_tickers.append(ticker)
                results.append(
                    ScanResult(
                        ticker=ticker,
                        name=None,
                        price=float(prices[-1]) if len(prices) > 0 else None,
                        verdict="INSUFFICIENT_DATA",
                        quality_score=0.0,
                        tc_date=None,
                        tc_uncertainty=None,
                        hmm_regime=None,
                        r_squared=0.0,
                        data_points=len(prices),
                        scan_date=end,
                    )
                )
                continue

            # Run pipeline
            t = np.arange(len(prices), dtype=float)
            result = run_full_pipeline(t, prices, domain=request.domain, n_bootstrap=20)

            # Extract tc date
            tc_date_str = None
            tc_uncertainty = None
            if result.fit and result.fit.tc_estimate is not None:
                tc_idx = int(result.fit.tc_estimate)
                if tc_idx < len(df.index):
                    tc_date_str = str(df.index[tc_idx].date())
                else:
                    days_ahead = tc_idx - len(df.index) + 1
                    tc_date_str = str((df.index[-1] + timedelta(days=days_ahead)).date())

                if result.fit.tc_lower is not None and result.fit.tc_upper is not None:
                    tc_uncertainty = {
                        "lower": float(result.fit.tc_lower),
                        "upper": float(result.fit.tc_upper),
                        "width": float(result.fit.tc_upper - result.fit.tc_lower),
                    }

            # Get current price
            current_price = float(prices[-1]) if len(prices) > 0 else None

            scan_result = ScanResult(
                ticker=ticker,
                name=ticker,  # TODO: fetch from yf.Ticker(ticker).info
                price=current_price,
                verdict=result.final_verdict,
                quality_score=float(result.fit.quality_score if result.fit else 0.0),
                tc_date=tc_date_str,
                tc_uncertainty=tc_uncertainty,
                hmm_regime=result.screening.hmm_regime,
                r_squared=float(result.fit.r_squared if result.fit else 0.0),
                data_points=int(result.screening.n_points),
                scan_date=end,
            )

            results.append(scan_result)

            # Categorize
            if result.final_verdict == "BUBBLE":
                bubble_tickers.append(ticker)
            elif result.final_verdict == "POSSIBLE":
                possible_tickers.append(ticker)
            else:
                no_bubble_tickers.append(ticker)

        except Exception as e:
            log.error("scan_error", ticker=ticker, error=str(e))
            raise HTTPException(status_code=500, detail=f"Error scanning {ticker}: {str(e)}")

    return ScanResponse(
        scan_date=end,
        window_months=request.window_months,
        domain=request.domain,
        results=results,
        summary={
            "bubble": bubble_tickers,
            "possible": possible_tickers,
            "no_bubble": no_bubble_tickers,
        },
    )


@app.get("/api/v1/scan/{ticker}", response_model=ScanResult)
def scan_single_ticker(
    ticker: str,
    window_months: int = Query(12, ge=3, le=36),
    domain: str = Query("finance"),
):
    """Scan a single ticker for bubble signals."""
    batch_request = ScanRequest(tickers=[ticker], window_months=window_months, domain=domain)
    batch_response = scan_assets(batch_request)

    if not batch_response.results:
        raise HTTPException(status_code=404, detail="No results for ticker")

    return batch_response.results[0]


@app.get("/api/v1/scorecard", response_model=ScorecardResponse)
def get_scorecard():
    """Get current prediction scorecard.

    Loads the latest saved scorecard from data/ directory.
    """
    data_dir = PROJECT_ROOT / "data"

    # Look for latest scan file
    scan_files = sorted(data_dir.glob("live_scan_*.json"), reverse=True)
    if not scan_files:
        raise HTTPException(status_code=404, detail="No scorecard found. Run a scan first.")

    with open(scan_files[0]) as f:
        scan_data = json.load(f)

    results = scan_data.get("results", [])
    bubble_signals = [
        {
            "ticker": r.get("ticker"),
            "price": r.get("price"),
            "quality": r.get("quality_score"),
            "tc": r.get("tc_date_str"),
        }
        for r in results
        if r.get("verdict") == "BUBBLE"
    ]
    no_bubble_list = [
        {
            "ticker": r.get("ticker"),
            "price": r.get("price"),
            "quality": r.get("quality_score"),
        }
        for r in results
        if r.get("verdict") == "NO_BUBBLE"
    ]

    return ScorecardResponse(
        published_date=scan_data.get("scan_date", "unknown"),
        verification_deadline="",  # TODO: calculate from published_date + 30 days
        bubble_signals=bubble_signals,
        no_bubble=no_bubble_list,
        full_scan_summary={
            "total": len(results),
            "bubble": len(bubble_signals),
            "no_bubble": len(no_bubble_list),
        },
    )


@app.get("/api/v1/domains")
def list_domains() -> dict[str, list[dict[str, Any]]]:
    """List available domains for LPPLS analysis."""
    return {
        "domains": [
            {
                "name": "finance",
                "description": "Stock market indices and individual stocks",
                "data_frequency": "daily",
                "min_points": 50,
            },
            {
                "name": "commodities",
                "description": "Commodity futures (oil, gold, wheat, etc.)",
                "data_frequency": "daily",
                "min_points": 50,
            },
            {
                "name": "housing",
                "description": "Housing price indices (FHFA, Zillow)",
                "data_frequency": "quarterly/monthly",
                "min_points": 20,
            },
            {
                "name": "geology",
                "description": "Geological/satellite temporal series",
                "data_frequency": "irregular",
                "min_points": 20,
            },
            {
                "name": "adversarial",
                "description": "Synthetic control cases for validation",
                "data_frequency": "varies",
                "min_points": 20,
            },
        ]
    }


@app.get("/api/v1/domains/{domain}/episodes")
def get_domain_episodes(domain: str):
    """Get benchmark episodes for a domain."""
    # Load from benchmark results if available
    benchmark_path = PROJECT_ROOT / "data" / "v2_results" / "benchmark_results.json"
    if not benchmark_path.exists():
        raise HTTPException(status_code=404, detail="Benchmark data not found")

    with open(benchmark_path) as f:
        benchmark = json.load(f)

    episodes = [ep for ep in benchmark.get("episodes", []) if ep.get("domain") == domain]
    if not episodes:
        raise HTTPException(status_code=404, detail=f"No episodes for domain: {domain}")

    return {"domain": domain, "episodes": episodes, "count": len(episodes)}


@app.get("/api/v1/history/{ticker}", response_model=list[SignalHistoryEntry])
def get_signal_history(ticker: str, limit: int = Query(30, ge=1, le=365)):
    """Get signal history for a specific ticker."""
    history_path = PROJECT_ROOT / "data" / "signal_history.json"
    if not history_path.exists():
        return []

    with open(history_path) as f:
        history = json.load(f)

    ticker_history = history.get(ticker, [])
    limited_history = ticker_history[-limit:]

    return [
        SignalHistoryEntry(
            date=entry.get("date", ""),
            quality=entry.get("quality", 0.0),
            verdict=entry.get("verdict", ""),
            tc_date=entry.get("tc_date"),
            r_squared=entry.get("r_squared", 0.0),
        )
        for entry in limited_history
    ]


@app.get("/api/v1/signals")
def get_latest_signals():
    """Get latest signals from the most recent scan."""
    data_dir = PROJECT_ROOT / "data"
    scan_files = sorted(data_dir.glob("live_scan_*.json"), reverse=True)

    if not scan_files:
        return {"signals": [], "scan_date": None}

    with open(scan_files[0]) as f:
        scan_data = json.load(f)

    return {
        "scan_date": scan_data.get("scan_date"),
        "signals": scan_data.get("results", []),
    }


@app.get("/api/v1/benchmark")
def get_benchmark_summary():
    """Get benchmark results summary."""
    benchmark_path = PROJECT_ROOT / "data" / "v2_results" / "benchmark_results.json"
    if not benchmark_path.exists():
        raise HTTPException(status_code=404, detail="Benchmark results not found")

    with open(benchmark_path) as f:
        benchmark = json.load(f)

    summary = benchmark.get("summary", {})
    return {
        "total_episodes": summary.get("total_episodes", 0),
        "total_tp": summary.get("total_tp", 0),
        "total_fp": summary.get("total_fp", 0),
        "total_tn": summary.get("total_tn", 0),
        "total_fn": summary.get("total_fn", 0),
        "precision": summary.get("precision", 0.0),
        "recall": summary.get("recall", 0.0),
        "f1": summary.get("f1", 0.0),
    }


# ---------------------------------------------------------------------------
# Run server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "service.server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
