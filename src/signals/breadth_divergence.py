"""Cross-Asset Breadth Divergence — classic late-stage bubble signal.

When index makes new highs but fewer stocks participate (narrowing breadth),
it signals distribution: smart money exits, ETF inflows mask it.

Signal: % stocks above 50MA declining while index rises = DIVERGENCE.
"""

from __future__ import annotations

import numpy as np
import structlog

log = structlog.get_logger()


def compute_breadth_divergence(
    sector_tickers: list[str] | None = None,
    index_ticker: str = "^GSPC",
    period: str = "6mo",
) -> dict:
    """Compute breadth divergence between index and sector components.

    Returns:
        pct_above_50ma: what % of stocks are above their 50-day MA
        index_near_high: is index within 5% of 52-week high
        divergence: True if index near high but breadth < 50%
    """
    import yfinance as yf

    if sector_tickers is None:
        sector_tickers = [
            "NVDA",
            "AAPL",
            "MSFT",
            "GOOGL",
            "META",
            "AMZN",
            "TSLA",
            "JPM",
            "BAC",
            "GS",
            "XOM",
            "CVX",
            "UNH",
            "JNJ",
            "PG",
        ]

    try:
        # Index: is it near 52-week high?
        idx = yf.download(index_ticker, period="1y", progress=False, auto_adjust=True)
        if idx.empty:
            return _fallback()

        idx_prices = idx["Close"].dropna().values.flatten()
        idx_current = float(idx_prices[-1])
        idx_52w_high = float(np.max(idx_prices))
        idx_pct_from_high = (idx_current / idx_52w_high - 1) * 100
        index_near_high = idx_pct_from_high > -5  # within 5% of high

        # Breadth: % of stocks above 50-day MA
        above_50ma = 0
        total = 0

        for ticker in sector_tickers:
            try:
                df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
                if df.empty or len(df) < 50:
                    continue
                prices = df["Close"].dropna().values.flatten()
                ma50 = np.mean(prices[-50:])
                current = float(prices[-1])
                total += 1
                if current > ma50:
                    above_50ma += 1
            except Exception:
                continue

        if total == 0:
            return _fallback()

        pct_above = above_50ma / total * 100
        divergence = index_near_high and pct_above < 50

        # Severity
        if divergence and pct_above < 30:
            severity = "SEVERE"
        elif divergence:
            severity = "MODERATE"
        else:
            severity = "NONE"

        log.info(
            "breadth_divergence",
            pct_above_50ma=round(pct_above, 1),
            index_pct_from_high=round(idx_pct_from_high, 1),
            divergence=divergence,
            severity=severity,
        )

        return {
            "pct_above_50ma": round(pct_above, 1),
            "above_50ma_count": above_50ma,
            "total_stocks": total,
            "index_ticker": index_ticker,
            "index_pct_from_high": round(idx_pct_from_high, 1),
            "index_near_high": index_near_high,
            "divergence": divergence,
            "severity": severity,
        }
    except Exception as e:
        log.debug("breadth_divergence_failed", error=str(e))
        return _fallback()


def _fallback() -> dict:
    return {
        "pct_above_50ma": None,
        "divergence": False,
        "severity": "UNKNOWN",
    }
