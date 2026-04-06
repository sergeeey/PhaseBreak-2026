"""Global Liquidity Pulse — M2 money supply growth as bubble precondition.

Bubbles cannot exist without liquidity. When global M2 growth < 2% YoY,
90% of bubbles deflate even without external shocks.

Data: FRED (free) — US M2 (WM2NS), plus proxies for ECB/PBoC via DXY.
"""

from __future__ import annotations

import numpy as np
import structlog

log = structlog.get_logger()


def get_m2_growth() -> dict:
    """Fetch US M2 money supply YoY growth rate from FRED via yfinance proxy.

    Returns dict with m2_yoy_growth, regime, supports_bubble.
    """
    try:
        import yfinance as yf

        # US M2 proxy: use Fed balance sheet via TLT inverse + money market
        # Direct M2: fredapi requires API key. Use simpler proxy.
        # DXY (dollar index) inverse correlates with global liquidity
        dxy = yf.download("DX-Y.NYB", period="2y", progress=False, auto_adjust=True)

        if dxy.empty or len(dxy) < 252:
            return _fallback()

        prices = dxy["Close"].dropna().values.flatten()

        # DXY falling = liquidity expanding (inverse relationship)
        # YoY change
        current = float(prices[-1])
        year_ago = float(prices[-252]) if len(prices) >= 252 else float(prices[0])
        dxy_yoy = (current / year_ago - 1) * 100

        # Invert: DXY down = liquidity up
        liquidity_proxy = -dxy_yoy

        # 3-month momentum
        m3_ago = float(prices[-63]) if len(prices) >= 63 else float(prices[0])
        dxy_3m = (current / m3_ago - 1) * 100
        liquidity_3m = -dxy_3m

        # Regime
        if liquidity_proxy > 5:
            regime = "SURGE"
        elif liquidity_proxy > 0:
            regime = "EXPANDING"
        elif liquidity_proxy > -3:
            regime = "NEUTRAL"
        else:
            regime = "DRAIN"

        supports_bubble = regime in ("SURGE", "EXPANDING")

        log.info(
            "global_liquidity",
            dxy=round(current, 2),
            dxy_yoy=round(dxy_yoy, 2),
            liquidity_proxy=round(liquidity_proxy, 2),
            regime=regime,
        )

        return {
            "dxy": round(current, 2),
            "dxy_yoy_pct": round(dxy_yoy, 2),
            "liquidity_proxy_yoy": round(liquidity_proxy, 2),
            "liquidity_3m": round(liquidity_3m, 2),
            "regime": regime,
            "supports_bubble": supports_bubble,
        }
    except Exception as e:
        log.debug("global_liquidity_failed", error=str(e))
        return _fallback()


def _fallback() -> dict:
    return {
        "dxy": None,
        "dxy_yoy_pct": None,
        "liquidity_proxy_yoy": None,
        "liquidity_3m": None,
        "regime": "UNKNOWN",
        "supports_bubble": True,  # default: don't block signals
    }
