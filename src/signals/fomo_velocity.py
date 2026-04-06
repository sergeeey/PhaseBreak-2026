"""Retail FOMO Velocity — acceleration of retail search interest.

Not the LEVEL of Google Trends (always high in bubbles), but the
ACCELERATION (jerk) — d³(search)/dt³. When jerk > threshold,
critical mass of retail FOMO is reached.

Uses pytrends (free Google Trends API).
"""

from __future__ import annotations

import numpy as np
import structlog

log = structlog.get_logger()


def compute_fomo_velocity(
    ticker: str,
    keywords: list[str] | None = None,
) -> dict:
    """Compute FOMO velocity for a ticker using Google Trends.

    Returns:
        trend_level: current search interest (0-100)
        velocity: first derivative (acceleration of interest)
        jerk: second derivative (acceleration of acceleration)
        fomo_alert: True if jerk > 2σ historical
    """
    if keywords is None:
        # Map ticker to search terms
        ticker_keywords = {
            "NVDA": ["buy nvidia stock", "nvidia bubble"],
            "TSLA": ["buy tesla stock", "tesla bubble"],
            "BTC-USD": ["buy bitcoin", "bitcoin crash"],
            "GC=F": ["buy gold", "gold bubble"],
            "GME": ["buy gamestop", "gamestop squeeze"],
        }
        clean = ticker.replace("-USD", "").replace("=F", "").replace("^", "")
        keywords = ticker_keywords.get(ticker, [f"buy {clean} stock"])

    try:
        from pytrends.request import TrendReq

        pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
        pytrends.build_payload(keywords[:1], cat=0, timeframe="today 3-m", geo="US")
        df = pytrends.interest_over_time()

        if df.empty:
            return _fallback(ticker)

        values = df.iloc[:, 0].values.astype(float)

        if len(values) < 10:
            return _fallback(ticker)

        # Current level
        level = float(values[-1])

        # Velocity (first derivative, 7-day window)
        velocity = float(np.mean(np.diff(values[-8:])))

        # Jerk (second derivative)
        if len(values) >= 15:
            diffs = np.diff(values)
            jerk = float(np.mean(np.diff(diffs[-8:])))
        else:
            jerk = 0.0

        # Historical std of velocity for anomaly detection
        all_velocities = np.diff(values)
        vel_std = float(np.std(all_velocities)) if len(all_velocities) > 5 else 1.0

        # FOMO alert: jerk > 2σ of velocity distribution
        fomo_alert = abs(jerk) > 2 * vel_std and jerk > 0

        # Percentile of current level vs history
        level_pctile = float(np.sum(values < level) / len(values) * 100)

        log.info(
            "fomo_velocity",
            ticker=ticker,
            keyword=keywords[0],
            level=round(level, 1),
            velocity=round(velocity, 2),
            jerk=round(jerk, 2),
            fomo_alert=fomo_alert,
        )

        return {
            "ticker": ticker,
            "keyword": keywords[0],
            "level": round(level, 1),
            "level_percentile": round(level_pctile, 1),
            "velocity": round(velocity, 2),
            "jerk": round(jerk, 2),
            "vel_std": round(vel_std, 2),
            "fomo_alert": fomo_alert,
            "available": True,
        }
    except ImportError:
        log.debug("pytrends_not_installed", msg="pip install pytrends")
        return _fallback(ticker)
    except Exception as e:
        log.debug("fomo_velocity_failed", ticker=ticker, error=str(e))
        return _fallback(ticker)


def _fallback(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "keyword": None,
        "level": None,
        "velocity": None,
        "jerk": None,
        "fomo_alert": False,
        "available": False,
    }
