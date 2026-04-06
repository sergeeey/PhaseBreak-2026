"""Telegram alert dispatcher for PhaseBreak Hybrid 2.0.

Sends alerts when LPPLS detects bubble signals or sector-wide contagion.
Uses httpx POST to Telegram Bot API — no python-telegram-bot dependency.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, Field

log = structlog.get_logger()

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class AlertPayload(BaseModel):
    """Typed contract for what an alert contains."""

    ticker: str
    verdict: str  # BUBBLE | POSSIBLE | SECTOR_ALERT
    quality_score: float = Field(ge=0.0, le=1.0)
    tc_date: str | None = None
    current_price: float | None = None
    domain: str = "finance"
    top_factors: list[str] = Field(default_factory=list)
    sector_name: str | None = None
    sector_tickers: list[str] = Field(default_factory=list)

    def format_telegram(self) -> str:
        """Format as Telegram Markdown message."""
        icon = {"BUBBLE": "\U0001f534", "POSSIBLE": "\U0001f7e1", "SECTOR_ALERT": "\U0001f6a8"}.get(
            self.verdict, "\u2753"
        )

        lines = [f"{icon} *PhaseBreak Alert: {self.ticker}*"]
        lines.append(f"Verdict: *{self.verdict}*")
        lines.append(f"Quality: {self.quality_score:.2f}")

        if self.tc_date:
            lines.append(f"Predicted crash: {self.tc_date}")
        if self.current_price:
            lines.append(f"Price: ${self.current_price:,.2f}")

        if self.top_factors:
            lines.append("\nKey factors:")
            for f in self.top_factors[:3]:
                lines.append(f"  - {f}")

        if self.sector_name:
            lines.append(f"\nSector: *{self.sector_name}*")
            if self.sector_tickers:
                lines.append(f"Alert tickers: {', '.join(self.sector_tickers)}")

        return "\n".join(lines)


class TelegramAlerter:
    """Sends alerts to Telegram via Bot API using httpx."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self.bot_token and self.chat_id)

        if not self._enabled:
            log.warning(
                "telegram_alerter_disabled", reason="missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID"
            )
        else:
            log.info("telegram_alerter_initialized")

    def send_alert(self, payload: AlertPayload) -> bool:
        """Send single alert. Returns True on success."""
        if not self._enabled:
            log.debug("telegram_skip", ticker=payload.ticker, reason="not configured")
            return False
        return self._post_message(payload.format_telegram())

    def send_batch(self, payloads: list[AlertPayload], delay_sec: float = 0.5) -> int:
        """Send multiple alerts with rate limiting. Returns count of successful sends."""
        sent = 0
        for i, p in enumerate(payloads):
            if self.send_alert(p):
                sent += 1
            if i < len(payloads) - 1:
                time.sleep(delay_sec)
        log.info("telegram_batch_complete", sent=sent, total=len(payloads))
        return sent

    def _post_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Low-level Telegram sendMessage call."""
        try:
            url = TELEGRAM_API.format(token=self.bot_token)
            resp = httpx.post(
                url,
                json={"chat_id": self.chat_id, "text": text, "parse_mode": parse_mode},
                timeout=10.0,
            )
            if resp.status_code == 200:
                log.info("telegram_sent", length=len(text))
                return True
            else:
                log.warning("telegram_error", status=resp.status_code, body=resp.text[:200])
                return False
        except Exception as e:
            log.warning("telegram_failed", error=str(e))
            return False


def should_alert(result: dict[str, Any]) -> bool:
    """Determine if a prediction result warrants an alert."""
    verdict = result.get("verdict", "NO_BUBBLE")
    quality = result.get("quality_score", 0.0)
    return verdict in ("BUBBLE", "POSSIBLE") and quality > 0.7


def build_alert_payload(
    result: dict[str, Any],
    rag_briefing: dict[str, Any] | None = None,
) -> AlertPayload:
    """Build AlertPayload from prediction result."""
    factors = []
    if rag_briefing:
        factors.extend(rag_briefing.get("speculative_factors", [])[:2])
        factors.extend(rag_briefing.get("fundamental_factors", [])[:1])

    return AlertPayload(
        ticker=result.get("ticker", "UNKNOWN"),
        verdict=result.get("verdict", "POSSIBLE"),
        quality_score=min(1.0, max(0.0, result.get("quality_score", 0.5))),
        tc_date=result.get("tc_date_str"),
        current_price=result.get("price"),
        domain=result.get("domain", "finance"),
        top_factors=factors[:3],
    )
