"""Tests for RSS provider, Alerter, and Sector Scan."""
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml


# ─── RSS Provider Tests ──────────────────────────────────────────────

class TestRSSFeedsConfig:
    def test_config_exists(self):
        cfg = Path("configs/rss_feeds.yaml")
        assert cfg.exists(), "configs/rss_feeds.yaml must exist"

    def test_config_valid_yaml(self):
        with open("configs/rss_feeds.yaml") as f:
            data = yaml.safe_load(f)
        assert "feeds" in data
        assert "ticker_aliases" in data

    def test_feeds_have_urls(self):
        with open("configs/rss_feeds.yaml") as f:
            data = yaml.safe_load(f)
        for feed in data["feeds"]:
            assert "url" in feed
            assert "name" in feed
            assert feed["url"].startswith("http")

    def test_minimum_working_feeds(self):
        """After cleanup, should have at least 20 working feeds."""
        with open("configs/rss_feeds.yaml") as f:
            data = yaml.safe_load(f)
        assert len(data["feeds"]) >= 20, f"Only {len(data['feeds'])} feeds (need >= 20)"


# ─── Alerter Tests ───────────────────────────────────────────────────

class TestAlerterContract:
    def test_alert_payload_model(self):
        from src.pipeline.alerter import AlertPayload
        payload = AlertPayload(
            ticker="NVDA",
            verdict="BUBBLE",
            quality_score=0.85,
            tc_date="2026-06-28",
            current_price=167.50,
            top_factors=["High quality score", "DS confidence 0.8"]
        )
        assert payload.ticker == "NVDA"
        assert payload.quality_score == 0.85
        msg = payload.format_telegram()
        assert "NVDA" in msg
        assert "BUBBLE" in msg

    def test_telegram_env_vars_missing(self):
        """Should not crash if env vars are missing."""
        from src.pipeline.alerter import TelegramAlerter
        alerter = TelegramAlerter(
            bot_token="test_token",
            chat_id="123456"
        )
        assert alerter.bot_token == "test_token"

    @patch("src.pipeline.alerter.TelegramAlerter._post_message", return_value=True)
    def test_send_alert_calls_telegram_api(self, mock_post):
        from src.pipeline.alerter import TelegramAlerter, AlertPayload
        alerter = TelegramAlerter(bot_token="test", chat_id="123")
        payload = AlertPayload(ticker="BTC", verdict="BUBBLE", quality_score=0.9)
        result = alerter.send_alert(payload)
        assert result is True
        mock_post.assert_called_once()


# ─── Sector Scan Tests ───────────────────────────────────────────────

class TestSectorScanConfig:
    def test_sectors_config_exists(self):
        cfg = Path("configs/sectors.yaml")
        assert cfg.exists(), "configs/sectors.yaml must exist"

    def test_sectors_have_tickers(self):
        with open("configs/sectors.yaml") as f:
            data = yaml.safe_load(f)
        for sector_id, sector in data["sectors"].items():
            assert "tickers" in sector
            assert len(sector["tickers"]) > 0
            for ticker in sector["tickers"]:
                assert isinstance(ticker, str)
                assert len(ticker) > 0

    def test_minimum_sectors(self):
        with open("configs/sectors.yaml") as f:
            data = yaml.safe_load(f)
        assert len(data["sectors"]) >= 5, f"Only {len(data['sectors'])} sectors (need >= 5)"
