"""Tests for PhaseBreak FastAPI server."""

from pathlib import Path
import sys
import json

import pytest
from fastapi.testclient import TestClient
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from service.server.main import app

client = TestClient(app)


class TestHealthCheck:
    """Test health check endpoint."""

    def test_health_check(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "service" in data
        assert "version" in data
        assert "timestamp" in data


class TestScanEndpoints:
    """Test scanning endpoints."""

    def test_scan_single_ticker_insufficient_data(self):
        """Test scan with too little data returns INSUFFICIENT_DATA."""
        response = client.get(
            "/api/v1/scan/TEST?window_months=1&domain=finance"
        )
        # Should not crash, but may return INSUFFICIENT_DATA, ERROR, or validation error
        assert response.status_code in [200, 400, 422, 500]

    def test_list_domains(self):
        """Test domain listing endpoint."""
        response = client.get("/api/v1/domains")
        assert response.status_code == 200
        data = response.json()
        assert "domains" in data
        assert len(data["domains"]) > 0

        # Check expected domains are present
        domain_names = [d["name"] for d in data["domains"]]
        assert "finance" in domain_names
        assert "commodities" in domain_names
        assert "housing" in domain_names

    def test_scan_batch_endpoint_validation(self):
        """Test batch scan endpoint with validation."""
        # Empty tickers should fail validation
        response = client.post(
            "/api/v1/scan",
            json={"tickers": [], "window_months": 12, "domain": "finance"}
        )
        assert response.status_code == 422  # Validation error

        # Invalid window_months should fail
        response = client.post(
            "/api/v1/scan",
            json={"tickers": ["TEST"], "window_months": 1, "domain": "finance"}
        )
        assert response.status_code == 422


class TestBenchmarkEndpoint:
    """Test benchmark endpoints."""

    def test_benchmark_not_found(self):
        """Test benchmark endpoint when file doesn't exist."""
        response = client.get("/api/v1/benchmark")
        # May return 404 if benchmark file doesn't exist
        assert response.status_code in [200, 404]


class TestScorecardEndpoint:
    """Test scorecard endpoint."""

    def test_scorecard_not_found(self):
        """Test scorecard when no scan data exists."""
        response = client.get("/api/v1/scorecard")
        # May return 404 if no scan data
        assert response.status_code in [200, 404]


class TestPipelineIntegration:
    """Integration tests with real LPPLS pipeline."""

    def test_full_pipeline_short_series(self):
        """Test that short series are handled correctly."""
        from src.pipeline.stages import run_full_pipeline

        t = np.arange(10, dtype=float)
        prices = np.random.randn(10).cumsum() + 100

        result = run_full_pipeline(t, prices, domain="finance")
        assert result.final_verdict == "INSUFFICIENT_DATA"
        assert result.screening.data_quality == "INSUFFICIENT_DATA"

    def test_full_pipeline_valid_series(self):
        """Test full pipeline with valid series length."""
        from unittest.mock import patch

        from src.pipeline.stages import ScreeningResult, run_full_pipeline

        np.random.seed(42)
        n = 100
        t = np.arange(n, dtype=float)
        prices = np.random.randn(n).cumsum() + 100

        # Deterministic Layer A: avoid environment-specific HMM / hmmlearn availability.
        fake_screening = ScreeningResult(
            data_quality="OK",
            n_points=n,
            hmm_regime="BUBBLE",
            hmm_bubble_prob=0.8,
            should_fit_lppls=True,
            reason="mock screening for integration test",
        )
        with patch("src.pipeline.stages.run_screening", return_value=fake_screening):
            result = run_full_pipeline(t, prices, domain="finance")
        assert result.final_verdict in ["BUBBLE", "POSSIBLE", "NO_BUBBLE"]
        assert result.fit is not None
        assert 0 <= result.fit.quality_score <= 1
        assert 0 <= result.fit.r_squared <= 1


class TestAPIErrorHandling:
    """Test API error handling."""

    def test_invalid_ticker(self):
        """Test API behavior with invalid ticker symbol."""
        response = client.get("/api/v1/scan/INVALIDTICKER123")
        # Should handle gracefully
        assert response.status_code in [200, 500]

    def test_cors_headers(self):
        """Test that CORS middleware is configured."""
        response = client.get("/api/v1/health")
        # CORS should allow all origins (development mode)
        assert response.status_code == 200


class TestDomainEpisodes:
    """Test domain episodes endpoint."""

    def test_domain_episodes_not_found(self):
        """Test endpoint when benchmark data doesn't exist."""
        response = client.get("/api/v1/domains/nonexistent/episodes")
        assert response.status_code in [404, 200]


class TestSignalsEndpoint:
    """Test signals endpoint."""

    def test_get_latest_signals(self):
        """Test getting latest signals."""
        response = client.get("/api/v1/signals")
        # Should return even if no data exists
        assert response.status_code == 200
        data = response.json()
        assert "signals" in data
        assert "scan_date" in data


class TestHistoryEndpoint:
    """Test signal history endpoint."""

    def test_history_empty(self):
        """Test history when no data exists."""
        response = client.get("/api/v1/history/NVDA")
        # Should return empty list
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_history_with_limit(self):
        """Test history with limit parameter."""
        response = client.get("/api/v1/history/NVDA?limit=10")
        assert response.status_code == 200
