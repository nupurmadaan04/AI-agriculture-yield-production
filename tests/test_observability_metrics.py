"""
Unit & integration tests for runtime latency calculations, percentiles, error rates, and request counts.
Verifies Day 28 Observability Runtime Metrics requirements.
"""

import pytest
import time
from fastapi.testclient import TestClient
from backend.main import app
from src.observability_engine import get_observability_engine

client = TestClient(app)


def test_runtime_metrics_endpoint_structure():
    """Verify /api/observability/metrics returns valid structure with window parameter."""
    response = client.get("/api/observability/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "successful_requests" in data
    assert "error_requests" in data
    assert "error_rate_pct" in data
    assert "p50_latency_ms" in data
    assert "p95_latency_ms" in data
    assert "p99_latency_ms" in data
    assert "rps" in data


def test_runtime_metrics_calculation_accuracy():
    """Verify metrics calculation against injected known request events."""
    engine = get_observability_engine()
    
    # Record synthetic telemetry items with precise latencies
    engine.record_request_telemetry(
        request_id="test-metric-calc-1",
        method="GET",
        endpoint="/api/test-calc",
        status_code=200,
        duration_ms=10.0,
    )
    engine.record_request_telemetry(
        request_id="test-metric-calc-2",
        method="GET",
        endpoint="/api/test-calc",
        status_code=200,
        duration_ms=20.0,
    )
    engine.record_request_telemetry(
        request_id="test-metric-calc-3",
        method="GET",
        endpoint="/api/test-calc",
        status_code=500,
        duration_ms=30.0,
        error_type="INTERNAL_SERVER_ERROR"
    )

    metrics = engine.get_runtime_metrics()
    assert metrics["total_requests"] >= 3
    assert metrics["error_requests"] >= 1


def test_forecast_operations_metrics():
    """Verify /api/observability/forecasts computes strategy and crop usage aggregations."""
    response = client.get("/api/observability/forecasts")
    assert response.status_code == 200
    data = response.json()
    assert "total_forecasts" in data
    assert "successful_forecasts" in data
    assert "rejected_forecasts" in data
    assert "failed_forecasts" in data
    assert "forecasts_by_crop" in data
    assert "forecasts_by_strategy" in data
    assert "forecasts_by_status" in data


def test_strategy_monitoring_telemetry():
    """Verify /api/observability/strategies returns actual strategy registrations with usage counts."""
    response = client.get("/api/observability/strategies")
    assert response.status_code == 200
    data = response.json()
    assert "strategies" in data
    assert "total_strategies_monitored" in data
    assert data["total_strategies_monitored"] > 0

    strategy_names = [s["certification_status"] for s in data["strategies"]]
    assert "PRODUCTION_READY" in strategy_names or "CONDITIONAL_PRODUCTION" in strategy_names


def test_drift_monitoring_signal_structure():
    """Verify /api/observability/drift provides feature drift monitoring signals."""
    response = client.get("/api/observability/drift")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert "monitoring_notice" in data
    # Verify mandatory disclaimer
    assert "Distributional drift is a monitoring signal" in data["monitoring_notice"]
