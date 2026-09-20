"""
Unit & integration tests for operational health telemetry, endpoints, and request tracking.
Verifies Day 28 Observability requirements.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from src.observability_engine import get_observability_engine, ObservabilityEngine

client = TestClient(app)


def test_observability_engine_singleton():
    """Verify singleton instance of ObservabilityEngine."""
    engine1 = get_observability_engine()
    engine2 = get_observability_engine()
    assert engine1 is engine2
    assert isinstance(engine1, ObservabilityEngine)


def test_get_system_health_endpoint():
    """Verify /api/observability/health returns actual measured system components."""
    response = client.get("/api/observability/health")
    assert response.status_code == 200
    data = response.json()
    assert "api_status" in data
    assert "readiness_status" in data
    assert "backend_status" in data
    assert "uptime_seconds" in data
    assert data["uptime_seconds"] >= 0
    assert "cpu_percent" in data
    assert "memory_rss_mb" in data
    assert "environment" in data
    assert data["environment"] in ["LOCAL", "CONTAINERIZED", "DEVELOPMENT", "PRODUCTION"]


def test_get_observability_summary_endpoint():
    """Verify /api/observability/summary returns unified operational status."""
    response = client.get("/api/observability/summary")
    assert response.status_code == 200
    data = response.json()
    assert "system_health" in data
    assert "runtime_metrics" in data
    assert "forecast_operations" in data
    assert "model_integrity_status" in data
    assert "dataset_integrity_status" in data
    assert "strategy_registry_status" in data
    assert "active_alerts_count" in data


def test_request_id_propagation_and_middleware():
    """Verify X-Request-ID and X-Response-Time-Ms headers are attached and recorded."""
    custom_id = "test-req-trace-9999"
    response = client.get("/api/observability/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == custom_id
    assert "x-response-time-ms" in response.headers


def test_structured_error_recording():
    """Verify operational error recording and classification."""
    engine = get_observability_engine()
    engine.record_request_telemetry(
        request_id="test-err-001",
        method="POST",
        endpoint="/api/forecast/predict",
        status_code=500,
        duration_ms=12.5,
        error_type="INTERNAL_SERVER_ERROR"
    )
    errors_resp = client.get("/api/observability/errors")
    assert errors_resp.status_code == 200
    err_data = errors_resp.json()
    assert "recent_events" in err_data
    assert "total_errors_count" in err_data
    assert "errors_by_category" in err_data
    
    found = any(e.get("request_id") == "test-err-001" for e in err_data["recent_events"])
    assert found is True


def test_empty_telemetry_handling():
    """Verify system handles empty initial buffer without fabrication."""
    engine = get_observability_engine()
    metrics = engine.get_runtime_metrics()
    assert metrics["total_requests"] >= 0
    assert metrics["error_rate_pct"] >= 0.0
    assert "p50_latency_ms" in metrics


def test_no_secrets_in_telemetry():
    """Verify that sensitive header values and tokens are never captured in operational telemetry."""
    secret_token = "SUPER_SECRET_BEARER_TOKEN_998877"
    response = client.get(
        "/api/observability/health",
        headers={
            "Authorization": f"Bearer {secret_token}",
            "X-API-Key": "my-secret-key-12345",
            "X-Request-ID": "test-secret-leak-check"
        }
    )
    assert response.status_code == 200
    
    # Inspect logged telemetry in engine
    engine = get_observability_engine()
    telemetry_events = list(engine._request_buffer)
    for t in telemetry_events:
        t_str = str(t)
        assert secret_token not in t_str
        assert "my-secret-key-12345" not in t_str


def test_concurrent_request_telemetry():
    """Verify thread-safe concurrent recording of request telemetry."""
    import concurrent.futures
    engine = get_observability_engine()

    def record_item(idx: int):
        engine.record_request_telemetry(
            request_id=f"test-concurrent-{idx}",
            method="GET",
            endpoint="/api/health",
            status_code=200,
            duration_ms=5.0 + (idx % 10),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(record_item, i) for i in range(50)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    metrics = engine.get_runtime_metrics()
    assert metrics["total_requests"] >= 50


def test_telemetry_retention_and_buffer_bounds():
    """Verify telemetry ring buffers respect maximum size bounds."""
    engine = get_observability_engine()
    assert len(engine._request_buffer) <= engine._max_buffer_size
    assert len(engine._event_buffer) <= 500

