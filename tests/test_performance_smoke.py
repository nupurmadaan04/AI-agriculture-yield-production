"""
Performance Smoke Test Suite for Day 27 Performance Engineering.
Validates latency bounds, cold/warm behavior, concurrent determinism, and governance integrity.
"""

import time
import pytest
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_latency_smoke():
    """Verify health endpoint responds with minimal latency."""
    t0 = time.perf_counter()
    resp = client.get("/health")
    dur_ms = (time.perf_counter() - t0) * 1000.0
    assert resp.status_code == 200
    assert resp.json().get("status") in ("ok", "HEALTHY", "UP")
    assert dur_ms < 200.0, f"Health check took too long: {dur_ms:.2f}ms"


def test_readiness_latency_smoke():
    """Verify readiness check responds rapidly."""
    t0 = time.perf_counter()
    resp = client.get("/ready")
    dur_ms = (time.perf_counter() - t0) * 1000.0
    assert resp.status_code == 200
    assert resp.json().get("status") in ("READY", "ready", "UP")
    assert dur_ms < 250.0, f"Readiness check took too long: {dur_ms:.2f}ms"


def test_strategy_registry_cached_latency():
    """Verify strategy registry is served from in-memory cache without disk reload."""
    # First call (warm)
    client.get("/api/forecast/strategies")
    
    t0 = time.perf_counter()
    resp = client.get("/api/forecast/strategies")
    dur_ms = (time.perf_counter() - t0) * 1000.0
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_strategies"] >= 14
    assert dur_ms < 100.0, f"Cached strategy registry took too long: {dur_ms:.2f}ms"


def test_certification_summary_cached_latency():
    """Verify certification summary is served rapidly from cache."""
    client.get("/api/forecast/certification")
    
    t0 = time.perf_counter()
    resp = client.get("/api/forecast/certification")
    dur_ms = (time.perf_counter() - t0) * 1000.0
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_crops_certified"] == 14
    assert dur_ms < 100.0, f"Cached certification summary took too long: {dur_ms:.2f}ms"


def test_deterministic_forecast_sequential_invariance():
    """Verify identical sequential forecast requests yield exactly 0 delta."""
    payload = {
        "crop": "Oilseeds",
        "state": "Punjab",
        "district": "Ludhiana",
        "forecast_year": 2018,
        "yield_lag_1": 810.5,
        "yield_rolling_3yr_mean": 795.0,
        "area_lag_1": 12.0
    }
    
    predictions = []
    strategies = []
    model_hashes = []
    
    for _ in range(10):
        resp = client.post("/api/forecast/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "SUCCESS"
        predictions.append(body["prediction"])
        strategies.append(body["strategy"])
        model_hashes.append(body["provenance"]["model_artifact_hash"])
        
    assert len(set(predictions)) == 1, f"Sequential predictions varied: {set(predictions)}"
    assert len(set(strategies)) == 1, f"Sequential strategies varied: {set(strategies)}"
    assert len(set(model_hashes)) == 1, f"Sequential model hashes varied: {set(model_hashes)}"


def test_deterministic_forecast_concurrent_invariance():
    """Verify concurrent requests for identical inputs yield identical outputs (delta = 0)."""
    payload = {
        "crop": "Oilseeds",
        "state": "Punjab",
        "district": "Ludhiana",
        "forecast_year": 2018,
        "yield_lag_1": 810.5,
        "yield_rolling_3yr_mean": 795.0,
        "area_lag_1": 12.0
    }
    
    def _run_req(_):
        r = client.post("/api/forecast/predict", json=payload)
        assert r.status_code == 200
        return r.json()

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(_run_req, range(10)))

    preds = [r["prediction"] for r in results]
    strats = [r["strategy"] for r in results]
    hashes = [r["provenance"]["model_artifact_hash"] for r in results]

    assert len(set(preds)) == 1, f"Concurrent predictions differed: {set(preds)}"
    assert len(set(strats)) == 1, f"Concurrent strategies differed: {set(strats)}"
    assert len(set(hashes)) == 1, f"Concurrent model hashes differed: {set(hashes)}"


def test_unsupported_crop_rejection_guard_performance():
    """Verify unsupported crops are rejected with structured status rapidly."""
    payload = {
        "crop": "NonExistentCrop",
        "state": "Punjab",
        "district": "Ludhiana",
        "forecast_year": 2018
    }
    
    t0 = time.perf_counter()
    resp = client.post("/api/forecast/predict", json=payload)
    dur_ms = (time.perf_counter() - t0) * 1000.0
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "REJECTED"
    assert "NOT_SUPPORTED" in body["certification_status"] or body["error_code"] is not None
    assert dur_ms < 150.0, f"Rejection guard took too long: {dur_ms:.2f}ms"
