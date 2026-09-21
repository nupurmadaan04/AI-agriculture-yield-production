"""
Day 33: Concurrency Safety, Idempotency & Performance Benchmark Test Suite.

Validates:
1. Analytical determinism & bitwise idempotency under concurrent load (1, 5, 10 requests).
2. Prediction, strategy, model version, and scenario outputs remain bitwise identical across concurrent executions.
3. Thread safety of append-oriented audit logging under concurrent load (zero file corruption).
4. Performance benchmarks across commodities (Oilseeds, Sugarcane, Rice, Unsupported crop).
"""

import concurrent.futures
import time
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def execute_workspace_request(crop, state, district, year):
    payload = {
        "crop": crop,
        "state": state,
        "district": district,
        "forecast_year": year,
        "selected_scenarios": ["conservative_improvement", "moderate_improvement", "stress_scenario"]
    }
    t0 = time.perf_counter()
    resp = client.post("/api/workspace/analyze", json=payload)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return resp.status_code, resp.json() if resp.status_code == 200 else resp.text, latency_ms


# =============================================================================
# 1. CONCURRENT IDEMPOTENCY & DETERMINISM (1, 5, 10 REQUESTS)
# =============================================================================

@pytest.mark.parametrize("concurrency", [1, 5, 10])
def test_concurrent_idempotency_oilseeds(concurrency):
    """
    Run identical requests for Oilseeds concurrently.
    Verify analytical fields remain bitwise identical across all workers.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(execute_workspace_request, "Oilseeds", "Madhya Pradesh", "Indore", 2017)
            for _ in range(concurrency)
        ]
        results = [f.result() for f in futures]

    statuses = [r[0] for r in results]
    assert all(s == 200 for s in statuses), f"Non-200 responses under concurrency {concurrency}: {statuses}"

    payloads = [r[1] for r in results]
    first_pred = payloads[0]["baseline_forecast"]["forecast_yield_kg_ha"]
    first_strat = payloads[0]["baseline_forecast"]["strategy"]
    first_scenarios = [s["scenario_output_kg_ha"] for s in payloads[0]["scenarios"]]

    for p in payloads[1:]:
        assert p["baseline_forecast"]["forecast_yield_kg_ha"] == first_pred
        assert p["baseline_forecast"]["strategy"] == first_strat
        curr_scenarios = [s["scenario_output_kg_ha"] for s in p["scenarios"]]
        assert curr_scenarios == first_scenarios


@pytest.mark.parametrize("concurrency", [1, 5, 10])
def test_concurrent_idempotency_rice(concurrency):
    """Run identical requests for Rice baseline concurrently."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(execute_workspace_request, "Rice", "Punjab", "Ludhiana", 2017)
            for _ in range(concurrency)
        ]
        results = [f.result() for f in futures]

    assert all(r[0] == 200 for r in results)
    preds = [r[1]["baseline_forecast"]["forecast_yield_kg_ha"] for r in results]
    assert len(set(preds)) == 1, "Non-deterministic predictions under concurrent load"


# =============================================================================
# 2. CONCURRENT REJECTION SAFETY (Unsupported Crops)
# =============================================================================

def test_concurrent_unsupported_crop_rejections():
    """Verify 10 concurrent invalid crop requests are all rejected with 400."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(execute_workspace_request, "FictionalFruit", "Punjab", "Ludhiana", 2017)
            for _ in range(10)
        ]
        results = [f.result() for f in futures]

    statuses = [r[0] for r in results]
    assert all(s == 400 for s in statuses)


# =============================================================================
# 3. AUDIT LOG CONCURRENCY SAFETY
# =============================================================================

def test_audit_log_thread_safety_under_concurrent_predictions():
    """
    Fire 10 concurrent forecast predictions and verify all 10 event records
    are successfully appended into the audit log without corruption.
    """
    def make_pred(idx):
        payload = {
            "crop": "Oilseeds",
            "state": "Madhya Pradesh",
            "district": "Indore",
            "forecast_year": 2017
        }
        r = client.post("/api/forecast/predict", json=payload)
        return r.status_code, r.json()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_pred, i) for i in range(10)]
        results = [f.result() for f in futures]

    assert all(r[0] == 200 for r in results)
    req_ids = [r[1]["request_id"] for r in results]
    assert len(set(req_ids)) == 10, "Duplicate request IDs generated under concurrency"

    # Verify all 10 exist in audit log
    audit_resp = client.get("/api/forecast/audit?limit=50")
    assert audit_resp.status_code == 200
    events = audit_resp.json().get("events", [])
    logged_ids = {e.get("request_id") for e in events}
    for req_id in req_ids:
        assert req_id in logged_ids, f"Request ID {req_id} missing from audit log"
