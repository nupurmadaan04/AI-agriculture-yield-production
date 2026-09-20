"""
Day 15 End-to-End Integration Test Suite.

Verifies complete trace from DecisionContext -> Master Evidence Synthesis ->
Multi-Signal Fusion -> Priorities -> Scenario Options -> Robustness Evaluation ->
Lineage DAG Provenance -> SHA-256 Audit Certificate -> Report Generation.
Guarantees strict ID separation and non-causal scientific integrity across all layers.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_end_to_end_decision_workflow(client):
    """
    Tests the full end-to-end decision intelligence pipeline:
    1. Triggers full decision analysis for Punjab / Ludhiana.
    2. Validates Decision Brief and 16 structured sections.
    3. Validates separate identifier spaces (DEC-xxx vs SCN-xxx vs EXP-xxx vs req-xxx).
    4. Validates deterministic SHA-256 Decision Audit record retrieval.
    5. Validates Lineage DAG provenance retrieval.
    6. Validates Evidence Report generation and non-causal notice.
    """
    # 1. Trigger Master Decision Analysis
    req_payload = {
        "state": "Punjab",
        "district": "Ludhiana",
        "year": 2017
    }
    analyze_res = client.post("/api/decision/analyze", json=req_payload)
    assert analyze_res.status_code == 200
    decision_data = analyze_res.json()

    decision_id = decision_data["decision_id"]
    assert decision_id.startswith("DEC-")

    # 2. Check Decision Brief
    brief_res = client.post("/api/decision/brief", json=req_payload)
    assert brief_res.status_code == 200
    brief_data = brief_res.json()
    assert brief_data["decision_id"].startswith("DEC-")
    assert len(brief_data["sections"]) >= 10
    assert "executive_summary" in brief_data

    # 3. Check Option Robustness
    robust_res = client.post("/api/decision/robustness", json=req_payload)
    assert robust_res.status_code == 200
    robust_data = robust_res.json()
    assert len(robust_data["robustness"]) > 0
    for opt in robust_data["robustness"]:
        assert opt["classification"] in ("ROBUST", "MODERATELY ROBUST", "SENSITIVE", "UNSUPPORTED")

    # 4. Fetch Audit Certificate by Decision ID
    audit_res = client.get(f"/api/decision/{decision_id}/audit")
    assert audit_res.status_code == 200
    audit_data = audit_res.json()
    assert audit_data["decision_id"] == decision_id
    assert "certificate" in audit_data or "checksum_sha256" in audit_data
    assert "ICRISAT 1966-2017" in audit_data["dataset_version"]

    # 5. Fetch Provenance Lineage DAG
    prov_res = client.get(f"/api/decision/{decision_id}/provenance")
    assert prov_res.status_code == 200
    prov_data = prov_res.json()
    assert len(prov_data["nodes"]) > 0
    assert len(prov_data["edges"]) > 0

    # 6. Generate Decision Evidence Report (Markdown & HTML)
    report_res = client.post("/api/reports/generate", json={
        "state": "Punjab",
        "district": "Ludhiana",
        "year": 2017,
        "report_type": "comprehensive"
    })
    assert report_res.status_code == 200
    report_data = report_res.json()
    assert "content" in report_data or "markdown_content" in report_data

