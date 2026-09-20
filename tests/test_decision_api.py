"""
Integration tests for Day 14 Decision Intelligence REST endpoints in backend/main.py.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_api_decision_analyze():
    resp = client.post("/api/decision/analyze", json={
        "crop": "Rice",
        "state": "Punjab",
        "district": "Ludhiana",
        "year": 2017
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "decision_id" in data
    assert data["decision_id"].startswith("DEC-")
    assert "brief" in data
    assert data["is_scientifically_validated"] is True


def test_api_decision_brief():
    resp = client.post("/api/decision/brief", json={
        "state": "Punjab",
        "year": 2017
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "executive_summary" in data
    assert "sections" in data
    assert len(data["sections"]) == 16


def test_api_decision_options():
    resp = client.post("/api/decision/options", json={
        "state": "Punjab"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "options" in data
    assert "robustness" in data
    assert len(data["options"]) >= 2


def test_api_decision_history():
    resp = client.get("/api/decision/history")
    assert resp.status_code == 200
    data = resp.json()
    assert "records" in data


def test_api_decision_methodology():
    resp = client.get("/api/decision/methodology")
    assert resp.status_code == 200
    data = resp.json()
    assert "evidence_taxonomies" in data
    assert "confidence_dimensions" in data
