import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_api_geospatial_overview():
    res = client.get("/api/geospatial/overview")
    assert res.status_code == 200
    data = res.json()
    assert data["total_states_monitored"] == 20
    assert data["total_districts_monitored"] == 311
    assert data["clusters_count"] == 4

def test_api_geospatial_states():
    res = client.get("/api/geospatial/states")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 20
    assert "lat" in data[0]
    assert "lon" in data[0]

def test_api_geospatial_state_profile():
    res = client.get("/api/geospatial/state/Punjab")
    assert res.status_code == 200
    data = res.json()
    assert data["state"] == "Punjab"
    assert len(data["districts"]) > 10

def test_api_geospatial_clusters():
    res = client.get("/api/geospatial/clusters")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 4

def test_api_geospatial_query():
    res = client.post("/api/geospatial/query", json={"state": "Punjab", "risk_threshold": 20.0})
    assert res.status_code == 200
    data = res.json()
    assert data["matched_count"] >= 1
