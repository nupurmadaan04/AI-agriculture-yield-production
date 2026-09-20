import pytest

def test_api_early_warning_assess(client):
    payload = {"state": "Punjab"}
    response = client.post("/api/early-warning/assess", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "Punjab"
    assert "warning_score" in data
    assert "severity" in data

def test_api_early_warning_dashboard(client):
    response = client.get("/api/early-warning/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert data["total_states_monitored"] >= 18
    assert "top_priority_warnings" in data
    assert "state_matrix" in data

def test_api_early_warning_states(client):
    response = client.get("/api/early-warning/states")
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) >= 18
