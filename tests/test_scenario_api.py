import pytest

def test_scenario_simulate_api_valid(client):
    payload = {
        "year": 2017,
        "state": "Punjab",
        "district": "Ludhiana",
        "baseline_rice_area": 250.0,
        "scenario_rice_area": 275.0
    }
    response = client.post("/api/scenario/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "Punjab" in str(data)
    assert "disclaimer" in data or "scientific_disclaimer" in data

def test_scenario_simulate_api_day10_archetype(client):
    payload = {
        "state": "Punjab",
        "district": "Ludhiana",
        "horizon": 1,
        "scenario_type": "conservative_improvement",
        "modifications": {
            "rice_area_pct": 5.0
        }
    }
    response = client.post("/api/scenario/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["scenario_id"].startswith("SCN-")
    assert data["baseline_prediction"] > 0
    assert data["scenario_prediction"] > 0
    assert "validation_context" in data

def test_scenario_compare_api(client):
    payload = {
        "state": "Punjab",
        "horizon": 1
    }
    response = client.post("/api/scenario/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["location"] == "Punjab"
    assert len(data["comparison_matrix"]) >= 4

def test_scenario_sensitivity_api(client):
    payload = {
        "state": "Punjab",
        "horizon": 1
    }
    response = client.post("/api/scenario/sensitivity", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["location"] == "Punjab"
    assert len(data["sensitivity_matrix"]) >= 3

def test_scenario_templates_api(client):
    response = client.get("/api/scenario/templates")
    assert response.status_code == 200
    data = response.json()
    assert "archetypes" in data
    assert len(data["archetypes"]) >= 4

def test_scenario_history_and_audit_api(client):
    # 1. Run simulation to create history entry
    sim_res = client.post("/api/scenario/simulate", json={
        "state": "Punjab",
        "horizon": 1,
        "scenario_type": "moderate_improvement"
    })
    assert sim_res.status_code == 200
    scen_id = sim_res.json()["scenario_id"]

    # 2. Get history
    hist_res = client.get("/api/scenario/history")
    assert hist_res.status_code == 200
    hist_data = hist_res.json()
    assert hist_data["total_scenarios"] >= 1

    # 3. Get audit certificate
    audit_res = client.get(f"/api/scenario/{scen_id}/audit")
    assert audit_res.status_code == 200
    audit_data = audit_res.json()
    assert audit_data["scenario_id"] == scen_id
    assert audit_data["is_reproducible"] is True
