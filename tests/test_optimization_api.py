import pytest

def test_optimization_api_solve_default(client):
    payload = {
        "state": "Punjab",
        "horizon": 1,
        "weights": {
            "yield_improvement": 0.40,
            "risk_reduction": 0.25,
            "resource_efficiency": 0.20,
            "model_reliability": 0.15
        },
        "constraints": {
            "max_risk_score": 60.0
        }
    }
    response = client.post("/api/scenario/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["recommended_scenario"] is not None
    assert len(data["pareto_alternatives"]) >= 1
    assert data["optimization_method"] == "Weighted Linear Scalarization + Pareto Filtering"

def test_optimization_api_custom_weights(client):
    payload = {
        "state": "Haryana",
        "horizon": 2,
        "weights": {
            "yield_improvement": 0.70,
            "risk_reduction": 0.10,
            "resource_efficiency": 0.10,
            "model_reliability": 0.10
        },
        "constraints": {
            "min_yield": 2500.0
        }
    }
    response = client.post("/api/scenario/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["location"] == "Haryana"
    assert data["horizon"] == 2
