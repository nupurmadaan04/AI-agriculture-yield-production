import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

class TestPreSeasonAdvancedEndpoints:
    def test_advanced_preseason_prediction_success(self, client):
        payload = {
            "year": 2017,
            "state": "Punjab",
            "district": "Ludhiana",
            "area": 250.0,
            "total_cropped_area": 350.0,
            "rice_area_share": 0.71,
            "wheat_area": 120.0,
            "cotton_area": 10.0,
            "sugarcane_area": 15.0,
            "rice_yield_lag1": 4200.0,
            "rice_yield_roll3": 4150.0
        }
        response = client.post("/api/predict/pre-season/advanced", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "pre-season-advanced"
        assert data["predicted_yield"] > 0
        assert "uncertainty" in data
        assert data["uncertainty"]["lower_bound_10th_pct"] <= data["predicted_yield"]
        assert data["uncertainty"]["upper_bound_90th_pct"] >= data["predicted_yield"]
        assert data["uncertainty"]["prediction_spread"] >= 0
        assert len(data["feature_contributions"]) >= 4
        assert data["validation_metrics"]["temporal_r2"] == 0.7785

    def test_advanced_preseason_with_minimal_inputs_autocompletes(self, client):
        # Only year, state, area provided
        payload = {
            "year": 2017,
            "state": "Punjab",
            "district": "Ludhiana",
            "area": 250.0
        }
        response = client.post("/api/predict/pre-season/advanced", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_yield"] > 0
        assert "features_used" in data
        assert data["features_used"]["total_cropped_area"] >= 250.0
        assert data["features_used"]["rice_yield_lag1"] > 0

    def test_advanced_preseason_rejects_production(self, client):
        # Strict scientific constraint: Production is strictly forbidden
        payload = {
            "year": 2017,
            "state": "Punjab",
            "area": 250.0,
            "production": 1000.0
        }
        response = client.post("/api/predict/pre-season/advanced", json=payload)
        assert response.status_code == 400
        assert "Production input is strictly prohibited" in response.json()["detail"]

    def test_advanced_preseason_invalid_area_fails(self, client):
        payload = {
            "year": 2017,
            "state": "Punjab",
            "area": 0.0
        }
        response = client.post("/api/predict/pre-season/advanced", json=payload)
        assert response.status_code in [400, 422]

    def test_advanced_preseason_invalid_state_fails(self, client):
        payload = {
            "year": 2017,
            "state": "Narnia",
            "area": 100.0
        }
        response = client.post("/api/predict/pre-season/advanced", json=payload)
        assert response.status_code == 400
        assert "Unknown state" in response.json()["detail"]

    def test_models_metadata_lists_advanced_exogenous(self, client):
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        models = data["models"]
        exo_model = next((m for m in models if m["id"] == "rf-pre-season-exogenous"), None)
        assert exo_model is not None
        assert exo_model["is_post_harvest_only"] is False
        assert exo_model["temporal_r2"] == 0.7785
        assert exo_model["group_kfold_r2"] == 0.7407
        assert exo_model["mae"] == 268.83
