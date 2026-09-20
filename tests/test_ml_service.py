import os
import sys
import pytest

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

class TestMLServiceAndEndpoints:
    def test_post_harvest_prediction_success(self, client):
        payload = {
            "year": 2017,
            "state": "Punjab",
            "area": 250.0,
            "production": 1000.0
        }
        response = client.post("/api/predict/post-harvest", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "post-harvest"
        assert data["predicted_yield"] > 0
        assert data["deterministic_yield"] == 4000.0
        assert "feature_dependency_warning" in data
        assert abs(data["predicted_yield"] - 4000.0) < 1000  # RF curve fits the ratio

    def test_post_harvest_with_state_code(self, client):
        payload = {
            "year": 2016,
            "state_code": 9,  # Punjab
            "area": 100.0,
            "production": 380.0
        }
        response = client.post("/api/predict/post-harvest", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "Punjab"
        assert data["state_code"] == 9
        assert data["deterministic_yield"] == 3800.0

    def test_post_harvest_historical_match(self, client):
        # In 2017, Punjab has records (Ludhiana)
        payload = {
            "year": 2017,
            "state": "Punjab",
            "area": 200.0,
            "production": 800.0,
            "district": "Ludhiana"
        }
        response = client.post("/api/predict/post-harvest", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["historical_matched"] is True
        assert data["actual_yield"] is not None
        assert data["ml_error"] is not None
        assert data["deterministic_error"] is not None

    def test_pre_season_prediction_success(self, client):
        payload = {
            "year": 2017,
            "state": "Punjab",
            "area": 250.0
        }
        response = client.post("/api/predict/pre-season", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "pre-season"
        assert data["predicted_yield"] > 0
        assert "validation_metrics" in data
        assert data["validation_metrics"]["temporal_r2"] == 0.6479
        assert data["validation_metrics"]["group_kfold_r2"] == -0.0038
        assert "warning" in data
        assert "Production is excluded" in data["warning"]

    def test_pre_season_rejects_production(self, client):
        # Strict scientific requirement: reject production in pre-season
        payload = {
            "year": 2017,
            "state": "Punjab",
            "area": 250.0,
            "production": 1000.0
        }
        response = client.post("/api/predict/pre-season", json=payload)
        assert response.status_code == 400
        assert "Production input is strictly prohibited" in response.json()["detail"]

    def test_invalid_area_zero_or_negative(self, client):
        payload = {
            "year": 2017,
            "state": "Punjab",
            "area": 0.0,
            "production": 100.0
        }
        response = client.post("/api/predict/post-harvest", json=payload)
        assert response.status_code in [400, 422]

        payload_neg = {
            "year": 2017,
            "state": "Punjab",
            "area": -10.0
        }
        response_neg = client.post("/api/predict/pre-season", json=payload_neg)
        assert response_neg.status_code in [400, 422]

    def test_invalid_state_name_rejected(self, client):
        payload = {
            "year": 2017,
            "state": "Atlantis",
            "area": 100.0
        }
        response = client.post("/api/predict/pre-season", json=payload)
        assert response.status_code == 400
        assert "Unknown state" in response.json()["detail"]

    def test_models_metadata_endpoint(self, client):
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        models = data["models"]
        assert len(models) >= 3
        # Deterministic baseline must be listed
        det = next(m for m in models if m["id"] == "deterministic-baseline")
        assert det["is_post_harvest_only"] is True
        assert det["mae"] == 4.21
        # Pre-season model must be listed
        pre = next(m for m in models if m["id"] == "rf-pre-season")
        assert pre["is_post_harvest_only"] is False
        assert pre["temporal_r2"] == 0.6479

    def test_error_analysis_endpoint(self, client):
        response = client.get("/api/error-analysis")
        assert response.status_code == 200
        data = response.json()
        assert len(data["worst_performing_states"]) > 0
        assert len(data["top_extreme_errors"]) > 0
        assert len(data["yearly_error_stability"]) == 8
