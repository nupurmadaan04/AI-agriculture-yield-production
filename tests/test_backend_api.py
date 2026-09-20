import os
import sys
import pytest

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

class TestBackendApi:
    def test_health_endpoint(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["dataset_loaded"] is True
        assert data["records"] == 2469

    def test_summary_endpoint(self, client):
        response = client.get("/api/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["total_records"] == 2469
        assert data["total_states"] == 20
        assert data["total_districts"] == 311
        assert data["min_year"] == 2010
        assert data["max_year"] == 2017
        assert data["average_yield"] > 0
        assert data["missing_values"] == 0
        assert data["duplicate_rows"] == 0

    def test_filters_endpoint(self, client):
        response = client.get("/api/filters")
        assert response.status_code == 200
        data = response.json()
        assert len(data["years"]) == 8
        assert 2017 in data["years"]
        assert len(data["states"]) == 20
        assert "Punjab" in data["states"]
        assert len(data["districts"]) == 311
        assert "Rice" in data["crops"]

    def test_records_endpoint_pagination(self, client):
        response = client.get("/api/records?page=1&page_size=10")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 10
        assert data["pagination"]["page"] == 1
        assert data["pagination"]["page_size"] == 10
        assert data["pagination"]["total"] == 2469
        assert data["pagination"]["total_pages"] == 247

    def test_records_endpoint_filtering(self, client):
        response = client.get("/api/records?state=Punjab&year=2017")
        assert response.status_code == 200
        data = response.json()
        for r in data["data"]:
            assert r["state"].lower() == "punjab"
            assert r["year"] == 2017

    def test_records_endpoint_search(self, client):
        response = client.get("/api/records?search=Ludhiana")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) > 0
        for r in data["data"]:
            assert "ludhiana" in r["district"].lower() or "ludhiana" in r["state"].lower()

    def test_trends_endpoint(self, client):
        response = client.get("/api/trends")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 8  # 2010 to 2017
        years = [t["year"] for t in data["data"]]
        assert sorted(years) == years

    def test_states_endpoint(self, client):
        response = client.get("/api/states")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 20
        # Ranks must be sequential 1..20
        ranks = [s["rank"] for s in data["data"]]
        assert ranks == list(range(1, 21))

    def test_districts_endpoint(self, client):
        response = client.get("/api/districts?state=Punjab&year=2017")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) > 0
        for d in data["data"]:
            assert d["state"] == "Punjab"
            assert d["year"] == 2017

    def test_model_metrics_endpoint(self, client):
        response = client.get("/api/model-metrics")
        assert response.status_code == 200
        data = response.json()
        assert len(data["leaderboard"]) >= 1
        # Check deterministic baseline is present
        baseline = data["leaderboard"][0]
        assert baseline["id"] == "det-baseline"
        assert baseline["model_type"] == "deterministic"
        assert len(data["ablation_experiments"]) == 4

    def test_deterministic_estimate_endpoint(self, client):
        payload = {"area": 250.0, "production": 1000.0}
        response = client.post("/api/estimate/deterministic", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["estimated_yield"] == 4000.0
        assert data["type"] == "deterministic"

    def test_deterministic_estimate_zero_area_fails(self, client):
        payload = {"area": 0.0, "production": 1000.0}
        response = client.post("/api/estimate/deterministic", json=payload)
        assert response.status_code == 422 or response.status_code == 400
