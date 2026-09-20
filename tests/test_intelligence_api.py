import pytest
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

class TestIntelligenceAPI:
    def test_post_intelligence_risk(self, client):
        payload = {
            "year": 2017,
            "state": "Punjab",
            "district": "Ludhiana",
            "predicted_yield": 3400.0,
            "lower_bound": 2900.0,
            "upper_bound": 3900.0,
            "area": 250.0
        }
        res = client.post("/api/intelligence/risk", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert 0.0 <= data['risk_score'] <= 100.0
        assert data['risk_level'] in ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']
        assert "confidence_label" in data
        assert len(data['risk_factors']) >= 1

    def test_post_intelligence_explain(self, client):
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
        res = client.post("/api/intelligence/explain", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "summary" in data
        assert len(data['feature_contributions']) >= 5

    def test_post_intelligence_anomaly(self, client):
        payload = {
            "year": 2017,
            "state": "Punjab",
            "district": "Ludhiana",
            "area": 250.0,
            "yield": 3800.0,
            "production": 950.0
        }
        res = client.post("/api/intelligence/anomaly", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "is_anomaly" in data
        assert "anomaly_score" in data
        assert len(data['reasons']) >= 1

    def test_get_intelligence_dashboard(self, client):
        res = client.get("/api/intelligence/dashboard")
        assert res.status_code == 200
        data = res.json()
        assert data['total_records'] > 2000
        assert "anomalies_detected" in data
        assert len(data['highest_risk_states']) <= 5
        assert len(data['recent_anomalies']) <= 8

    def test_get_state_risk(self, client):
        res = client.get("/api/intelligence/state-risk")
        assert res.status_code == 200
        data = res.json()
        assert len(data['data']) >= 15
        assert all('risk_score' in s for s in data['data'])

    def test_get_anomalies_feed(self, client):
        res = client.get("/api/intelligence/anomalies?limit=15")
        assert res.status_code == 200
        data = res.json()
        assert len(data['data']) <= 15
        assert len(data['data']) > 0

    def test_existing_endpoints_unaffected(self, client):
        # Health
        res_h = client.get("/api/health")
        assert res_h.status_code == 200
        assert res_h.json()["dataset_loaded"] is True

        # Summary
        res_s = client.get("/api/summary")
        assert res_s.status_code == 200

        # Pre-season advanced prediction
        res_p = client.post("/api/predict/pre-season/advanced", json={
            "year": 2017,
            "state": "Punjab",
            "area": 250.0
        })
        assert res_p.status_code == 200
