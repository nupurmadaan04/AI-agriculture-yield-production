import pytest

def test_api_forecast_yield_endpoint(client):
    payload = {
        "state": "Punjab",
        "district": "Ludhiana",
        "horizons": [1, 2, 3]
    }
    response = client.post("/api/forecast/yield", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "Punjab"
    assert data["district"] == "Ludhiana"
    assert len(data["forecasts"]) == 3
    assert data["forecasts"][0]["forecast_year"] == 2018

def test_api_forecast_state_get(client):
    response = client.get("/api/forecast/state/Haryana")
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "Haryana"
    assert len(data["forecasts"]) >= 1

def test_api_forecast_district_get(client):
    response = client.get("/api/forecast/district/Ludhiana?state=Punjab")
    assert response.status_code == 200
    data = response.json()
    assert data["district"] == "Ludhiana"
