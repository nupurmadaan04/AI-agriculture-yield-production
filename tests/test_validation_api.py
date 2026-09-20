import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_api_validation_overview():
    res = client.get('/api/validation/overview')
    assert res.status_code == 200
    data = res.json()
    assert 'primary_model' in data
    assert 'metrics' in data
    assert data['metrics']['r2'] > 0.70
    assert len(data['benchmark_comparison']) >= 5

def test_api_validation_states():
    res = client.get('/api/validation/states')
    assert res.status_code == 200
    data = res.json()
    assert 'data' in data
    assert len(data['data']) == 20

def test_api_validation_prediction_scatter():
    res = client.post('/api/validation/prediction')
    assert res.status_code == 200
    data = res.json()
    assert len(data) > 0
    assert 'observed' in data[0]
    assert 'predicted' in data[0]

def test_api_errors_summary():
    res = client.get('/api/errors/summary')
    assert res.status_code == 200
    data = res.json()
    assert 'mean_absolute_error' in data
    assert 'residual_bins' in data

def test_api_calibration_summary():
    res = client.get('/api/calibration/summary')
    assert res.status_code == 200
    data = res.json()
    assert 'calibration_buckets' in data
    assert len(data['calibration_buckets']) == 5

def test_api_drift_overview():
    res = client.get('/api/drift/overview')
    assert res.status_code == 200
    data = res.json()
    assert 'overall_status' in data
    assert len(data['features']) > 0

def test_api_data_quality():
    res = client.get('/api/data-quality')
    assert res.status_code == 200
    data = res.json()
    assert 'overall_quality_score' in data
    assert data['overall_quality_score'] >= 90.0

def test_api_model_registry():
    res = client.get('/api/models/registry')
    assert res.status_code == 200
    data = res.json()
    assert data['total_registered_models'] >= 4

def test_api_model_detail():
    res = client.get('/api/models/exogenous_rf_forecaster')
    assert res.status_code == 200
    data = res.json()
    assert data['model_id'] == 'exogenous_rf_forecaster'
