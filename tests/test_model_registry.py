import pytest
from backend.services.model_registry_service import model_registry_service

def test_model_registry_entries():
    models = model_registry_service.get_registered_models()
    assert len(models) >= 4
    for m in models:
        assert 'model_id' in m
        assert 'model_name' in m
        assert 'version' in m
        assert 'status' in m
        assert 'features' in m

def test_get_specific_model():
    rf = model_registry_service.get_model('exogenous_rf_forecaster')
    assert rf is not None
    assert rf['is_primary'] is True
    assert rf['version'] == '2.1.0'

def test_get_nonexistent_model():
    m = model_registry_service.get_model('nonexistent_model_123')
    assert m is None
