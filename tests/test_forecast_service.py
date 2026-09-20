import pytest
from backend.services.forecast_service import forecast_service

def test_forecast_service_load():
    metadata = forecast_service.get_benchmark_metadata()
    assert metadata is not None
    assert 'benchmark_results' in metadata
    assert len(metadata['benchmark_results']) >= 4

def test_forecast_region_punjab():
    res = forecast_service.forecast_region(state_val="Punjab", horizons=[1, 2, 3])
    assert res['state'] == "Punjab"
    assert res['latest_observed_year'] == 2017
    assert len(res['forecasts']) == 3
    assert res['forecasts'][0]['forecast_year'] == 2018
    assert res['forecasts'][0]['predicted_yield'] > 3000.0
    assert res['forecasts'][0]['lower_bound_p10'] <= res['forecasts'][0]['predicted_yield']
    assert res['forecasts'][0]['upper_bound_p90'] >= res['forecasts'][0]['predicted_yield']

def test_forecast_region_district():
    res = forecast_service.forecast_region(state_val="Punjab", district="Ludhiana", horizons=[1, 2])
    assert res['district'] == "Ludhiana"
    assert len(res['forecasts']) == 2
    assert res['forecasts'][0]['forecast_year'] == 2018
