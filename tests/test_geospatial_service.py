import pytest
from backend.services.geospatial_service import geospatial_service

def test_geospatial_overview():
    res = geospatial_service.get_spatial_overview()
    assert res['total_states_monitored'] == 20
    assert res['total_districts_monitored'] == 311
    assert res['national_average_yield_kg_ha'] > 2000.0
    assert res['clusters_count'] == 4

def test_get_states_spatial():
    states = geospatial_service.get_states_spatial()
    assert len(states) == 20
    pb = next((s for s in states if s['state'] == 'Punjab'), None)
    assert pb is not None
    assert pb['average_yield_kg_ha'] > 3500.0
    assert pb['cluster_id'] == 0

def test_get_state_spatial_profile():
    prof = geospatial_service.get_state_spatial_profile('Punjab')
    assert prof['state'] == 'Punjab'
    assert len(prof['districts']) >= 10
    assert len(prof['forecasts']) == 3
