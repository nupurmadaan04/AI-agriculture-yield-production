import pytest
from src.scenario_engine import scenario_engine
from backend.services.scenario_service import scenario_service

def test_scenario_engine_delta_math():
    res = scenario_engine.compute_deltas(
        baseline_pred=3000.0,
        scenario_pred=3300.0,
        baseline_risk=40.0,
        scenario_risk=35.0,
        baseline_spread=400.0,
        scenario_spread=420.0
    )
    assert res['yield_delta_kg_ha'] == 300.0
    assert res['yield_percent_change'] == 10.0
    assert res['risk_delta'] == -5.0
    assert res['spread_delta_kg_ha'] == 20.0
    assert res['direction'] == 'positive'
    assert res['risk_direction'] == 'decreased'

def test_scenario_engine_changed_features():
    b_feat = {'rice_area': 100.0, 'wheat_area': 50.0}
    s_feat = {'rice_area': 120.0, 'wheat_area': 50.0}
    changed = scenario_engine.identify_changed_features(b_feat, s_feat)
    assert len(changed) == 1
    assert changed[0]['feature_key'] == 'rice_area'
    assert changed[0]['baseline_value'] == 100.0
    assert changed[0]['scenario_value'] == 120.0
    assert changed[0]['percent_change'] == 20.0

def test_scenario_engine_archetypes():
    base = {'RICE AREA (1000 ha)': 100.0, 'RICE_YIELD_LAG1': 2500.0}
    
    # Baseline
    m_base = scenario_engine.get_archetype_modifications('baseline', base)
    assert m_base['RICE AREA (1000 ha)'] == 100.0

    # Conservative (+5%)
    m_cons = scenario_engine.get_archetype_modifications('conservative_improvement', base)
    assert m_cons['RICE AREA (1000 ha)'] == 105.0

    # Moderate (+12% area, +10% lag)
    m_mod = scenario_engine.get_archetype_modifications('moderate_improvement', base)
    assert m_mod['RICE AREA (1000 ha)'] == 112.0
    assert m_mod['RICE_YIELD_LAG1'] == 2750.0

    # Stress (-15%)
    m_stress = scenario_engine.get_archetype_modifications('stress_scenario', base)
    assert m_stress['RICE AREA (1000 ha)'] == 85.0

def test_scenario_engine_unsupported_features():
    supported, unsupported = scenario_engine.validate_modifications({
        'rice_area_pct': 10.0,
        'rainfall_pct': 20.0,
        'fertilizer_pct': 5.0
    })
    assert 'rice_area_pct' in supported
    assert 'rainfall_pct' in unsupported
    assert 'fertilizer_pct' in unsupported

def test_scenario_service_simulation_valid():
    res = scenario_service.simulate_scenario(
        year=2017,
        state_val="Punjab",
        district="Ludhiana",
        baseline_rice_area=250.0,
        scenario_rice_area=275.0
    )
    assert res['state'] == "Punjab"
    assert res['baseline']['predicted_yield'] > 0
    assert res['scenario']['predicted_yield'] > 0
    assert 'delta' in res
    assert 'changed_features' in res
    assert 'disclaimer' in res

def test_scenario_service_invalid_state():
    with pytest.raises(ValueError):
        scenario_service.simulate_scenario(
            year=2017,
            state_val="InvalidStateXYZ",
            district="Ludhiana"
        )
