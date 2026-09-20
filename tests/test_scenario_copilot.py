import pytest
from backend.services.query_service import query_service
from backend.services.copilot_service import copilot_service

def test_query_service_scenario_intents():
    # 1. scenario_simulation intent
    intent_sim = query_service.classify_intent("Simulate what happens if rice area increases by 10% in Punjab")
    assert intent_sim['intent'] in ['scenario_simulation', 'scenario_lab']

    # 2. scenario_comparison intent
    intent_comp = query_service.classify_intent("Compare alternative agricultural scenarios in Punjab")
    assert intent_comp['intent'] == 'scenario_comparison'

    # 3. sensitivity_analysis intent
    intent_sens = query_service.classify_intent("Run sensitivity analysis on agricultural inputs for Punjab")
    assert intent_sens['intent'] == 'sensitivity_analysis'

    # 4. scenario_optimization intent
    intent_opt = query_service.classify_intent("Find optimal strategy balancing yield and risk for Punjab")
    assert intent_opt['intent'] == 'scenario_optimization'

def test_copilot_service_scenario_simulation():
    res = copilot_service.handle_query(
        query="Simulate scenario with 10% rice area increase in Punjab",
        session_id="test-session-scen"
    )
    assert res['status'] == 'success'
    assert 'scenario' in res['intent'] or 'simulat' in res['response'].lower()
    assert 'evidence' in res

def test_copilot_service_sensitivity_analysis():
    res = copilot_service.handle_query(
        query="What is the input elasticity and sensitivity matrix in Punjab?",
        session_id="test-session-sens"
    )
    assert res['status'] == 'success'
    assert 'sensitivity' in res['response'].lower() or 'elasticity' in res['response'].lower()
