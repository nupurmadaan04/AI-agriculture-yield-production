from backend.services.query_service import query_service

def test_query_service_ranking_intent():
    res = query_service.classify_intent("Which state had the highest rice yield?")
    assert res['intent'] == 'state_ranking'
    assert res['metric'] == 'yield'
    assert res['ascending'] is False

def test_query_service_comparison_intent():
    res = query_service.classify_intent("Compare Punjab and Haryana.")
    assert res['intent'] == 'state_comparison'
    assert 'Punjab' in [res.get('state_1'), res.get('state_2')]
    assert 'Haryana' in [res.get('state_1'), res.get('state_2')]

def test_query_service_risk_intent():
    res = query_service.classify_intent("Why is Kerala high risk?")
    assert res['intent'] == 'risk_analysis'
    assert res['state'] == 'Kerala'

def test_query_service_anomaly_intent():
    res = query_service.classify_intent("Show unusual yield observations and anomalies in 2017.")
    assert res['intent'] == 'anomaly_analysis'
    assert res['year'] == 2017

def test_query_service_district_filter_intent():
    res = query_service.classify_intent("Find districts with yield above 3000 kg/ha in Punjab.")
    assert res['intent'] == 'district_search'
    assert res['state'] == 'Punjab'
    assert res['yield_min'] == 3000.0
