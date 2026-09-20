import pytest
from backend.services.query_service import query_service
from backend.services.copilot_service import copilot_service

def test_query_service_spatial_intent_routing():
    # Test Cluster analysis intent
    q1 = "Which clusters contain high-risk districts?"
    res1 = query_service.classify_intent(q1)
    assert res1['intent'] == 'cluster_analysis'

    # Test Spatial outlier intent
    q2 = "Which districts in Punjab are spatial outliers?"
    res2 = query_service.classify_intent(q2)
    assert res2['intent'] == 'geographic_outlier'
    assert res2['state'] == 'Punjab'

def test_copilot_spatial_cluster_evidence():
    ans = copilot_service.answer_query("Show me regional spatial clusters")
    assert 'evidence' in ans
    assert 'tools_used' in ans
    assert 'get_spatial_clusters' in ans['tools_used']
    assert len(ans['answer']) > 20

def test_copilot_spatial_outliers_evidence():
    ans = copilot_service.answer_query("Which districts are spatial outliers in Punjab?")
    assert 'evidence' in ans
    assert 'tools_used' in ans
    assert 'get_spatial_outliers' in ans['tools_used']
