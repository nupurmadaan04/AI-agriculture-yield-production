import pytest
from backend.services.geospatial_service import geospatial_service

def test_regional_similarity_search():
    similar = geospatial_service.get_similar_regions(state="Punjab", top_n=5)
    assert len(similar) == 5
    for item in similar:
        assert "state" in item
        assert "district" in item
        assert "similarity_score" in item
        assert -1.0 <= item["similarity_score"] <= 1.0

def test_neighboring_regions_scientific_safeguard():
    res = geospatial_service.get_neighboring_regions(state="Punjab")
    assert res["neighbor_data_available"] is False
    assert "reason" in res
