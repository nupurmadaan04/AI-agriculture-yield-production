import pytest
from backend.services.spatial_outlier_service import spatial_outlier_service

def test_spatial_outlier_detection_national():
    outliers = spatial_outlier_service.get_spatial_outliers()
    assert isinstance(outliers, list)
    assert len(outliers) > 0
    first = outliers[0]
    assert "state" in first
    assert "district" in first
    assert "within_state_zscore" in first
    assert "reasons" in first
    assert len(first["reasons"]) > 0

def test_spatial_outlier_detection_state_filter():
    outliers = spatial_outlier_service.get_spatial_outliers(state="Punjab")
    for o in outliers:
        assert o["state"].lower() == "punjab"
