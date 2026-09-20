import pytest
from src.calibration import calibration_engine

def test_calibration_buckets():
    res = calibration_engine.analyze_calibration()
    assert res['total_evaluated_samples'] > 0
    assert 'spread_error_correlation' in res
    assert len(res['calibration_buckets']) == 5
    assert 'scientific_disclaimer' in res
    for b in res['calibration_buckets']:
        assert 'bucket_label' in b
        assert 'sample_count' in b
        assert 'mean_absolute_error_kg_ha' in b
