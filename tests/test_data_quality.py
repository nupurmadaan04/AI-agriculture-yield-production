import pytest
from src.data_quality_monitor import data_quality_monitor

def test_data_quality_scoring():
    res = data_quality_monitor.audit_dataset()
    assert 0 <= res['overall_quality_score'] <= 100
    assert res['status'] in ['EXCELLENT', 'GOOD', 'NEEDS_REVIEW']
    assert res['records_evaluated'] == 2469
    assert 'sub_scores' in res
    assert 'completeness' in res['sub_scores']
    assert 'validity' in res['sub_scores']
    assert 'consistency' in res['sub_scores']
    assert 'temporal_integrity' in res['sub_scores']
