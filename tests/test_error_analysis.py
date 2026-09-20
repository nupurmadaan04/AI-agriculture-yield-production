import pytest
from src.error_analysis import error_analysis_engine

def test_error_analysis_diagnostics():
    res = error_analysis_engine.analyze_errors()
    assert res['total_test_samples'] > 0
    assert res['mean_absolute_error'] > 0
    assert len(res['residual_bins']) == 10
    assert 'percentiles' in res
    assert 'p50' in res['percentiles']
    assert 'severity_breakdown' in res
    assert res['severity_breakdown']['low_error_count'] > 0
    assert len(res['largest_errors']) <= 10
    assert len(res['state_error_rankings']) == 20
