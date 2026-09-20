import pytest
import numpy as np
from src.model_drift import model_drift_engine

def test_calculate_psi_identical():
    data = np.random.normal(100, 15, 500)
    psi = model_drift_engine.calculate_psi(data, data)
    assert psi == pytest.approx(0.0, abs=1e-3)

def test_calculate_psi_shifted():
    ref = np.random.normal(100, 15, 500)
    tgt = np.random.normal(140, 15, 500)
    psi = model_drift_engine.calculate_psi(ref, tgt)
    assert psi > 0.10

def test_detect_drift_agricultural_features():
    res = model_drift_engine.detect_drift()
    assert 'features' in res
    assert len(res['features']) > 0
    assert 'overall_status' in res
    assert res['overall_status'] in ['NORMAL', 'WATCH', 'DRIFT_DETECTED']
    for f in res['features']:
        assert 'psi_score' in f
        assert 'ks_statistic' in f
        assert 'status' in f
