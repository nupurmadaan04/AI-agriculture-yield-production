"""
Tests for Statistical Drift Monitoring (PSI / KS) and Controlled Synthetic Shifts (Day 30).
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from src.model_drift import ModelDriftEngine

client = TestClient(app)


def test_drift_monitoring_endpoint():
    response = client.get("/api/monitoring/drift")
    assert response.status_code == 200
    data = response.json()
    assert data["overall_drift_status"] in ["STABLE", "MODERATE_DRIFT", "DRIFT_DETECTED", "MONITORING_ONLY"]
    assert len(data["features"]) > 0
    assert len(data["coverage_drift"]) > 0
    assert data["semantic_classification"] == "MONITORING"
    
    for feat in data["features"]:
        assert feat["metric"] == "PSI"
        assert feat["observed_value"] >= 0.0
        assert feat["threshold"] > 0.0
        assert feat["reference_window"] == "2010-2015"
        assert feat["evaluation_window"] == "2016-2017"
        assert feat["reference_samples"] > 0
        assert feat["evaluation_samples"] > 0


def test_drift_controlled_synthetic_invariance():
    # Identical distributions must yield PSI == 0.0
    rng = np.random.RandomState(42)
    reference = rng.normal(loc=100.0, scale=15.0, size=1000)
    target = reference.copy()
    
    psi = ModelDriftEngine.calculate_psi(reference, target, num_bins=10)
    assert abs(psi) < 0.01, f"Expected near-zero PSI for identical distribution, got {psi}"


def test_drift_controlled_synthetic_shift():
    # Significantly shifted distribution must yield PSI > 0.25
    rng = np.random.RandomState(42)
    reference = rng.normal(loc=100.0, scale=10.0, size=1000)
    target = rng.normal(loc=150.0, scale=10.0, size=1000)
    
    psi = ModelDriftEngine.calculate_psi(reference, target, num_bins=10)
    assert psi > 0.25, f"Expected high PSI for shifted distribution, got {psi}"


def test_drift_empty_sample_handling():
    # Empty inputs must safely return 0.0 without crashing
    psi = ModelDriftEngine.calculate_psi(np.array([]), np.array([]))
    assert psi == 0.0
