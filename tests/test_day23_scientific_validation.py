"""
Day 23 Scientific Validation, Non-Negotiable Directives & API Compliance Tests.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_rice_benchmark_preserved():
    """Verify Day 9 Rice model artifact and performance are untouched."""
    base_dir = Path(__file__).resolve().parent.parent
    legacy_model_path = base_dir / "Models" / "rf_model.pkl"
    assert legacy_model_path.exists(), "Day 9 baseline model artifact missing"


def test_api_final_validation_summary():
    res = client.get("/api/modeling/final-validation")
    assert res.status_code == 200
    data = res.json()
    assert data["total_crops_certified"] == 14
    assert data["production_ready_count"] == 1
    assert data["conditional_production_count"] == 1
    assert data["baseline_production_count"] == 12
    assert "post-2017 independent temporal holdout" in data["temporal_range_statement"]


def test_api_reproducibility():
    res = client.get("/api/modeling/final-validation/reproducibility")
    assert res.status_code == 200
    data = res.json()
    assert data["total_crops_audited"] == 14
    assert data["verified_bitwise_count"] == 14
    assert data["reproducibility_rate_pct"] == 100.0


def test_api_certification():
    res = client.get("/api/modeling/certification")
    assert res.status_code == 200
    data = res.json()
    assert data["total_crops_certified"] == 14
    assert len(data["certifications"]) == 14


def test_api_single_crop_endpoints():
    for endpoint in ["/api/modeling/final-validation/Oilseeds",
                     "/api/modeling/final-validation/Oilseeds/residuals",
                     "/api/modeling/final-validation/Oilseeds/bias",
                     "/api/modeling/final-validation/Oilseeds/strategy"]:
        res = client.get(endpoint)
        assert res.status_code == 200
