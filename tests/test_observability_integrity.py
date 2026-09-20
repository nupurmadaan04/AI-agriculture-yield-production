"""
Unit & integration tests for model cryptographic integrity, dataset integrity, and strategy registry verification.
Verifies Day 28 Observability Integrity requirements.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from src.observability_engine import get_observability_engine

client = TestClient(app)


def test_model_integrity_verification():
    """Verify /api/observability/models performs live SHA-256 artifact verification."""
    response = client.get("/api/observability/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "overall_integrity_status" in data
    assert "verified_models_count" in data
    assert len(data["models"]) >= 2
    
    for m in data["models"]:
        assert "crop" in m
        assert "algorithm" in m
        assert "file_exists" in m
        assert "actual_sha256" in m
        assert "integrity_status" in m
        assert m["file_exists"] is True
        assert "VERIFIED" in m["integrity_status"]


def test_dataset_integrity_verification():
    """Verify /api/observability/dataset reports canonical dataset metadata without fabrication."""
    response = client.get("/api/observability/dataset")
    assert response.status_code == 200
    data = response.json()
    assert "dataset_version" in data
    assert data["dataset_version"] == "AGRI_PANEL_1.0"
    assert "total_records" in data
    assert data["total_records"] == 71601
    assert "schema_status" in data
    assert "sha256_checksum" in data


def test_strategy_registry_health():
    """Verify /api/observability/registry verifies registry availability, models, and certification guard."""
    response = client.get("/api/observability/registry")
    assert response.status_code == 200
    data = response.json()
    assert "registry_available" in data
    assert data["registry_available"] is True
    assert "certification_guard_status" in data
    assert "total_strategies_registered" in data
    assert data["total_strategies_registered"] >= 14
    assert "production_ready_count" in data
    assert "conditional_production_count" in data
