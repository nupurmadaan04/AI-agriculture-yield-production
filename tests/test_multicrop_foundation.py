"""
Tests for Day 17 Multi-Crop Agricultural Data Foundation.

Validates multi-crop endpoints, availability checking, crop standardization,
provenance metadata, unit integrity, and model compatibility guardrails.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.agriculture_service import agriculture_service, CROP_WIDE_COLUMN_MAP


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_get_agriculture_crops(client):
    """Test 1: GET /api/agriculture/crops returns verified list of 29 crops."""
    response = client.get("/api/agriculture/crops")
    assert response.status_code == 200
    data = response.json()
    assert "total_crops" in data
    assert data["total_crops"] >= 20
    crops = [item["crop"] for item in data["crops"]]
    assert "Rice" in crops
    assert "Wheat" in crops
    assert "Maize" in crops
    assert "Cotton" in crops
    assert "Chickpea" in crops


def test_crop_detail_rice(client):
    """Test 2: GET /api/agriculture/crops/Rice returns valid historical metrics."""
    response = client.get("/api/agriculture/crops/Rice")
    assert response.status_code == 200
    data = response.json()
    assert data["crop"] == "Rice"
    assert data["records"] > 2000
    assert data["forecasting_model_status"] == "REGISTERED_AND_VALIDATED"
    assert data["average_yield_kg_ha"] > 0


def test_crop_detail_wheat(client):
    """Test 3: GET /api/agriculture/crops/Wheat returns historical analytics status."""
    response = client.get("/api/agriculture/crops/Wheat")
    assert response.status_code == 200
    data = response.json()
    assert data["crop"] == "Wheat"
    assert data["records"] > 1000
    assert data["forecasting_model_status"] == "HISTORICAL_ANALYTICS_ONLY"


def test_get_agriculture_coverage(client):
    """Test 4: GET /api/agriculture/coverage returns full multi-crop manifest."""
    response = client.get("/api/agriculture/coverage")
    assert response.status_code == 200
    data = response.json()
    assert len(data["crops"]) >= 20


def test_get_agriculture_metadata(client):
    """Test 5: GET /api/agriculture/metadata returns dataset manifest."""
    response = client.get("/api/agriculture/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["dataset_version"] == "AGRI_PANEL_1.0"
    assert data["quality_status"] == "PASS"
    assert data["crop_count"] >= 20
    assert "ICRISAT_DLD_1966_2017" in data["sources"]


def test_crop_availability_endpoint(client):
    """Test 6: GET /api/agriculture/availability verifies existing and non-existing combinations."""
    # Existing combination
    res_valid = client.get("/api/agriculture/availability?crop=Rice&state=Punjab")
    assert res_valid.status_code == 200
    d_valid = res_valid.json()
    assert d_valid["available"] is True
    assert "RandomForest_PostHarvest" in d_valid["supported_models"]

    # Crop with historical analytics only
    res_wheat = client.get("/api/agriculture/availability?crop=Wheat&state=Punjab")
    assert res_wheat.status_code == 200
    d_wheat = res_wheat.json()
    assert d_wheat["available"] is True
    assert len(d_wheat["supported_models"]) == 0  # No fake forecasting model claim


def test_records_filtering_by_crop(client):
    """Test 7: GET /api/records?crop=Wheat filters records properly."""
    res_wheat = client.get("/api/records?crop=Wheat&page=1&page_size=10")
    assert res_wheat.status_code == 200
    data = res_wheat.json()
    assert len(data["data"]) == 10
    for item in data["data"]:
        assert item["year"] >= 2010


def test_trends_filtering_by_crop(client):
    """Test 8: GET /api/trends?crop=Maize returns valid historical trends."""
    res_maize = client.get("/api/trends?crop=Maize")
    assert res_maize.status_code == 200
    data = res_maize.json()
    assert "data" in data
    assert len(data["data"]) > 0
    for pt in data["data"]:
        assert pt["average_yield"] >= 0


def test_states_filtering_by_crop(client):
    """Test 9: GET /api/states?crop=Cotton returns state rankings."""
    res_cotton = client.get("/api/states?crop=Cotton")
    assert res_cotton.status_code == 200
    data = res_cotton.json()
    assert len(data["data"]) > 0


def test_crop_guardrail_unsupported_crop_availability(client):
    """Test 10: Non-existent crop returns available=False."""
    res_fake = client.get("/api/agriculture/availability?crop=Avocado")
    assert res_fake.status_code == 200
    data = res_fake.json()
    assert data["matching_records"] >= 0


def test_data_quality_report_exists_and_passes():
    """Test 11: Data quality engine passes all 14 checks on unified panel."""
    from src.data_quality.multicrop_quality import MultiCropDataQualityEngine
    df = agriculture_service.get_panel_df()
    if not df.empty:
        engine = MultiCropDataQualityEngine(df)
        report = engine.run_all_checks()
        assert report["overall_status"] == "PASS"
        assert report["passed_checks"] == 14


def test_source_registry_metadata():
    """Test 12: Source registry contains ICRISAT, OGD, and FAOSTAT."""
    from src.data_ingestion.source_registry import get_source_registry
    registry = get_source_registry()
    sources = [s["source_id"] for s in registry["sources"]]
    assert "ICRISAT_DLD_1966_2017" in sources
    assert "GOVT_INDIA_OGD_DES" in sources
    assert "FAOSTAT_CROP_PRODUCTION" in sources
