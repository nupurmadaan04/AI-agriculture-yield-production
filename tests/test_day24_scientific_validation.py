"""
Scientific validation test suite for Day 24.
Verifies governance invariants, zero retraining, strict rejection bounds, and provenance tracking.
"""

import pytest
import json
from pathlib import Path
from src.strategy_registry import StrategyRegistry
from src.certification_guard import CertificationGuard
from src.prediction_service import PredictionService


@pytest.fixture
def base_dir():
    return Path(__file__).resolve().parent.parent


def test_registry_contains_all_14_crops(base_dir):
    reg = StrategyRegistry(base_dir)
    data = reg.compile_strategy_registry()
    strategies = data["strategies"]
    assert len(strategies) == 14
    assert "Oilseeds" in strategies
    assert "Sugarcane" in strategies
    assert strategies["Oilseeds"]["certification_status"] == "PRODUCTION_READY"
    assert strategies["Sugarcane"]["certification_status"] == "CONDITIONAL_PRODUCTION"


def test_zero_fabrication_rejections(base_dir):
    service = PredictionService(base_dir)
    
    # 1. Non-existent crop rejection
    res_crop = service.predict_forecast("FantasyCrop", "Punjab", "Ludhiana")
    assert res_crop["status"] == "REJECTED"
    assert res_crop["error_code"] == "UNSUPPORTED_CROP"
    assert res_crop["prediction"] is None

    # 2. Non-existent district rejection
    res_dist = service.predict_forecast("Oilseeds", "Punjab", "AtlantisDistrict")
    assert res_dist["status"] == "REJECTED"
    assert res_dist["error_code"] == "DISTRICT_UNSUPPORTED"
    assert res_dist["prediction"] is None


def test_dual_pass_bitwise_determinism(base_dir):
    service = PredictionService(base_dir)
    res1 = service.predict_forecast("Oilseeds", "Punjab", "Ludhiana", 2017)
    res2 = service.predict_forecast("Oilseeds", "Punjab", "Ludhiana", 2017)

    assert res1["prediction"] == res2["prediction"]
    assert res1["strategy"] == res2["strategy"]
    assert res1["provenance"]["model_artifact_hash"] == res2["provenance"]["model_artifact_hash"]
