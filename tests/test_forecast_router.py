"""
Test suite for Day 24 Forecast Router.
Verifies routing to Production ML, Conditional ML (variance clipping), and Statistical Baselines.
"""

import pytest
from pathlib import Path
from src.forecast_router import ForecastRouter


@pytest.fixture
def router():
    return ForecastRouter()


def test_router_production_ready_oilseeds(router):
    strategy_meta = {
        "crop": "Oilseeds",
        "certification_status": "PRODUCTION_READY",
        "primary_strategy": "Historical ML (RandomForestRegressor)",
        "model_artifact": "oilseeds/model_pipeline.pkl",
    }
    res = router.route_and_predict(
        crop="Oilseeds",
        state="Punjab",
        district="Ludhiana",
        forecast_year=2017,
        strategy_meta=strategy_meta,
    )
    assert res["selected_route"] == "CERTIFIED_MACHINE_LEARNING"
    assert res["prediction"] > 0
    assert res["unit"] == "kg/ha"
    assert res["fallback_used"] is False


def test_router_conditional_production_sugarcane(router):
    strategy_meta = {
        "crop": "Sugarcane",
        "certification_status": "CONDITIONAL_PRODUCTION",
        "primary_strategy": "Historical ML (GradientBoostingRegressor)",
        "model_artifact": "sugarcane/model_pipeline.pkl",
    }
    res = router.route_and_predict(
        crop="Sugarcane",
        state="Uttar Pradesh",
        district="Meerut",
        forecast_year=2017,
        strategy_meta=strategy_meta,
    )
    assert res["selected_route"] == "CONDITIONAL_VARIANCE_CLIPPED_ML"
    assert res["prediction"] > 0
    assert res["unit"] == "kg/ha"


def test_router_baseline_production_rice(router):
    strategy_meta = {
        "crop": "Rice",
        "certification_status": "BASELINE_PRODUCTION",
        "primary_strategy": "Historical District Mean / Persistence",
        "model_artifact": None,
    }
    res = router.route_and_predict(
        crop="Rice",
        state="West Bengal",
        district="Burdwan",
        forecast_year=2017,
        strategy_meta=strategy_meta,
    )
    assert res["selected_route"] == "CERTIFIED_STATISTICAL_BASELINE"
    assert res["prediction"] > 0
    assert res["unit"] == "kg/ha"
