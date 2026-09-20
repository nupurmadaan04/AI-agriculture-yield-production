"""
Unit Tests for Exogenous Data Ingestion & Source Registry (Day 22).
"""

import os
from pathlib import Path
import pandas as pd
import pytest

from src.exogenous.source_registry import ExogenousSourceRegistry
from src.exogenous.weather_ingestion import WeatherIngestionEngine


@pytest.fixture
def base_dir():
    return Path(__file__).resolve().parent.parent


def test_source_registry_metadata(base_dir):
    registry = ExogenousSourceRegistry(base_dir)
    sources = registry.get_sources()
    assert len(sources) >= 3

    source_ids = [s["source_id"] for s in sources]
    assert "IMD_DISTRICT_MET_SERIES" in source_ids
    assert "NASA_POWER_ERA5_AGROCLIM" in source_ids
    assert "ICRISAT_AGROCLIMATIC_MESONET" in source_ids

    # Check export files
    csv_p, json_p = registry.export_registry()
    assert csv_p.exists()
    assert json_p.exists()


def test_raw_weather_panel_structure(base_dir):
    raw_weather_csv = base_dir / "Datasets" / "raw" / "exogenous" / "raw_district_weather_panel.csv"
    assert raw_weather_csv.exists()

    df = pd.read_csv(raw_weather_csv)
    assert len(df) > 0
    assert "district" in df.columns
    assert "state" in df.columns
    assert "year" in df.columns
    assert "preseason_rainfall_mm" in df.columns
    assert "preseason_temp_mean_c" in df.columns
    assert "preseason_soil_moisture_index" in df.columns
