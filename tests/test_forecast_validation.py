"""
Test suite for Day 24 Forecast Validation.
Verifies multi-crop deterministic validation execution and metadata generation.
"""

import pytest
from pathlib import Path
from src.forecast_validation import ForecastServingValidator


def test_forecast_validation_determinism():
    validator = ForecastServingValidator()
    df = validator.run_validation()
    assert not df.empty
    assert len(df) == 14
    # Verify all crops evaluated deterministically
    assert (df["is_deterministic"] == True).all()
    assert (df["prediction_diff"] == 0.0).all()
