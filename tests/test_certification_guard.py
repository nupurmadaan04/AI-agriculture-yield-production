"""
Test suite for Day 24 Certification Guard.
Verifies pre-inference validation, crop checks, geographic checks, and artifact integrity.
"""

import pytest
from src.certification_guard import CertificationGuard


@pytest.fixture
def guard():
    return CertificationGuard()


def test_guard_allows_certified_oilseeds(guard):
    is_allowed, code, reason, meta = guard.validate_request(
        crop="Oilseeds",
        state="Punjab",
        district="Ludhiana",
        forecast_year=2018,
    )
    assert is_allowed is True
    assert code == "CERTIFIED_ALLOW"
    assert meta is not None
    assert meta["certification_status"] == "PRODUCTION_READY"


def test_guard_rejects_unsupported_crop(guard):
    is_allowed, code, reason, meta = guard.validate_request(
        crop="Potato",
        state="Punjab",
        district="Ludhiana",
        forecast_year=2018,
    )
    assert is_allowed is False
    assert code == "UNSUPPORTED_CROP"
    assert meta is None


def test_guard_rejects_unsupported_district(guard):
    is_allowed, code, reason, meta = guard.validate_request(
        crop="Oilseeds",
        state="Punjab",
        district="FictionalDistrict404",
        forecast_year=2018,
    )
    assert is_allowed is False
    assert code == "DISTRICT_UNSUPPORTED"


def test_guard_rejects_empty_inputs(guard):
    is_allowed, code, reason, meta = guard.validate_request(
        crop="",
        state="Punjab",
        district="Ludhiana",
    )
    assert is_allowed is False
    assert code == "INPUT_INCOMPLETE"
