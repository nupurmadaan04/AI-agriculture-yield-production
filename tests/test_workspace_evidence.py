"""
Day 32 Unit Tests: Workspace Evidence Labeling & Semantic Typing.

Verifies:
- Distinct semantic classifications for every evidence entity
- Preservation of OBSERVED vs DERIVED vs PREDICTED vs SCENARIO
- SHA-256 provenance fingerprint presence
"""

import pytest
from backend.services.decision_workspace_service import decision_workspace_service


def test_semantic_classification_categories():
    """Verify semantic classifications across all sub-contexts."""
    res = decision_workspace_service.analyze_workspace(
        crop="Oilseeds",
        state="Punjab",
        district="Ludhiana",
        forecast_year=2017
    )

    # 1. Baseline forecast -> PREDICTED
    assert res["baseline_forecast"]["semantic_classification"] == "PREDICTED"

    # 2. Historical context -> HISTORICAL_REFERENCE
    assert res["historical_context"]["semantic_classification"] == "HISTORICAL_REFERENCE"

    # 3. Recent harvest points -> OBSERVED
    for pt in res["historical_context"]["recent_observations"]:
        assert pt["semantic_classification"] == "OBSERVED"

    # 4. Validation evidence -> VALIDATION
    assert res["validation"]["semantic_classification"] == "VALIDATION"

    # 5. Uncertainty -> DERIVED
    assert res["uncertainty"]["semantic_classification"] == "DERIVED"

    # 6. Monitoring -> MONITORING
    assert res["monitoring"]["semantic_classification"] == "MONITORING"

    # 7. Model attribution -> MODEL_ATTRIBUTION
    assert res["attribution"]["semantic_classification"] == "MODEL_ATTRIBUTION"

    # 8. Provenance -> PROVENANCE
    assert res["provenance"]["semantic_classification"] == "PROVENANCE"

    # 9. Scenarios -> SCENARIO
    for sc in res["scenarios"]:
        assert sc["evidence_type"] == "SCENARIO"


def test_provenance_fingerprint_integrity():
    """Every workspace response must expose valid cryptographic provenance references."""
    res = decision_workspace_service.analyze_workspace(
        crop="Sugarcane",
        state="Uttar Pradesh",
        district="Meerut",
        forecast_year=2017
    )
    prov = res["provenance"]
    assert "prediction_fingerprint" in prov
    assert len(prov["prediction_fingerprint"]) > 0
    assert "request_id" in prov
    assert prov["request_id"].startswith("REQ-")
    assert "audit_reference" in prov
