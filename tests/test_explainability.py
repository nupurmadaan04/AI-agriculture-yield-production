import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from backend.services.explainability_service import ExplainabilityService

@pytest.fixture
def explain_service():
    return ExplainabilityService()

class TestExplainabilityService:
    def test_explain_prediction_returns_valid_structure(self, explain_service):
        res = explain_service.explain_prediction(
            year=2017,
            state_val="Punjab",
            district="Ludhiana",
            area=250.0,
            total_cropped_area=350.0,
            rice_area_share=0.71,
            wheat_area=120.0,
            cotton_area=10.0,
            sugarcane_area=15.0,
            rice_yield_lag1=4200.0,
            rice_yield_roll3=4150.0
        )
        assert res['predicted_yield'] > 0
        assert "summary" in res
        assert "top_positive_factor" in res
        assert "top_negative_factor" in res
        assert len(res['feature_contributions']) >= 5

        # Contributions check
        total_pct = sum(c['normalized_percentage'] for c in res['feature_contributions'])
        assert abs(total_pct - 100.0) < 1.0

        for c in res['feature_contributions']:
            assert c['direction'] in ['positive', 'negative', 'neutral']
            assert 'feature_name' in c
            assert 'raw_value' in c
            assert c['contribution_score'] >= 0

    def test_explain_with_missing_optional_inputs(self, explain_service):
        res = explain_service.explain_prediction(
            year=2017,
            state_val="Punjab",
            district="Ludhiana",
            area=150.0
        )
        assert res['predicted_yield'] > 0
        assert len(res['feature_contributions']) >= 5
