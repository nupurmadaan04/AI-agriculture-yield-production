import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from backend.services.risk_service import (
    RiskService,
    WEIGHT_UNCERTAINTY,
    WEIGHT_HIST_DEVIATION,
    WEIGHT_MODEL_ERROR,
    WEIGHT_ANOMALY
)

@pytest.fixture
def risk_service():
    return RiskService()

class TestRiskService:
    def test_weights_sum_to_one(self):
        total = WEIGHT_UNCERTAINTY + WEIGHT_HIST_DEVIATION + WEIGHT_MODEL_ERROR + WEIGHT_ANOMALY
        assert abs(total - 1.0) < 1e-5

    def test_risk_score_in_bounds(self, risk_service):
        res = risk_service.assess_risk(
            predicted_yield=3200.0,
            lower_bound=2800.0,
            upper_bound=3600.0,
            year=2017,
            state_val="Punjab",
            district="Ludhiana",
            area=250.0
        )
        assert 0.0 <= res['risk_score'] <= 100.0
        assert res['risk_level'] in ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']
        assert res['uncertainty_percent'] > 0
        assert res['confidence_label'] in ["Low prediction spread", "Moderate prediction spread", "High prediction spread"]
        assert len(res['risk_factors']) >= 1

    def test_zero_or_negative_yield_handled_safely(self, risk_service):
        res = risk_service.assess_risk(predicted_yield=0.0)
        assert res['risk_score'] >= 0.0
        assert res['risk_level'] in ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']

    def test_missing_bounds_handled_gracefully(self, risk_service):
        res = risk_service.assess_risk(
            predicted_yield=2500.0,
            lower_bound=None,
            upper_bound=None
        )
        assert 0.0 <= res['risk_score'] <= 100.0
        assert res['uncertainty_percent'] == 25.0

    def test_state_risk_analytics(self, risk_service):
        state_risks = risk_service.get_state_risk_analytics()
        assert len(state_risks) >= 15
        for s in state_risks:
            assert 'state' in s
            assert 'risk_score' in s
            assert 0.0 <= s['risk_score'] <= 100.0
            assert s['risk_level'] in ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']
            assert s['average_yield'] > 0

    def test_intelligence_dashboard_summary(self, risk_service):
        dash = risk_service.get_intelligence_dashboard_summary()
        assert dash['total_records'] > 2000
        assert dash['anomalies_detected'] >= 0
        assert len(dash['highest_risk_states']) <= 5
        assert len(dash['recent_anomalies']) <= 8
