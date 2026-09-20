import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
from backend.services.anomaly_service import AnomalyService

@pytest.fixture
def anomaly_service():
    service = AnomalyService()
    service.load_model()
    return service

class TestAnomalyService:
    def test_detect_anomaly_structure(self, anomaly_service):
        res = anomaly_service.detect_anomaly(
            year=2017,
            state_val="Punjab",
            district="Ludhiana",
            area=250.0,
            yield_val=3800.0,
            production=950.0
        )
        assert isinstance(res['is_anomaly'], bool)
        assert 0.0 <= res['anomaly_score'] <= 100.0
        assert res['severity'] in ["LOW", "MODERATE", "HIGH", "EXTREME"]
        assert len(res['reasons']) >= 1
        assert res['state'] == "Punjab"

    def test_extreme_outlier_yield_flagged(self, anomaly_service):
        # Very low yield (200 kg/ha vs ~4000 kg/ha in Punjab)
        res = anomaly_service.detect_anomaly(
            year=2017,
            state_val="Punjab",
            district="Ludhiana",
            area=250.0,
            yield_val=200.0
        )
        assert res['yield_z_score'] is not None
        assert res['yield_z_score'] < -2.0 # Substantial negative deviation
        assert any("deviates" in r.lower() or "outlier" in r.lower() for r in res['reasons'])

    def test_small_acreage_reason_included(self, anomaly_service):
        res = anomaly_service.detect_anomaly(
            year=2017,
            state_val="Kerala",
            area=0.5, # 500 ha
            yield_val=2200.0
        )
        assert any("small acreage" in r.lower() for r in res['reasons'])

    def test_dataset_anomalies_feed(self, anomaly_service):
        anomalies = anomaly_service.get_dataset_anomalies(limit=10)
        assert len(anomalies) <= 10
        assert len(anomalies) > 0
        for a in anomalies:
            assert 'state' in a
            assert 'district' in a
            assert 'anomaly_score' in a
            assert 0.0 <= a['anomaly_score'] <= 100.0
            assert 'severity' in a
            assert 'reason' in a
