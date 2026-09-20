"""
Unit tests for src/decision_priority.py.
"""

import pytest
from src.decision_priority import decision_priority_engine
from src.decision_signal_fusion import decision_signal_fusion_engine


def test_priority_ranking():
    signals = [{
        "signal_name": "productivity_trajectory_concern",
        "signal_label": "Declining Yield Trajectory",
        "strength": "HIGH",
        "evidence_count": 4,
        "severity": "HIGH",
        "persistence": "HIGH",
        "supporting_evidence": ["EV-HIST-0001"],
        "interpretation": "Declining yield trend."
    }]

    ev_items = [{
        "evidence_id": "EV-HIST-0001",
        "category": "historical",
        "statement": "Mean yield is 2000 kg/ha",
        "value": 2000,
        "unit": "kg/ha",
        "source_module": "data_loader",
        "source_method": "aggregation",
        "evidence_type": "OBSERVED",
        "confidence_status": "VALIDATED",
        "timestamp": "2026-01-01",
        "model_version": "v2.1.0",
        "dataset_version": "ICRISAT"
    }]

    priorities = decision_priority_engine.evaluate_priorities(
        signals=signals,
        early_warning_severity="HIGH",
        trend_slope=-25.0,
        spatial_zscore=-1.8,
        prediction_spread_kg_ha=380.0,
        model_r2=0.7866,
        data_quality_score=100.0,
        evidence_items=ev_items
    )

    assert isinstance(priorities, list)
    assert len(priorities) >= 2
    assert priorities[0]["priority_rank"] == 1
    assert priorities[0]["priority_level"] == "HIGH"
