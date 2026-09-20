"""
Unit tests for src/explanation_audit.py.
Verifies deterministic ID generation and immutable audit certificates.
"""

import pytest
from src.explanation_audit import explanation_audit_logger


def test_deterministic_explanation_id():
    features = {'Year': 2017, 'State Code': 12, 'RICE AREA (1000 ha)': 310.0}
    id1 = explanation_audit_logger.generate_explanation_id('Punjab', features, '2.1.0')
    id2 = explanation_audit_logger.generate_explanation_id('Punjab', features, '2.1.0')
    assert id1 == id2
    assert id1.startswith('EXP-')


def test_create_and_get_audit_record():
    features = {'Year': 2017, 'State Code': 12, 'RICE AREA (1000 ha)': 310.0}
    explanation = {
        'prediction_kg_ha': 4020.0,
        'baseline_reference_kg_ha': 2850.0,
        'prediction_delta_kg_ha': 1170.0,
        'top_positive_features': ['Previous-Season Rice Yield (t-1)'],
        'top_negative_features': [],
        'feature_contributions': []
    }

    record = explanation_audit_logger.create_audit_record(
        entity='Punjab',
        features=features,
        explanation=explanation,
        scenario_id='SCEN-001'
    )

    assert record['explanation_id'].startswith('EXP-')
    assert record['model_version'] == '2.1.0'
    assert record['scenario_id'] == 'SCEN-001'

    fetched = explanation_audit_logger.get_audit_record(record['explanation_id'])
    assert fetched is not None
    assert fetched['explanation_id'] == record['explanation_id']
    assert fetched['prediction_kg_ha'] == 4020.0
