"""
Unit tests for src/evidence_provenance.py.
"""

import pytest
from src.evidence_provenance import provenance_builder


def test_provenance_dag_construction():
    context = {"state": "Punjab", "year": 2017}
    ev_items = [{
        "evidence_id": "EV-FORE-0001",
        "category": "forecast",
        "statement": "Model predicts 4150 kg/ha",
        "value": 4150,
        "unit": "kg/ha",
        "source_module": "forecast_service",
        "source_method": "rf",
        "evidence_type": "PREDICTED",
        "confidence_status": "VALIDATED",
        "timestamp": "2026-01-01",
        "model_version": "v2.1.0",
        "dataset_version": "ICRISAT 1966-2017"
    }]

    statements = [{
        "statement_id": "ST-01",
        "statement": "Maintain current targets",
        "evidence_ids": ["EV-FORE-0001"],
        "source_modules": ["forecast_service"]
    }]

    dag = provenance_builder.build_graph(context, ev_items, statements)
    assert isinstance(dag, dict)
    assert dag["total_nodes"] >= 3
    assert dag["total_edges"] >= 2
    assert any(n["type"] == "DATASET" for n in dag["nodes"])
