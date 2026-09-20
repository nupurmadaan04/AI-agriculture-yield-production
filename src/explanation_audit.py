"""
Explanation Audit Trail Engine for Agricultural Decision Intelligence.

Creates and manages immutable, reproducible explanation audit certificates (EXP-xxxx).
Tracks model versions, input vectors, attribution breakdowns, and scenario/alert links.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from typing import Dict, Any, List, Optional


class ExplanationAuditLogger:
    """Manages verifiable explanation audit certificates."""

    _audit_store: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def generate_explanation_id(cls, entity: str, features: Dict[str, Any], model_version: str = "2.1.0") -> str:
        """Generates a deterministic, reproducible hash-based explanation ID."""
        raw_str = f"{entity}_{json.dumps(features, sort_keys=True)}_{model_version}"
        sha = hashlib.sha256(raw_str.encode('utf-8')).hexdigest()[:8].upper()
        return f"EXP-{sha}"

    @classmethod
    def create_audit_record(
        cls,
        entity: str,
        features: Dict[str, Any],
        explanation: Dict[str, Any],
        scenario_id: Optional[str] = None,
        alert_id: Optional[str] = None,
        model_version: str = "2.1.0",
        dataset_version: str = "ICRISAT 1966–2017 Panel"
    ) -> Dict[str, Any]:
        """
        Builds and logs a complete explanation audit certificate.
        """
        exp_id = cls.generate_explanation_id(entity, features, model_version)

        record = {
            'explanation_id': exp_id,
            'timestamp': datetime.utcnow().isoformat() + "Z",
            'model_name': 'Exogenous Random Forest Forecaster',
            'model_version': model_version,
            'dataset_version': dataset_version,
            'entity': entity,
            'scenario_id': scenario_id,
            'alert_id': alert_id,
            'prediction_kg_ha': explanation.get('prediction_kg_ha', 0.0),
            'baseline_reference_kg_ha': explanation.get('baseline_reference_kg_ha', 0.0),
            'prediction_delta_kg_ha': explanation.get('prediction_delta_kg_ha', 0.0),
            'explanation_method': explanation.get('explanation_method', 'Marginal Reference Perturbation Attribution'),
            'input_features': features,
            'top_positive_features': explanation.get('top_positive_features', []),
            'top_negative_features': explanation.get('top_negative_features', []),
            'feature_contributions': explanation.get('feature_contributions', []),
            'limitations': [
                "Explanations describe empirical model response curves, not agronomic physical causality.",
                "Marginal attribution evaluates one-at-a-time departures from median reference values.",
                "Valid strictly for the trained feature distribution of the ICRISAT panel."
            ]
        }

        cls._audit_store[exp_id] = record
        return record

    @classmethod
    def get_audit_record(cls, explanation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a previously generated explanation audit record by ID."""
        return cls._audit_store.get(explanation_id)

    @classmethod
    def list_audit_records(cls, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists recently generated explanation audit certificates."""
        return list(cls._audit_store.values())[-limit:]


explanation_audit_logger = ExplanationAuditLogger()
