"""
Decision Audit & Cryptographic Certification Engine.

Issues immutable, deterministic SHA-256 audit certificates (DEC-xxxx)
binding decision context, dataset version, model metadata, evidence IDs,
scenarios, explanations, and method limitations.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import threading
from typing import Dict, Any, List, Optional


class DecisionAuditLogger:
    """
    Thread-safe cryptographic audit logger for agricultural decision briefs.
    """

    def __init__(self):
        self._audit_store: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def generate_decision_id(
        self,
        context: Dict[str, Any],
        dataset_version: str,
        model_version: str,
        evidence_ids: List[str],
        scenario_ids: List[str],
        explanation_ids: List[str]
    ) -> str:
        """
        Generates a deterministic SHA-256 hash formatted as DEC-{hash}.
        """
        canonical_dict = {
            "crop": str(context.get("crop", "Rice")).strip().lower(),
            "state": str(context.get("state", "Punjab")).strip().lower(),
            "district": str(context.get("district", "all")).strip().lower() if context.get("district") else "all",
            "year": int(context.get("year", 2017)),
            "decision_horizon": str(context.get("decision_horizon", "next_season")).strip().lower(),
            "dataset_version": str(dataset_version).strip(),
            "model_version": str(model_version).strip(),
            "evidence_ids": sorted(list(set(evidence_ids))),
            "scenario_ids": sorted(list(set(scenario_ids))),
            "explanation_ids": sorted(list(set(explanation_ids)))
        }

        canonical_json = json.dumps(canonical_dict, sort_keys=True, separators=(',', ':'))
        sha256_hash = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
        return f"DEC-{sha256_hash[:10].upper()}"

    def create_audit_record(
        self,
        context: Dict[str, Any],
        dataset_version: str,
        model_version: str,
        evidence_ids: List[str],
        scenario_ids: List[str],
        explanation_ids: List[str],
        brief_summary: Dict[str, Any],
        limitations: List[str]
    ) -> Dict[str, Any]:
        """
        Builds and registers an immutable decision audit certificate.
        """
        decision_id = self.generate_decision_id(
            context=context,
            dataset_version=dataset_version,
            model_version=model_version,
            evidence_ids=evidence_ids,
            scenario_ids=scenario_ids,
            explanation_ids=explanation_ids
        )

        record = {
            "decision_id": decision_id,
            "certificate": f"CERT-{decision_id}",
            "context": context,
            "dataset_version": dataset_version,
            "model_version": model_version,
            "evidence_count": len(evidence_ids),
            "evidence_ids": sorted(list(set(evidence_ids))),
            "scenario_count": len(scenario_ids),
            "scenario_ids": sorted(list(set(scenario_ids))),
            "explanation_count": len(explanation_ids),
            "explanation_ids": sorted(list(set(explanation_ids))),
            "brief_summary": brief_summary,
            "limitations": limitations,
            "methodology_version": "3.2.0",
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "audit_disclaimer": (
                "Audit certificate provides computational traceability and reproducibility. "
                "It does not certify agronomic correctness or biological guarantee."
            )
        }

        with self._lock:
            self._audit_store[decision_id] = record

        return record

    def get_audit_record(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a registered audit record by Decision ID."""
        with self._lock:
            return self._audit_store.get(decision_id)

    def get_recent_records(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns recent audit certificates."""
        with self._lock:
            records = list(self._audit_store.values())
            records.sort(key=lambda r: r.get("generated_at", ""), reverse=True)
            return records[:limit]


decision_audit_logger = DecisionAuditLogger()
