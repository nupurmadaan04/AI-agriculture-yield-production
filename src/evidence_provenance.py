"""
Evidence Provenance & Traceability DAG Engine.

Builds structured provenance records linking decisions and analytical statements
to specific evidence IDs, source computational modules, model versions,
dataset versions, input features, and scientific methodologies.
"""

from __future__ import annotations

import datetime
from typing import Dict, Any, List, Optional


class EvidenceProvenanceBuilder:
    """
    Constructs a Directed Acyclic Graph (DAG) of analytical evidence provenance.
    """

    def __init__(
        self,
        dataset_version: str = "ICRISAT 1966-2017",
        methodology_version: str = "3.2.0"
    ):
        self.dataset_version = dataset_version
        self.methodology_version = methodology_version

    def build_statement_provenance(
        self,
        statement_id: str,
        statement_text: str,
        evidence_ids: List[str],
        source_modules: List[str],
        model_versions: Optional[Dict[str, str]] = None,
        methodology_refs: Optional[List[str]] = None,
        feature_inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Builds a single provenance node for an analytical statement.
        """
        return {
            "statement_id": statement_id,
            "statement": statement_text,
            "evidence_ids": list(set(evidence_ids)),
            "source_modules": list(set(source_modules)),
            "dataset_version": self.dataset_version,
            "model_versions": model_versions or {"forecaster": "exogenous_rf_forecaster v2.1.0"},
            "methodology_refs": methodology_refs or ["ICRISAT Empirical Panel Aggregation", "Out-of-Time Chronological Validation"],
            "feature_inputs": feature_inputs or {},
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

    def build_graph(
        self,
        context: Dict[str, Any],
        evidence_items: List[Dict[str, Any]],
        statements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Builds a complete multi-node provenance graph for the decision brief.
        """
        nodes = []
        edges = []

        # 1. Dataset Root Node
        nodes.append({
            "id": "dataset_root",
            "type": "DATASET",
            "label": f"Dataset: {self.dataset_version}",
            "metadata": {
                "records": 2469,
                "coverage": "1966-2017 ICRISAT Panel",
                "features_tracked": 10
            }
        })

        # 2. Model & Module Nodes
        modules_seen = set()
        for ev in evidence_items:
            mod = ev.get("source_module", "core_analytics")
            if mod not in modules_seen:
                modules_seen.add(mod)
                mod_node_id = f"mod_{mod}"
                nodes.append({
                    "id": mod_node_id,
                    "type": "MODULE",
                    "label": f"Module: {mod}",
                    "metadata": {
                        "model_version": ev.get("model_version", "v2.1.0"),
                        "method": ev.get("source_method", "deterministic_eval")
                    }
                })
                edges.append({
                    "from": "dataset_root",
                    "to": mod_node_id,
                    "relation": "FEEDS_INTO"
                })

        # 3. Evidence Nodes
        for ev in evidence_items:
            ev_id = ev.get("evidence_id", "EV-UNKNOWN")
            mod = ev.get("source_module", "core_analytics")
            mod_node_id = f"mod_{mod}"
            nodes.append({
                "id": ev_id,
                "type": "EVIDENCE",
                "label": f"{ev_id} ({ev.get('evidence_type', 'DERIVED')}): {ev.get('category', 'signal')}",
                "metadata": {
                    "statement": ev.get("statement", ""),
                    "value": ev.get("value"),
                    "unit": ev.get("unit", "")
                }
            })
            edges.append({
                "from": mod_node_id,
                "to": ev_id,
                "relation": "PRODUCES"
            })

        # 4. Decision Statement Nodes
        for st in statements:
            st_id = st.get("statement_id", "ST-UNKNOWN")
            nodes.append({
                "id": st_id,
                "type": "DECISION_STATEMENT",
                "label": f"Statement: {st.get('statement', '')[:40]}...",
                "metadata": st
            })
            for ev_ref in st.get("evidence_ids", []):
                edges.append({
                    "from": ev_ref,
                    "to": st_id,
                    "relation": "SUPPORTS"
                })

        return {
            "dataset_version": self.dataset_version,
            "methodology_version": self.methodology_version,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "nodes": nodes,
            "edges": edges,
            "context": context
        }


provenance_builder = EvidenceProvenanceBuilder()
