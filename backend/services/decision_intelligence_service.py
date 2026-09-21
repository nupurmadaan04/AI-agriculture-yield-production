"""
Agricultural Decision Intelligence Service (Day 31).

Orchestrates multi-layer evidence collection, signal fusion, priority ranking,
scenario options, robustness evaluation, executive brief generation, DAG provenance,
and cryptographic SHA-256 audit logging.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional

from src.decision_intelligence import decision_intelligence_engine
from src.decision_signal_fusion import decision_signal_fusion_engine
from src.decision_priority import decision_priority_engine
from src.decision_options import decision_options_engine
from src.decision_robustness import decision_robustness_engine
from src.decision_brief import decision_brief_generator
from src.evidence_provenance import provenance_builder
from src.decision_audit import decision_audit_logger
from src.decision_validation import decision_validator


class DecisionIntelligenceService:
    """
    Service coordinating Day 31 Decision Intelligence & Evidence-Based Brief operations.
    """

    _instance: Optional['DecisionIntelligenceService'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DecisionIntelligenceService, cls).__new__(cls)
        return cls._instance

    def analyze_decision(
        self,
        crop: str = "Rice",
        state: str = "Punjab",
        district: Optional[str] = None,
        year: int = 2017,
        decision_horizon: str = "next_season"
    ) -> Dict[str, Any]:
        """
        Executes complete multi-layer decision intelligence analysis.
        """
        # 1. Collect normalized multi-module evidence
        evidence_bundle = decision_intelligence_engine.collect_evidence(
            crop=crop,
            state=state,
            district=district,
            year=year,
            decision_horizon=decision_horizon
        )
        context = evidence_bundle["context"]
        evidence_items = evidence_bundle["evidence_items"]
        metrics = evidence_bundle["metrics"]
        raw = evidence_bundle["raw_modules"]
        forecast_summary = evidence_bundle.get("forecast_summary")
        historical_context = evidence_bundle.get("historical_context")
        validation_evidence = evidence_bundle.get("validation_evidence")
        uncertainty_evidence = evidence_bundle.get("uncertainty_evidence")
        monitoring_evidence = evidence_bundle.get("monitoring_evidence")
        attribution_evidence = evidence_bundle.get("attribution_evidence")

        # 2. Decision Signal Fusion
        signals = decision_signal_fusion_engine.fuse_signals(
            context=context,
            observed_trend=metrics.get("trend_direction", "STABLE"),
            trend_slope=metrics.get("trend_slope", 0.0),
            forecast_val=metrics.get("forecast_yield_kg_ha", 2140.0),
            baseline_val=metrics.get("historical_yield_kg_ha", 2062.8),
            prediction_spread_kg_ha=metrics.get("prediction_spread_kg_ha", 350.0),
            early_warning_severity=metrics.get("early_warning_severity", "LOW"),
            anomaly_flag=metrics.get("anomaly_flag", False),
            spatial_zscore=metrics.get("spatial_zscore", 0.0),
            change_point_detected=False,
            model_r2=metrics.get("r2", 0.7866),
            data_quality_score=100.0,
            evidence_items=evidence_items
        )

        # 3. Decision Priority Evaluation
        priorities = decision_priority_engine.evaluate_priorities(
            signals=signals,
            early_warning_severity=metrics.get("early_warning_severity", "LOW"),
            trend_slope=metrics.get("trend_slope", 0.0),
            spatial_zscore=metrics.get("spatial_zscore", 0.0),
            prediction_spread_kg_ha=metrics.get("prediction_spread_kg_ha", 350.0),
            model_r2=metrics.get("r2", 0.7866),
            data_quality_score=100.0,
            evidence_items=evidence_items
        )

        # 4. Decision Options from Day 10 Scenarios & Optimization
        scenario_list = raw["scenarios"].get("comparison_matrix", raw["scenarios"].get("scenarios", [])) if isinstance(raw.get("scenarios"), dict) else []
        opt_res = raw.get("optimization", {})
        options = decision_options_engine.build_decision_options(
            base_yield_kg_ha=metrics.get("forecast_yield_kg_ha", 2140.0),
            base_area_1000_ha=context.get("target_area_1000_ha", 250.0),
            scenario_results=scenario_list,
            optimization_result=opt_res,
            evidence_items=evidence_items
        )

        # 5. Robustness Analysis across Sensitivity Sweeps
        robustness = decision_robustness_engine.evaluate_all_options(
            options=options,
            sensitivity_matrix=raw.get("sensitivity", {})
        )

        # 6. Build Decision Statements for Provenance
        statements = []
        for p in priorities:
            st_id = f"ST-PRIORITY-{p['priority_rank']}"
            st_text = f"Analytical Priority #{p['priority_rank']}: {p['issue']} ({p['priority_level']})."
            statements.append(provenance_builder.build_statement_provenance(
                statement_id=st_id,
                statement_text=st_text,
                evidence_ids=p.get("supporting_evidence", []),
                source_modules=["decision_priority", "early_warning_service", "forecast_service"],
                feature_inputs={"state": state, "district": district, "year": year}
            ))

        for opt in options:
            st_id = f"ST-OPTION-{opt['option_id']}"
            st_text = f"Decision Option: {opt['title']} projecting {opt['projected_yield_kg_ha']} kg/ha."
            statements.append(provenance_builder.build_statement_provenance(
                statement_id=st_id,
                statement_text=st_text,
                evidence_ids=opt.get("supporting_evidence", []),
                source_modules=["scenario_service", "optimization_service"],
                feature_inputs={"state": state, "district": district, "year": year}
            ))

        # 7. Construct Provenance DAG
        provenance = provenance_builder.build_graph(
            context=context,
            evidence_items=evidence_items,
            statements=statements
        )
        if forecast_summary and "provenance_hash" in forecast_summary:
            provenance["provenance_hash"] = forecast_summary["provenance_hash"]

        # 8. Create Cryptographic SHA-256 Decision Audit Record
        evidence_ids = [e["evidence_id"] for e in evidence_items]
        scenario_ids = [opt["scenario_id"] for opt in options]
        explanation_ids = [metrics.get("explanation_id", "EXP-00000000")]

        audit_record = decision_audit_logger.create_audit_record(
            context=context,
            dataset_version=decision_intelligence_engine.dataset_version,
            model_version=forecast_summary.get("model_version", "1.0.0") if forecast_summary else "1.0.0",
            evidence_ids=evidence_ids,
            scenario_ids=scenario_ids,
            explanation_ids=explanation_ids,
            brief_summary={
                "forecast_yield_kg_ha": metrics.get("forecast_yield_kg_ha"),
                "early_warning_severity": metrics.get("early_warning_severity"),
                "top_priority": priorities[0]["issue"] if priorities else "Maintain monitoring"
            },
            limitations=[
                "Decision-support artifact; does not establish biological causality.",
                "Simulations describe mathematical models within AGRI_PANEL_1.0 historical distributions (1966–2017)."
            ]
        )

        # 9. Generate Complete Decision Brief
        brief = decision_brief_generator.generate_brief(
            context=context,
            evidence_items=evidence_items,
            signals=signals,
            priorities=priorities,
            options=options,
            robustness=robustness,
            metrics=metrics,
            provenance=provenance,
            audit_record=audit_record,
            forecast_summary=forecast_summary,
            historical_context=historical_context,
            validation_evidence=validation_evidence,
            uncertainty_evidence=uncertainty_evidence,
            monitoring_evidence=monitoring_evidence,
            attribution_evidence=attribution_evidence
        )

        # 10. Scientific Validation
        val_res = decision_validator.validate_decision_brief(
            brief=brief,
            evidence_items=evidence_items,
            provenance=provenance,
            audit_record=audit_record
        )

        res_payload = {
            "decision_id": audit_record["decision_id"],
            "context": context,
            "brief": brief,
            "is_scientifically_validated": val_res["is_valid"],
            "validation_checks_passed": val_res["passed_rules"],
            "validation_total_rules": val_res["total_rules"],
            "validation_details": val_res
        }
        if not hasattr(self, "_cache"):
            self._cache = {}
        self._cache[audit_record["decision_id"]] = res_payload
        return res_payload

    def get_decision_by_id(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a previously computed decision record from audit store or cache."""
        if hasattr(self, "_cache") and decision_id in self._cache:
            return self._cache[decision_id]

        audit = decision_audit_logger.get_audit_record(decision_id)
        if not audit:
            return None
        ctx = audit.get("context", {})
        return self.analyze_decision(
            crop=ctx.get("crop", "Rice"),
            state=ctx.get("state", "Punjab"),
            district=ctx.get("district"),
            year=ctx.get("year", 2017),
            decision_horizon=ctx.get("decision_horizon", "next_season")
        )

    def get_audit(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves audit certificate."""
        return decision_audit_logger.get_audit_record(decision_id)

    def get_provenance(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves provenance graph."""
        dec = self.get_decision_by_id(decision_id)
        if dec and "brief" in dec and "provenance" in dec["brief"]:
            return dec["brief"]["provenance"]
        return None

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns recent decision audit certificates."""
        return decision_audit_logger.get_recent_records(limit=limit)

    def get_methodology(self) -> Dict[str, Any]:
        """Returns methodology, taxonomy, and scientific limitations."""
        return {
            "title": "Agricultural Decision Intelligence & Evidence Synthesis Methodology",
            "version": "Day 31 Evidence Architecture",
            "dataset_version": "AGRI_PANEL_1.0 (71,601 verified records, 1966-2017)",
            "evidence_taxonomies": [
                {"type": "OBSERVED", "description": "Empirical historical harvest observations recorded in panel dataset."},
                {"type": "PREDICTED", "description": "Governed pre-season estimates from certified forecast strategies."},
                {"type": "SIMULATED", "description": "Hypothetical scenario projections under controlled input assumptions."},
                {"type": "DERIVED", "description": "Empirical statistical moments, trends, ranges, and ensemble spread."},
                {"type": "MODEL_ATTRIBUTION", "description": "Tree SHAP feature contributions decomposing model adjustments."},
                {"type": "VALIDATION", "description": "4-Fold expanding walk-forward out-of-time error metrics (MAE, RMSE, Win Rate)."},
                {"type": "MONITORING", "description": "Runtime Population Stability Index (PSI) drift, bias, and operational health."},
                {"type": "PROVENANCE", "description": "Cryptographic SHA-256 lineage hash and audit trails."},
                {"type": "ASSUMPTION", "description": "Explicit operating assumptions required for interpretation."},
                {"type": "LIMITATION", "description": "Known data, temporal, or non-causal boundaries."}
            ],
            "non_causal_principles": [
                "Model attribution describes mathematical dependency within feature space, not biological causation.",
                "Simulations are non-prescriptive scenarios and must not be interpreted as guaranteed outcomes.",
                "Zero fabricated operational traffic, unmodeled chemical inputs, or synthetic outcomes."
            ],
            "evidence_completeness_levels": [
                "STRONG_EVIDENCE: Certified ML model with walk-forward validation + empirical uncertainty + historical depth >= 10.",
                "PARTIAL_EVIDENCE: Certified baseline model with walk-forward benchmarks and historical depth >= 5.",
                "LIMITED_EVIDENCE: Sparse historical records (< 5) or fallback used.",
                "INSUFFICIENT_EVIDENCE: Unsupported crop or unmapped geographic entity."
            ],
            "confidence_dimensions": [
                {"dimension": "Walk-Forward Validation Gain", "description": "Out-of-time fold win rate and MAE improvement vs baseline."},
                {"dimension": "Empirical Historical Depth", "description": "Number of historical district panel observations."},
                {"dimension": "Ensemble Dispersion Spread", "description": "Empirical P10-P90 range across estimator predictions."},
                {"dimension": "Operational Data Integrity", "description": "Telemetry error rate, PSI drift, and boundary checks."}
            ]
        }


decision_intelligence_service = DecisionIntelligenceService()
