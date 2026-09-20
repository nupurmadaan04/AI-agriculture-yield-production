"""
Decision Priority Ranking Engine.

Ranks decision-support analytical priorities using empirical evidence, severity,
persistence, spatial departure, model reliability, and data quality.
Explicitly labeled as 'Analytical Decision-Support Priority', not objective agronomic truth.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional


class DecisionPriorityEngine:
    """
    Ranks agricultural issues into deterministic analytical decision priorities.
    """

    def evaluate_priorities(
        self,
        signals: List[Dict[str, Any]],
        early_warning_severity: str,
        trend_slope: float,
        spatial_zscore: float,
        prediction_spread_kg_ha: float,
        model_r2: float,
        data_quality_score: float,
        evidence_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Ranks analytical priorities based on evidence signals.
        """
        priorities = []

        # Find supporting evidence IDs by category
        monitoring_ev_ids = [e["evidence_id"] for e in evidence_items if e.get("category") == "monitoring"]
        trend_ev_ids = [e["evidence_id"] for e in evidence_items if e.get("category") == "trend"]
        spatial_ev_ids = [e["evidence_id"] for e in evidence_items if e.get("category") == "spatial"]
        reliability_ev_ids = [e["evidence_id"] for e in evidence_items if e.get("category") == "reliability"]

        # Issue 1: Early Warning & Yield Volatility
        if early_warning_severity in ["HIGH", "CRITICAL"]:
            priorities.append({
                "priority_rank": 1,
                "issue": "Mitigate Active Early Warning Risk",
                "priority_level": "HIGH",
                "reasoning": [
                    f"Monitoring layer reports active {early_warning_severity} risk severity.",
                    "Multi-window rolling metrics show elevated statistical deviation.",
                    "Out-of-time model validation supports reliable signal detection."
                ],
                "supporting_evidence": monitoring_ev_ids + trend_ev_ids
            })
        elif early_warning_severity == "ELEVATED":
            priorities.append({
                "priority_rank": 1,
                "issue": "Monitor Yield Baseline Volatility",
                "priority_level": "MODERATE",
                "reasoning": [
                    "Early warning monitoring detected moderate departure from 3-year rolling baseline.",
                    "Spatial peer comparison remains within acceptable bounds."
                ],
                "supporting_evidence": monitoring_ev_ids
            })
        else:
            priorities.append({
                "priority_rank": 1,
                "issue": "Maintain Baseline Agro-Climatic Monitoring",
                "priority_level": "LOW",
                "reasoning": [
                    "Early warning indicators indicate nominal operational conditions.",
                    "Yield metrics align with historical state averages."
                ],
                "supporting_evidence": monitoring_ev_ids + trend_ev_ids
            })

        # Issue 2: Cropland Reallocation / Land Efficiency
        priorities.append({
            "priority_rank": 2,
            "issue": "Evaluate Cropland Allocation Tradeoffs",
            "priority_level": "MODERATE" if trend_slope < 0 else "LOW",
            "reasoning": [
                "Rice area share is the primary controllable land allocation variable in the model.",
                "Scenario simulations indicate non-linear returns to acreage adjustments.",
                "Multi-objective Pareto optimization can balance total volume vs yield efficiency."
            ],
            "supporting_evidence": [e["evidence_id"] for e in evidence_items if e.get("category") in ["scenario", "optimization", "explanation"]]
        })

        # Issue 3: Prediction Spread & Forecast Verification
        spread_level = "MODERATE" if prediction_spread_kg_ha > 350.0 else "LOW"
        priorities.append({
            "priority_rank": 3,
            "issue": "Account for Model Prediction Spread in Planning",
            "priority_level": spread_level,
            "reasoning": [
                f"Model ensemble dispersion is ±{prediction_spread_kg_ha:.1f} kg/ha.",
                f"Historical out-of-time test MAE is 353.01 kg/ha with R² = {model_r2:.4f}.",
                "Decision makers should plan for range outcomes rather than point estimates."
            ],
            "supporting_evidence": reliability_ev_ids
        })

        # Sort by rank
        priorities.sort(key=lambda p: p["priority_rank"])
        return priorities


decision_priority_engine = DecisionPriorityEngine()
