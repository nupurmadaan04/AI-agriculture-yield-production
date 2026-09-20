"""
Decision Signal Fusion Engine.

Synthesizes multiple independent analytical signals (temporal trend, forecast deviation,
prediction spread, anomaly status, early warning severity, spatial departure,
change points, model reliability, data quality) into actionable decision-support signals
WITHOUT fabricating ungrounded probabilities.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional


class DecisionSignalFusionEngine:
    """
    Fuses empirical and model signals into transparent decision signals with evidence traceability.
    """

    def fuse_signals(
        self,
        context: Dict[str, Any],
        observed_trend: str,
        trend_slope: float,
        forecast_val: float,
        baseline_val: float,
        prediction_spread_kg_ha: float,
        early_warning_severity: str,
        anomaly_flag: bool,
        spatial_zscore: float,
        change_point_detected: bool,
        model_r2: float,
        data_quality_score: float,
        evidence_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Synthesizes structured decision signals from analytical inputs.
        """
        fused_signals = []

        # Find supporting evidence IDs
        trend_ev_ids = [e["evidence_id"] for e in evidence_items if e.get("category") == "trend"]
        forecast_ev_ids = [e["evidence_id"] for e in evidence_items if e.get("category") == "forecast"]
        monitoring_ev_ids = [e["evidence_id"] for e in evidence_items if e.get("category") == "monitoring"]
        anomaly_ev_ids = [e["evidence_id"] for e in evidence_items if e.get("category") == "anomaly"]
        spatial_ev_ids = [e["evidence_id"] for e in evidence_items if e.get("category") == "spatial"]
        reliability_ev_ids = [e["evidence_id"] for e in evidence_items if e.get("category") == "reliability"]

        forecast_delta = forecast_val - baseline_val

        # 1. Primary Productivity Trajectory Signal
        if observed_trend == "DECREASING" or (trend_slope < -15.0 and forecast_delta < 0):
            strength = "HIGH" if early_warning_severity in ["HIGH", "CRITICAL"] else "MODERATE"
            fused_signals.append({
                "signal_name": "productivity_trajectory_concern",
                "signal_label": "Declining Yield Trajectory",
                "strength": strength,
                "evidence_count": len(trend_ev_ids) + len(forecast_ev_ids),
                "severity": early_warning_severity,
                "persistence": "HIGH" if change_point_detected else "MODERATE",
                "supporting_evidence": trend_ev_ids + forecast_ev_ids + monitoring_ev_ids,
                "interpretation": (
                    f"Historical trend analysis shows a declining slope of {trend_slope:.1f} kg/ha/yr. "
                    f"Forecasting model projects an output of {forecast_val:.1f} kg/ha ({forecast_delta:+.1f} kg/ha vs baseline)."
                )
            })
        else:
            fused_signals.append({
                "signal_name": "productivity_trajectory_stable",
                "signal_label": "Stable Regional Baseline",
                "strength": "MODERATE",
                "evidence_count": len(trend_ev_ids) + len(forecast_ev_ids),
                "severity": "LOW",
                "persistence": "HIGH",
                "supporting_evidence": trend_ev_ids + forecast_ev_ids,
                "interpretation": (
                    f"Yield indicators reflect historical stability ({observed_trend}, slope {trend_slope:+.1f} kg/ha/yr). "
                    f"Model projection remains aligned with regional averages."
                )
            })

        # 2. Early Warning & Anomaly Signal
        if early_warning_severity in ["ELEVATED", "HIGH", "CRITICAL"] or anomaly_flag:
            sev_str = "HIGH" if early_warning_severity in ["HIGH", "CRITICAL"] else "MODERATE"
            fused_signals.append({
                "signal_name": "early_warning_risk_signal",
                "signal_label": f"Active {early_warning_severity} Early Warning Alert",
                "strength": sev_str,
                "evidence_count": len(monitoring_ev_ids) + len(anomaly_ev_ids),
                "severity": early_warning_severity,
                "persistence": "ELEVATED" if anomaly_flag else "MODERATE",
                "supporting_evidence": monitoring_ev_ids + anomaly_ev_ids,
                "interpretation": (
                    f"Monitoring layer flagged an active {early_warning_severity} risk signal. "
                    f"Observation anomaly status: {'Detected' if anomaly_flag else 'Normal'}."
                )
            })

        # 3. Spatial Coherence / Departure Signal
        if abs(spatial_zscore) >= 1.5:
            direction = "above" if spatial_zscore > 0 else "below"
            fused_signals.append({
                "signal_name": "spatial_divergence_signal",
                "signal_label": f"Spatial Departure ({spatial_zscore:+.2f}σ)",
                "strength": "MODERATE",
                "evidence_count": len(spatial_ev_ids),
                "severity": "MODERATE" if spatial_zscore < 0 else "LOW",
                "persistence": "MODERATE",
                "supporting_evidence": spatial_ev_ids,
                "interpretation": (
                    f"District yield deviates by {spatial_zscore:+.2f} standard deviations {direction} state peer median."
                )
            })

        # 4. Model Reliability & Spread Context
        spread_tier = "HIGH" if prediction_spread_kg_ha > 450.0 else ("MODERATE" if prediction_spread_kg_ha > 200.0 else "NARROW")
        fused_signals.append({
            "signal_name": "model_reliability_context",
            "signal_label": f"Model Reliability (R² {model_r2:.3f}, Spread: {spread_tier})",
            "strength": "HIGH",
            "evidence_count": len(reliability_ev_ids),
            "severity": "LOW",
            "persistence": "HIGH",
            "supporting_evidence": reliability_ev_ids,
            "interpretation": (
                f"Registered model validates with R² = {model_r2:.4f}. "
                f"Ensemble prediction spread spans ±{prediction_spread_kg_ha:.1f} kg/ha."
            )
        })

        return fused_signals


decision_signal_fusion_engine = DecisionSignalFusionEngine()
