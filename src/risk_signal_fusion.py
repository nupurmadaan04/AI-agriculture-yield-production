"""
Multi-Signal Risk Fusion Engine.

Combines independent empirical evidence streams (yield trajectory, deterministic risk,
isolation forest anomalies, forecast divergence, spatial outliers, model prediction error,
and data quality) into an auditable composite alert without collapsing into a fabricated probability.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np
from src.alert_severity import alert_severity_classifier, SEVERITY_WEIGHTS, SeverityTier


class RiskSignalFusion:
    """
    Fuses multiple independent agricultural risk signals and formats evidence chains.
    """

    @staticmethod
    def fuse_signals(
        location: str,
        state: str,
        district: Optional[str],
        year: int,
        signals: List[Dict[str, Any]],
        data_quality_score: float = 100.0,
        model_r2: float = 0.7866,
        model_mae: float = 353.01,
        drift_status: str = "NORMAL"
    ) -> Dict[str, Any]:
        """
        Aggregates active signals into a structured alert with full evidence traceability.
        """
        if not signals:
            return {
                "alert_id": f"ALR-INFO-{abs(hash(location)) % 100000:05d}",
                "location": location,
                "state": state,
                "district": district or "Representative",
                "year": year,
                "severity": SeverityTier.INFO.value,
                "signal_count": 0,
                "dominant_signal": "Normal Baseline Operational State",
                "supporting_signals": [],
                "evidence_strength": "LOW",
                "composite_risk_score": 10.0,
                "evidence_chain": [
                    f"Observation ({year}): Verified within historical variance bounds.",
                    "No anomalous feature shifts or severe spatial departures detected."
                ],
                "recommended_action": "Maintain routine statistical monitoring.",
                "model_validation": {
                    "r2": model_r2,
                    "mae": model_mae,
                    "drift_status": drift_status,
                    "data_quality_score": data_quality_score
                }
            }

        # Calculate weighted severity score
        total_weight = 0.0
        max_severity_tier = SeverityTier.INFO.value
        tier_order = [SeverityTier.INFO.value, SeverityTier.WATCH.value, SeverityTier.ELEVATED.value, SeverityTier.HIGH.value, SeverityTier.CRITICAL.value]

        severity_counts = {t: 0 for t in tier_order}
        for s in signals:
            sev = s.get("severity", SeverityTier.INFO.value).upper()
            if sev in severity_counts:
                severity_counts[sev] += 1
            w = SEVERITY_WEIGHTS.get(sev, 1.0)
            total_weight += w
            if tier_order.index(sev) > tier_order.index(max_severity_tier):
                max_severity_tier = sev

        # Determine composite severity
        if severity_counts[SeverityTier.CRITICAL.value] >= 1 or (severity_counts[SeverityTier.HIGH.value] >= 2 and total_weight >= 15.0):
            composite_sev = SeverityTier.CRITICAL.value
        elif severity_counts[SeverityTier.HIGH.value] >= 1 or (severity_counts[SeverityTier.ELEVATED.value] >= 2 and total_weight >= 10.0):
            composite_sev = SeverityTier.HIGH.value
        elif severity_counts[SeverityTier.ELEVATED.value] >= 1 or (severity_counts[SeverityTier.WATCH.value] >= 2 and total_weight >= 6.0):
            composite_sev = SeverityTier.ELEVATED.value
        elif severity_counts[SeverityTier.WATCH.value] >= 1:
            composite_sev = SeverityTier.WATCH.value
        else:
            composite_sev = SeverityTier.INFO.value

        # Identify dominant signal
        dominant = max(signals, key=lambda s: SEVERITY_WEIGHTS.get(s.get("severity", "INFO").upper(), 1.0))
        dominant_title = f"{dominant.get('signal_type', 'Statistical Deviation')} ({dominant.get('trigger_value', '')} {dominant.get('unit', '')})"

        supporting = [s.get("signal_type") for s in signals if s != dominant]

        # Evidence strength
        sig_count = len(signals)
        if sig_count >= 4 or composite_sev in [SeverityTier.CRITICAL.value, SeverityTier.HIGH.value]:
            evidence_str = "VERY_STRONG" if sig_count >= 4 else "STRONG"
        elif sig_count >= 2:
            evidence_str = "MODERATE"
        else:
            evidence_str = "LOW"

        # Evidence Chain
        chain = [
            f"Observation ({location}, {year}): Analyzed across {sig_count} independent monitoring streams.",
            f"Dominant Signal: {dominant_title} exceeded operational threshold of {dominant.get('threshold')} {dominant.get('unit', '')}.",
            f"Severity Assigned: {composite_sev} based on multi-signal severity synthesis (Weight: {total_weight:.1f})."
        ]
        for s in signals:
            for ev in s.get("evidence", []):
                if ev not in chain:
                    chain.append(ev)

        # Composite score
        comp_score = min(100.0, round(total_weight * 10.0, 1))

        # Recommended Action
        if composite_sev == SeverityTier.CRITICAL.value:
            rec_act = "Immediate agronomic review and multi-period stress simulation recommended."
        elif composite_sev == SeverityTier.HIGH.value:
            rec_act = "Elevated monitoring priority; verify local district yield lag and rainfall indicators."
        elif composite_sev == SeverityTier.ELEVATED.value:
            rec_act = "Track ongoing trajectory in next observation window; review within-state spatial differences."
        elif composite_sev == SeverityTier.WATCH.value:
            rec_act = "Maintain routine watch; model variance remains within acceptable bounds."
        else:
            rec_act = "Normal baseline operations."

        return {
            "alert_id": f"ALR-{abs(hash(location + str(year) + dominant_title)) % 1000000:06d}",
            "location": location,
            "state": state,
            "district": district or "Representative",
            "year": year,
            "severity": composite_sev,
            "signal_count": sig_count,
            "dominant_signal": dominant_title,
            "supporting_signals": supporting,
            "evidence_strength": evidence_str,
            "composite_risk_score": comp_score,
            "evidence_chain": chain,
            "recommended_action": rec_act,
            "model_validation": {
                "r2": model_r2,
                "mae": model_mae,
                "drift_status": drift_status,
                "data_quality_score": data_quality_score
            }
        }


risk_signal_fusion = RiskSignalFusion()
