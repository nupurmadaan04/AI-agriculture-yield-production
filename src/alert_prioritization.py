"""
Alert Prioritization & Ranking Engine.

Ranks alerts according to deterministic multi-attribute criteria:
severity weight, magnitude of deviation, persistence, spatial concentration,
model reliability, and observation recency.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from src.alert_severity import SEVERITY_WEIGHTS, SeverityTier


class AlertPrioritization:
    """
    Ranks and filters early warning alerts transparently.
    """

    @staticmethod
    def compute_priority_score(alert: Dict[str, Any]) -> float:
        """
        Computes deterministic ranking score based on:
        Severity (40%) + Composite Risk (25%) + Signal Count (20%) + Recency (15%)
        """
        sev = alert.get("severity", "INFO").upper()
        sev_weight = SEVERITY_WEIGHTS.get(sev, 1.0) / 8.0 * 100.0  # Normalized to 100

        risk_score = float(alert.get("composite_risk_score", 10.0))
        sig_count_score = min(100.0, alert.get("signal_count", 1) * 20.0)

        year = alert.get("year", 2017)
        recency_score = max(0.0, min(100.0, (year - 2000) * 5.0))

        priority_score = (
            0.40 * sev_weight +
            0.25 * risk_score +
            0.20 * sig_count_score +
            0.15 * recency_score
        )
        return round(priority_score, 2)

    @classmethod
    def rank_alerts(
        cls,
        alerts: List[Dict[str, Any]],
        state_filter: Optional[str] = None,
        district_filter: Optional[str] = None,
        severity_filter: Optional[str] = None,
        signal_type_filter: Optional[str] = None,
        year_filter: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Filters and sorts alerts in descending order of priority score.
        """
        filtered = []
        for a in alerts:
            if state_filter and state_filter.strip() and state_filter.lower() != "all":
                if a.get("state", "").lower() != state_filter.strip().lower():
                    continue
            if district_filter and district_filter.strip() and district_filter.lower() != "all":
                if a.get("district", "").lower() != district_filter.strip().lower():
                    continue
            if severity_filter and severity_filter.strip() and severity_filter.lower() != "all":
                if a.get("severity", "").upper() != severity_filter.strip().upper():
                    continue
            if signal_type_filter and signal_type_filter.strip() and signal_type_filter.lower() != "all":
                if signal_type_filter.lower() not in str(a.get("dominant_signal", "")).lower() and \
                   not any(signal_type_filter.lower() in str(s).lower() for s in a.get("supporting_signals", [])):
                    continue
            if year_filter is not None and int(a.get("year", 0)) != int(year_filter):
                continue
            filtered.append(a)

        # Compute priority scores and sort
        scored = []
        for a in filtered:
            item = dict(a)
            item["priority_score"] = cls.compute_priority_score(item)
            scored.append(item)

        scored.sort(key=lambda x: x["priority_score"], reverse=True)

        # Assign ordinal rank
        for idx, item in enumerate(scored, 1):
            item["priority_rank"] = idx
            item["priority_reason"] = (
                f"Rank #{idx} — Severity {item.get('severity')}, {item.get('signal_count', 1)} active signals, "
                f"Composite Score {item.get('composite_risk_score', 0.0):.1f}."
            )

        return scored


alert_prioritization = AlertPrioritization()
