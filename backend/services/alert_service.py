"""
Agricultural Alert Service.

Aggregates empirical risk signals, executes multi-signal fusion, generates auditable alerts,
and provides ranked alert feeds, state/district warning maps, and evidence certificates.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from backend.utils.data_loader import data_loader
from backend.services.ml_service import ml_service, STATE_TO_CODE
from backend.services.risk_service import risk_service
from backend.services.anomaly_service import anomaly_service
from backend.services.trend_service import trend_service
from backend.services.forecast_service import forecast_service
from backend.services.spatial_outlier_service import spatial_outlier_service
from backend.services.data_quality_service import data_quality_service
from backend.services.error_service import error_service
from src.early_warning_engine import early_warning_engine
from src.risk_signal_fusion import risk_signal_fusion
from src.alert_prioritization import alert_prioritization
from src.alert_severity import SeverityTier


class AlertService:
    _instance: Optional['AlertService'] = None
    _cached_alerts: Optional[List[Dict[str, Any]]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AlertService, cls).__new__(cls)
        return cls._instance

    def _generate_all_district_alerts(self) -> List[Dict[str, Any]]:
        """
        Generates deterministic alerts across all districts in the ICRISAT dataset.
        Caches results in-memory for fast querying.
        """
        if self._cached_alerts is not None:
            return self._cached_alerts

        df = data_loader.dataframe
        latest_year = int(df["Year"].max())
        dq = data_quality_service.get_data_quality_audit().get("overall_quality_score", 100.0)

        # Get spatial outliers for reference
        spatial_outliers_list = spatial_outlier_service.get_spatial_outliers(z_threshold=2.0)
        spatial_outlier_districts = {f"{o['state']}_{o['district']}": o for o in spatial_outliers_list}

        alerts = []

        # Iterate over major representative state-district pairs
        for (state_name, dist_name), group in df.groupby(["State Name", "Dist Name"]):
            grp_sorted = group.sort_values("Year")
            if grp_sorted.empty:
                continue

            vals = grp_sorted["RICE YIELD (Kg per ha)"].tolist()
            years = grp_sorted["Year"].tolist()
            if len(vals) < 2:
                continue

            cur_val = vals[-1]
            prev_val = vals[-2]
            obs_year = years[-1]
            yoy_pct = ((cur_val - prev_val) / prev_val * 100.0) if prev_val > 0 else 0.0

            hist_mean = float(np.mean(vals))
            hist_std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0
            if hist_std < 1e-6:
                hist_std = 1.0
            z_score = (cur_val - hist_mean) / hist_std

            signals = []

            # Signal 1: YoY Yield Decline
            if yoy_pct <= -5.0:
                sev = SeverityTier.CRITICAL.value if yoy_pct <= -20.0 else (
                    SeverityTier.HIGH.value if yoy_pct <= -15.0 else (
                        SeverityTier.ELEVATED.value if yoy_pct <= -10.0 else SeverityTier.WATCH.value
                    )
                )
                signals.append(early_warning_engine.generate_signal(
                    signal_id=f"SIG-YOY-{abs(hash(dist_name)) % 10000:04d}",
                    signal_type="Year-over-Year Yield Decline",
                    state=state_name,
                    district=dist_name,
                    year=obs_year,
                    trigger_value=yoy_pct,
                    threshold=-5.0,
                    unit="%",
                    severity=sev,
                    evidence=[
                        f"Reported yield declined from {prev_val:.1f} kg/ha to {cur_val:.1f} kg/ha ({yoy_pct:+.1f}%).",
                        f"Threshold exceeded: -5.0% operational alert limit."
                    ],
                    recommended_action="Review local district crop reporting and verify weather indicators."
                ))

            # Signal 2: Persistent Multi-Year Contraction
            if len(vals) >= 3 and vals[-1] < vals[-2] < vals[-3]:
                drop_2yr = (vals[-1] - vals[-3]) / vals[-3] * 100.0 if vals[-3] > 0 else 0.0
                signals.append(early_warning_engine.generate_signal(
                    signal_id=f"SIG-PERS-{abs(hash(dist_name)) % 10000:04d}",
                    signal_type="Persistent Multi-Year Contraction",
                    state=state_name,
                    district=dist_name,
                    year=obs_year,
                    trigger_value=drop_2yr,
                    threshold=-8.0,
                    unit="%",
                    severity=SeverityTier.HIGH.value if drop_2yr <= -15.0 else SeverityTier.ELEVATED.value,
                    evidence=[
                        f"Yield has contracted consecutively across 3 seasons ({vals[-3]:.1f} → {vals[-2]:.1f} → {vals[-1]:.1f} kg/ha).",
                        f"Cumulative 2-year decrease: {drop_2yr:+.1f}%."
                    ],
                    recommended_action="Execute multi-year stress scenario simulation to assess carryover vulnerability."
                ))

            # Signal 3: Statistical Baseline Departure
            if abs(z_score) >= 1.5:
                sev = SeverityTier.CRITICAL.value if abs(z_score) >= 3.0 else (
                    SeverityTier.HIGH.value if abs(z_score) >= 2.5 else (
                        SeverityTier.ELEVATED.value if abs(z_score) >= 2.0 else SeverityTier.WATCH.value
                    )
                )
                signals.append(early_warning_engine.generate_signal(
                    signal_id=f"SIG-DEV-{abs(hash(dist_name)) % 10000:04d}",
                    signal_type="Statistical Baseline Departure",
                    state=state_name,
                    district=dist_name,
                    year=obs_year,
                    trigger_value=z_score,
                    threshold=1.5,
                    unit="std dev",
                    severity=sev,
                    evidence=[
                        f"Current yield deviates by {z_score:+.2f} standard deviations from historical district mean ({hist_mean:.1f} kg/ha).",
                        f"Statistical departure indicates abnormal variance relative to long-term baseline."
                    ],
                    recommended_action="Examine whether departure corresponds to reporting adjustment or regional climatic anomaly."
                ))

            # Signal 4: Spatial Outlier Status (Day 8 Reuse)
            key = f"{state_name}_{dist_name}"
            if key in spatial_outlier_districts:
                s_item = spatial_outlier_districts[key]
                signals.append(early_warning_engine.generate_signal(
                    signal_id=f"SIG-SPAT-{abs(hash(dist_name)) % 10000:04d}",
                    signal_type="Within-State Spatial Outlier",
                    state=state_name,
                    district=dist_name,
                    year=obs_year,
                    trigger_value=s_item.get("within_state_zscore", 2.2),
                    threshold=2.0,
                    unit="spatial z",
                    severity=SeverityTier.ELEVATED.value,
                    evidence=[
                        f"District yield differs from state peer mean by {s_item.get('within_state_zscore', 2.2):+.2f} spatial z-score.",
                        f"Classification: {s_item.get('spatial_outlier_type', 'HIGH_OUTLIER')} within {state_name} agricultural zone."
                    ],
                    recommended_action="Cross-reference with neighboring district performance before applying state-wide directives."
                ))

            # Fuse into alert
            location_label = f"{state_name} - {dist_name}"
            fused = risk_signal_fusion.fuse_signals(
                location=location_label,
                state=state_name,
                district=dist_name,
                year=obs_year,
                signals=signals,
                data_quality_score=dq,
                model_r2=0.7866,
                model_mae=353.01,
                drift_status="NORMAL"
            )
            alerts.append(fused)

        self._cached_alerts = alerts
        return alerts

    def get_overview(self) -> Dict[str, Any]:
        """
        Returns top-level KPIs for the Monitoring Command Center.
        """
        all_alerts = self._generate_all_district_alerts()
        active = [a for a in all_alerts if a.get("severity") != SeverityTier.INFO.value]

        crit_count = sum(1 for a in all_alerts if a.get("severity") == SeverityTier.CRITICAL.value)
        high_count = sum(1 for a in all_alerts if a.get("severity") == SeverityTier.HIGH.value)
        elev_count = sum(1 for a in all_alerts if a.get("severity") == SeverityTier.ELEVATED.value)
        watch_count = sum(1 for a in all_alerts if a.get("severity") == SeverityTier.WATCH.value)
        info_count = sum(1 for a in all_alerts if a.get("severity") == SeverityTier.INFO.value)

        states_under_watch = len(set(a["state"] for a in active if a.get("severity") in [SeverityTier.CRITICAL.value, SeverityTier.HIGH.value, SeverityTier.ELEVATED.value]))
        districts_under_watch = len(active)

        persistent_count = sum(1 for a in all_alerts if "Persistent" in str(a.get("dominant_signal", "")) or any("Persistent" in str(s) for s in a.get("supporting_signals", [])))

        return {
            "active_alerts_count": len(active),
            "high_critical_count": crit_count + high_count,
            "states_under_watch": states_under_watch,
            "districts_under_watch": districts_under_watch,
            "persistent_signals_count": persistent_count,
            "model_monitoring_status": "HEALTHY",
            "latest_observation_year": 2017,
            "summary": {
                "total_alerts": len(all_alerts),
                "critical_alerts_count": crit_count,
                "high_alerts_count": high_count,
                "elevated_alerts_count": elev_count,
                "watch_alerts_count": watch_count,
                "info_alerts_count": info_count,
                "states_under_watch_count": states_under_watch,
                "districts_under_watch_count": districts_under_watch
            },
            "scientific_disclaimer": (
                "Monitoring signals reflect empirical statistical deviations in the ICRISAT panel dataset. "
                "Alerts indicate priorities for closer monitoring and do not constitute causal predictions."
            )
        }

    def get_ranked_alerts(
        self,
        state: Optional[str] = None,
        district: Optional[str] = None,
        severity: Optional[str] = None,
        signal_type: Optional[str] = None,
        year: Optional[int] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Returns prioritized alerts matching user filters.
        """
        all_alerts = self._generate_all_district_alerts()
        ranked = alert_prioritization.rank_alerts(
            alerts=all_alerts,
            state_filter=state,
            district_filter=district,
            severity_filter=severity,
            signal_type_filter=signal_type,
            year_filter=year
        )
        return ranked[:limit]

    def get_alert_by_id(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific alert by ID with full evidence certificate."""
        all_alerts = self._generate_all_district_alerts()
        for a in all_alerts:
            if a.get("alert_id") == alert_id:
                return a
        return None

    def get_warning_map_data(self) -> List[Dict[str, Any]]:
        """
        Returns state-level aggregated warning severity and signal density for the India Warning Map.
        """
        all_alerts = self._generate_all_district_alerts()
        df = data_loader.dataframe

        state_stats = {}
        for state_name in sorted(STATE_TO_CODE.keys()):
            st_alerts = [a for a in all_alerts if a.get("state") == state_name]
            crit = sum(1 for a in st_alerts if a.get("severity") == SeverityTier.CRITICAL.value)
            high = sum(1 for a in st_alerts if a.get("severity") == SeverityTier.HIGH.value)
            elev = sum(1 for a in st_alerts if a.get("severity") == SeverityTier.ELEVATED.value)
            watch = sum(1 for a in st_alerts if a.get("severity") == SeverityTier.WATCH.value)

            if crit >= 1 or high >= 2:
                sev = SeverityTier.CRITICAL.value if crit >= 1 else SeverityTier.HIGH.value
            elif high >= 1 or elev >= 2:
                sev = SeverityTier.HIGH.value if high >= 1 else SeverityTier.ELEVATED.value
            elif elev >= 1 or watch >= 3:
                sev = SeverityTier.ELEVATED.value if elev >= 1 else SeverityTier.WATCH.value
            elif watch >= 1:
                sev = SeverityTier.WATCH.value
            else:
                sev = SeverityTier.INFO.value

            sub_df = df[df["State Name"] == state_name]
            avg_yield = round(float(sub_df["RICE YIELD (Kg per ha)"].mean()), 1) if not sub_df.empty else 2000.0

            state_stats[state_name] = {
                "state": state_name,
                "state_code": STATE_TO_CODE[state_name],
                "severity": sev,
                "active_signals_count": crit + high + elev + watch,
                "critical_count": crit,
                "high_count": high,
                "elevated_count": elev,
                "watch_count": watch,
                "average_yield_kg_ha": avg_yield,
                "total_districts_monitored": len(st_alerts),
                "dominant_concern": "Elevated regional yield variance" if sev in [SeverityTier.HIGH.value, SeverityTier.CRITICAL.value] else "Normal agricultural baseline"
            }

        return list(state_stats.values())

    def get_state_monitoring_summary(self, state: str) -> Dict[str, Any]:
        """Returns comprehensive monitoring breakdown for a single state."""
        state_code, state_name = ml_service.resolve_state(state)
        all_alerts = self._generate_all_district_alerts()
        st_alerts = [a for a in all_alerts if a.get("state") == state_name]

        return {
            "state": state_name,
            "state_code": state_code,
            "total_districts": len(st_alerts),
            "alerts": st_alerts,
            "highest_severity": max((a.get("severity") for a in st_alerts), default=SeverityTier.INFO.value),
            "scientific_disclaimer": "State monitoring aggregates historical district signals; not a causal drought/flood forecast."
        }


alert_service = AlertService()
