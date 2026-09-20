"""
Agricultural Early Warning Engine.

Computes deterministic 0–100 agricultural early-warning scores and generates
evidence-grounded early warning signals across yield decline, persistent contraction,
high volatility, forecast deviation, spatial outliers, model error escalation, and scenario stress.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from src.alert_severity import alert_severity_classifier, SeverityTier


class EarlyWarningEngine:
    """
    Evaluates multi-variable warning signals and deterministic early warning metrics.
    """

    @staticmethod
    def calculate_score(
        trend_direction: str,
        trend_slope: float,
        forecast_change_pct: float,
        historical_z_score: float,
        is_anomaly: bool,
        anomaly_score: float,
        prediction_spread_pct: float
    ) -> Tuple[float, str, List[str], Dict[str, float]]:
        """
        Computes composite 0-100 Early Warning Score (Day 7 backward-compatible).
        Returns: (score, severity, trigger_signals, component_scores)
        """
        triggers = []

        # 1. Trend Component (30% weight)
        if trend_direction == "STRONG DECREASING":
            s_trend = 100.0
            triggers.append(f"Statistically significant negative historical yield trend ({trend_slope:.1f} kg/ha/yr).")
        elif trend_direction == "DECREASING":
            s_trend = 75.0
            triggers.append(f"Negative multi-year yield trajectory ({trend_slope:.1f} kg/ha/yr).")
        elif trend_direction == "STABLE":
            s_trend = 25.0
        elif trend_direction == "INCREASING":
            s_trend = 10.0
        else:  # STRONG INCREASING
            s_trend = 0.0

        # 2. Forecast Change Component (25% weight)
        if forecast_change_pct <= -15.0:
            s_forecast = 100.0
            triggers.append(f"Severe projected forward yield contraction ({forecast_change_pct:.1f}%).")
        elif forecast_change_pct <= -5.0:
            s_forecast = 65.0
            triggers.append(f"Projected forward yield decline ({forecast_change_pct:.1f}% vs latest observed).")
        elif forecast_change_pct < 0.0:
            s_forecast = 35.0
        else:
            s_forecast = 10.0

        # 3. Historical Deviation Component (20% weight)
        z_abs = abs(historical_z_score)
        if z_abs >= 3.0:
            s_dev = 100.0
            triggers.append(f"Extreme historical deviation ({historical_z_score:+.2f} std dev).")
        elif z_abs >= 2.0:
            s_dev = 65.0
            triggers.append(f"Notable historical departure ({historical_z_score:+.2f} std dev).")
        elif z_abs >= 1.0:
            s_dev = 30.0
        else:
            s_dev = 10.0

        # 4. Anomaly Component (15% weight)
        if is_anomaly or anomaly_score >= 70.0:
            s_anom = min(100.0, anomaly_score * 1.2)
            triggers.append(f"Isolation Forest multi-variable outlier signal (score: {anomaly_score:.1f}/100).")
        else:
            s_anom = max(0.0, anomaly_score * 0.5)

        # 5. Prediction Spread Component (10% weight)
        if prediction_spread_pct >= 30.0:
            s_spread = 100.0
            triggers.append(f"High model prediction spread (±{prediction_spread_pct/2:.1f}%).")
        elif prediction_spread_pct >= 20.0:
            s_spread = 60.0
        else:
            s_spread = 20.0

        # Composite Formula: 0.30*Trend + 0.25*Forecast + 0.20*Dev + 0.15*Anom + 0.10*Spread
        composite_score = (
            0.30 * s_trend +
            0.25 * s_forecast +
            0.20 * s_dev +
            0.15 * s_anom +
            0.10 * s_spread
        )
        composite_score = float(round(np.clip(composite_score, 0.0, 100.0), 1))

        # Severity Mapping (Day 7 legacy tiering)
        if composite_score >= 75.0:
            severity = "CRITICAL"
        elif composite_score >= 50.0:
            severity = "HIGH"
        elif composite_score >= 25.0:
            severity = "MODERATE"
        else:
            severity = "LOW"

        if not triggers:
            triggers.append("No statistically significant deterioration or distress signals detected.")

        components = {
            'trend_signal_score': round(s_trend, 1),
            'forecast_signal_score': round(s_forecast, 1),
            'historical_deviation_score': round(s_dev, 1),
            'anomaly_signal_score': round(s_anom, 1),
            'prediction_spread_score': round(s_spread, 1)
        }

        return composite_score, severity, triggers, components

    @staticmethod
    def generate_signal(
        signal_id: str,
        signal_type: str,
        state: str,
        district: str,
        year: int,
        trigger_value: float,
        threshold: float,
        unit: str,
        severity: str,
        evidence: List[str],
        recommended_action: str
    ) -> Dict[str, Any]:
        """Creates a standardized early warning signal item."""
        return {
            "signal_id": signal_id,
            "signal_type": signal_type,
            "state": state,
            "district": district,
            "year": year,
            "severity": severity,
            "trigger_value": round(trigger_value, 2),
            "threshold": round(threshold, 2),
            "unit": unit,
            "evidence": evidence,
            "recommended_action": recommended_action,
            "is_active": severity in [SeverityTier.CRITICAL.value, SeverityTier.HIGH.value, SeverityTier.ELEVATED.value, SeverityTier.WATCH.value]
        }


early_warning_engine = EarlyWarningEngine()
