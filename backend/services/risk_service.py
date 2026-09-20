"""
Agricultural Risk Intelligence Service.

Computes a deterministic, transparent composite risk score (0-100) combining:
1. Prediction Uncertainty (10th-90th percentile tree ensemble spread)
2. Historical Deviation (deviation from historical district/state mean)
3. Model Historical Error (residual error from similar agro-climatic conditions)
4. Agricultural Anomaly Score (Isolation Forest multi-dimensional outlier score)

Documented as a decision-support heuristic rather than a physical causal model.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from backend.utils.data_loader import data_loader
from backend.services.ml_service import ml_service, STATE_TO_CODE, CODE_TO_STATE
from backend.services.anomaly_service import anomaly_service

# Configurable Weights for Deterministic Composite Risk Scoring
WEIGHT_UNCERTAINTY = 0.35
WEIGHT_HIST_DEVIATION = 0.30
WEIGHT_MODEL_ERROR = 0.20
WEIGHT_ANOMALY = 0.15

# Risk Threshold Constants
RISK_THRESHOLD_MODERATE = 25.0
RISK_THRESHOLD_HIGH = 50.0
RISK_THRESHOLD_CRITICAL = 75.0

class RiskService:
    _instance: Optional['RiskService'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RiskService, cls).__new__(cls)
        return cls._instance

    def calculate_risk(
        self,
        predicted_yield: float,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        year: Optional[int] = None,
        state_val: Optional[Any] = None,
        district: Optional[str] = None,
        area: Optional[float] = None,
        historical_yield_mean: Optional[float] = None,
        historical_yield_std: Optional[float] = None,
        anomaly_score: Optional[float] = None
    ) -> Dict[str, Any]:
        return self.assess_risk(
            predicted_yield=predicted_yield,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            year=year,
            state_val=state_val,
            district=district,
            area=area,
            historical_yield_mean=historical_yield_mean,
            historical_yield_std=historical_yield_std,
            anomaly_score=anomaly_score
        )

    def assess_risk(
        self,
        predicted_yield: float,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        year: Optional[int] = None,
        state_val: Optional[Any] = None,
        district: Optional[str] = None,
        area: Optional[float] = None,
        historical_yield_mean: Optional[float] = None,
        historical_yield_std: Optional[float] = None,
        anomaly_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculates deterministic composite risk score and risk factors.
        """
        # Handle zero or negative predicted yield safely
        safe_predicted_yield = max(1.0, float(predicted_yield))

        # 1. Uncertainty Component
        if lower_bound is not None and upper_bound is not None and upper_bound >= lower_bound:
            spread = upper_bound - lower_bound
            uncertainty_pct = float(round((spread / safe_predicted_yield) * 100.0, 2))
        else:
            spread = safe_predicted_yield * 0.25
            uncertainty_pct = 25.0

        # Uncertainty risk normalized to 0-100 (where 50% relative spread = 100 risk)
        uncertainty_risk = float(np.clip(uncertainty_pct * 2.0, 0.0, 100.0))

        if uncertainty_pct < 15.0:
            confidence_label = "Low prediction spread"
        elif uncertainty_pct <= 30.0:
            confidence_label = "Moderate prediction spread"
        else:
            confidence_label = "High prediction spread"

        # 2. Historical Deviation Component
        df = data_loader.dataframe
        hist_mean = historical_yield_mean
        hist_std = historical_yield_std

        if (hist_mean is None or hist_std is None) and state_val is not None:
            try:
                state_code, state_name = ml_service.resolve_state(state_val)
                matches = df[df['State Code'] == state_code]
                if district and district.strip() and district.lower() != 'unknown':
                    d_matches = matches[matches['Dist Name'].str.lower() == district.strip().lower()]
                    if not d_matches.empty:
                        matches = d_matches
                if not matches.empty:
                    hist_mean = float(matches['RICE YIELD (Kg per ha)'].mean())
                    hist_std = float(matches['RICE YIELD (Kg per ha)'].std()) if len(matches) > 1 else 300.0
            except Exception:
                pass

        if hist_mean is None:
            hist_mean = 2062.8
        if hist_std is None or hist_std == 0:
            hist_std = 400.0

        z_dev = abs(safe_predicted_yield - hist_mean) / hist_std
        # z=3 maps to 100 risk
        hist_dev_risk = float(np.clip((z_dev / 3.0) * 100.0, 0.0, 100.0))

        # 3. Model Residual Error Component
        # Average baseline RMSE in pre-season is ~357 kg/ha (approx 15-20% relative error)
        expected_rel_error_pct = (357.0 / hist_mean) * 100.0 if hist_mean > 0 else 18.0
        model_error_risk = float(np.clip(expected_rel_error_pct * 3.0, 10.0, 85.0))

        # 4. Anomaly Component
        if anomaly_score is None and state_val is not None and area is not None:
            anom_res = anomaly_service.detect_anomaly(
                year=year or 2017,
                state_val=state_val,
                area=area,
                yield_val=safe_predicted_yield,
                district=district
            )
            anom_score_val = float(anom_res['anomaly_score'])
        else:
            anom_score_val = float(anomaly_score) if anomaly_score is not None else 20.0

        anomaly_risk = float(np.clip(anom_score_val, 0.0, 100.0))

        # Composite Deterministic Risk Formula
        composite_risk = (
            WEIGHT_UNCERTAINTY * uncertainty_risk +
            WEIGHT_HIST_DEVIATION * hist_dev_risk +
            WEIGHT_MODEL_ERROR * model_error_risk +
            WEIGHT_ANOMALY * anomaly_risk
        )
        composite_risk = float(round(np.clip(composite_risk, 0.0, 100.0), 1))

        # Categorize Risk Level
        if composite_risk < RISK_THRESHOLD_MODERATE:
            risk_level = "LOW"
        elif composite_risk < RISK_THRESHOLD_HIGH:
            risk_level = "MODERATE"
        elif composite_risk < RISK_THRESHOLD_CRITICAL:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        # Risk Factors Identification
        risk_factors: List[str] = []
        if uncertainty_pct > 30.0:
            risk_factors.append(f"High tree-ensemble prediction spread (±{spread/2:.1f} kg/ha, {uncertainty_pct:.1f}% relative spread).")
        elif uncertainty_pct > 20.0:
            risk_factors.append(f"Moderate tree-ensemble spread across decision paths ({uncertainty_pct:.1f}%).")

        if z_dev >= 2.0:
            risk_factors.append(f"Forecasted yield departs substantially from district historical baseline ({z_dev:.2f} standard deviations).")

        if anom_score_val >= 50.0:
            risk_factors.append(f"Unsupervised anomaly detector flags unusual agricultural feature profile (score: {anom_score_val:.1f}/100).")

        if area is not None and area < 2.0:
            risk_factors.append("Low cultivated acreage (<2,000 ha): higher historical volatility in district reporting.")

        if not risk_factors:
            risk_factors.append("Forecast aligns closely with regional historical stability and narrow ensemble prediction interval.")

        # Human-Readable Explanation without Fake Causal Claims
        explanation = (
            f"Prediction risk is evaluated as {risk_level} ({composite_risk:.1f}/100). "
            f"Model ensemble uncertainty is {uncertainty_pct:.1f}% with an expected {confidence_label.lower()}. "
            f"Historical departure from regional baseline is {z_dev:.2f} standard deviations."
        )

        return {
            'risk_level': risk_level,
            'risk_score': composite_risk,
            'confidence_label': confidence_label,
            'uncertainty_percent': uncertainty_pct,
            'spread': round(spread, 1),
            'risk_factors': risk_factors,
            'explanation': explanation,
            'components': {
                'uncertainty_risk': round(uncertainty_risk, 1),
                'historical_deviation_risk': round(hist_dev_risk, 1),
                'model_error_risk': round(model_error_risk, 1),
                'anomaly_risk': round(anomaly_risk, 1)
            }
        }

    def get_state_risk_analytics(self) -> List[Dict[str, Any]]:
        """
        Computes state-level risk profiles across all 20 Indian states in the ICRISAT panel.
        """
        df = data_loader.dataframe
        if df.empty:
            return []

        # Error analysis baseline
        state_err = ml_service.get_error_analysis_data()['worst_performing_states']
        err_map = {item['state']: item['mae'] for item in state_err}

        # Anomaly rates by state
        anomalies = anomaly_service.get_dataset_anomalies(limit=2500)
        anom_state_counts: Dict[str, int] = {}
        for a in anomalies:
            st = a['state']
            anom_state_counts[st] = anom_state_counts.get(st, 0) + 1

        results = []
        for state_name, state_code in sorted(STATE_TO_CODE.items()):
            s_data = df[df['State Code'] == state_code]
            if s_data.empty:
                continue

            count = len(s_data)
            avg_yield = float(round(s_data['RICE YIELD (Kg per ha)'].mean(), 1))
            std_yield = float(round(s_data['RICE YIELD (Kg per ha)'].std(), 1)) if count > 1 else 0.0
            volatility = float(round((std_yield / avg_yield * 100.0) if avg_yield > 0 else 0.0, 1))

            mae = float(round(err_map.get(state_name, 250.0), 1))
            anom_cnt = anom_state_counts.get(state_name, 0)
            anom_rate = float(round((anom_cnt / count * 100.0) if count > 0 else 0.0, 1))

            # Average uncertainty for state
            avg_unc = float(round(np.clip(volatility * 0.8 + (mae / avg_yield * 100.0) * 0.5, 10.0, 45.0), 1))

            # Composite State Risk Score
            st_risk = float(round(np.clip(
                (volatility * 0.35) + (anom_rate * 2.5) + (mae / 15.0) + (avg_unc * 0.5),
                10.0,
                95.0
            ), 1))

            if st_risk < 30.0:
                risk_level = "LOW"
            elif st_risk < 50.0:
                risk_level = "MODERATE"
            elif st_risk < 70.0:
                risk_level = "HIGH"
            else:
                risk_level = "CRITICAL"

            results.append({
                'state': state_name,
                'state_code': state_code,
                'record_count': count,
                'average_yield': avg_yield,
                'yield_volatility': volatility,
                'average_prediction_error_mae': mae,
                'anomaly_rate_pct': anom_rate,
                'average_uncertainty_pct': avg_unc,
                'risk_score': st_risk,
                'risk_level': risk_level
            })

        # Sort descending by risk score
        results.sort(key=lambda x: x['risk_score'], reverse=True)
        return results

    def get_state_risk_profile(self, state_name: str) -> Dict[str, Any]:
        """Returns risk profile for a single state."""
        all_states = self.get_state_risk_analytics()
        for s in all_states:
            if s['state'].lower() == state_name.lower():
                return s
        return {
            'state': state_name,
            'risk_score': 38.5,
            'risk_level': 'MODERATE',
            'yield_volatility': 15.0,
            'anomaly_rate_pct': 5.0,
            'average_uncertainty_pct': 21.0
        }

    def get_intelligence_dashboard_summary(self) -> Dict[str, Any]:
        """
        Aggregates system-wide intelligence metrics for the dashboard.
        """
        df = data_loader.dataframe
        state_risks = self.get_state_risk_analytics()
        anomalies = anomaly_service.get_dataset_anomalies(limit=20)

        # Risk distribution counts
        low_count = sum(1 for s in state_risks if s['risk_level'] == 'LOW')
        mod_count = sum(1 for s in state_risks if s['risk_level'] == 'MODERATE')
        high_count = sum(1 for s in state_risks if s['risk_level'] in ['HIGH', 'CRITICAL'])

        avg_uncertainty = float(round(np.mean([s['average_uncertainty_pct'] for s in state_risks]), 1)) if state_risks else 21.4

        return {
            'total_records': len(df),
            'anomalies_detected': len(anomaly_service.get_dataset_anomalies(limit=2500)),
            'high_risk_states_count': high_count,
            'moderate_risk_states_count': mod_count,
            'low_risk_states_count': low_count,
            'average_uncertainty_pct': avg_uncertainty,
            'highest_risk_states': state_risks[:5],
            'recent_anomalies': anomalies[:8]
        }

    def get_state_risk_summary(self) -> List[Dict[str, Any]]:
        """Alias for get_state_risk_analytics."""
        return self.get_state_risk_analytics()

risk_service = RiskService()
