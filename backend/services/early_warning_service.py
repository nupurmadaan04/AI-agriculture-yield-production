"""
Agricultural Early Warning Service.

Assesses state and district early-warning indicators combining historical trend slope,
multi-horizon forward forecasts, statistical deviations, and anomaly detections.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from backend.utils.data_loader import data_loader
from backend.services.ml_service import ml_service, STATE_TO_CODE
from backend.services.trend_service import trend_service
from backend.services.forecast_service import forecast_service
from backend.services.anomaly_service import anomaly_service
from src.early_warning_engine import early_warning_engine

class EarlyWarningService:
    _instance: Optional['EarlyWarningService'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EarlyWarningService, cls).__new__(cls)
        return cls._instance

    def assess_region(
        self,
        state: Optional[str] = "Punjab",
        district: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assesses early warning status and risk deterioration signals for a given state or district.
        """
        target_state = state or "Punjab"
        state_code, state_name = ml_service.resolve_state(target_state)

        # 1. Historical Trend Analysis
        trend_res = trend_service.analyze_region_trend(state=state_name, district=district)
        trend_dir = trend_res['direction']
        trend_slope = trend_res['theil_sen_slope']

        # 2. Forward Forecast
        fc_res = forecast_service.forecast_region(state_val=state_name, district=district, horizons=[1, 2, 3])
        latest_y = fc_res['latest_observed_yield']
        next_y = fc_res['forecasts'][0]['predicted_yield']
        fc_change_pct = ((next_y - latest_y) / latest_y * 100.0) if latest_y > 0 else 0.0
        pred_spread_pct = fc_res['forecasts'][0]['uncertainty_pct']

        # 3. Anomaly Evaluation
        anom_res = anomaly_service.detect_anomaly(
            year=fc_res['latest_observed_year'],
            state_val=state_name,
            area=100.0,
            yield_val=next_y,
            district=district
        )
        z_score = anom_res.get('yield_z_score') or 0.0
        is_anom = anom_res['is_anomaly']
        anom_score = anom_res['anomaly_score']

        # 4. Early Warning Calculation
        warning_score, severity, triggers, components = early_warning_engine.calculate_score(
            trend_direction=trend_dir,
            trend_slope=trend_slope,
            forecast_change_pct=fc_change_pct,
            historical_z_score=z_score,
            is_anomaly=is_anom,
            anomaly_score=anom_score,
            prediction_spread_pct=pred_spread_pct
        )

        return {
            'state': state_name,
            'district': district or "Regional Summary",
            'warning_score': warning_score,
            'severity': severity,
            'trend_direction': trend_dir,
            'trend_slope_kg_ha_yr': trend_slope,
            'trend_significance': trend_res['significance'],
            'forecast_1yr_kg_ha': next_y,
            'forecast_change_pct': round(fc_change_pct, 2),
            'latest_observed_yield': latest_y,
            'prediction_spread_pct': pred_spread_pct,
            'is_anomaly': is_anom,
            'anomaly_score': anom_score,
            'trigger_signals': triggers,
            'components': components,
            'disclaimer': 'Early warning scores reflect statistical deterioration signals and do not represent biological probabilities of crop failure.'
        }

    def get_all_states_early_warning(self) -> List[Dict[str, Any]]:
        """
        Computes early warning matrix across all 20 states.
        """
        results = []
        for state_name in sorted(STATE_TO_CODE.keys()):
            res = self.assess_region(state=state_name)
            results.append(res)

        # Sort descending by early warning score
        results.sort(key=lambda x: x['warning_score'], reverse=True)
        return results

    def get_early_warning_dashboard(self) -> Dict[str, Any]:
        """
        Aggregates system-wide early warning KPIs and priority regions.
        """
        all_states = self.get_all_states_early_warning()
        critical_cnt = sum(1 for s in all_states if s['severity'] == 'CRITICAL')
        high_cnt = sum(1 for s in all_states if s['severity'] == 'HIGH')
        mod_cnt = sum(1 for s in all_states if s['severity'] == 'MODERATE')
        low_cnt = sum(1 for s in all_states if s['severity'] == 'LOW')

        declining_states = [s['state'] for s in all_states if s['trend_direction'] in ['DECREASING', 'STRONG DECREASING']]
        avg_spread = float(round(np.mean([s['prediction_spread_pct'] for s in all_states]), 1)) if all_states else 20.0

        return {
            'total_states_monitored': len(all_states),
            'critical_states_count': critical_cnt,
            'high_states_count': high_cnt,
            'moderate_states_count': mod_cnt,
            'low_states_count': low_cnt,
            'declining_states_count': len(declining_states),
            'declining_states': declining_states,
            'average_forecast_spread_pct': avg_spread,
            'top_priority_warnings': all_states[:5],
            'state_matrix': all_states
        }

early_warning_service = EarlyWarningService()
