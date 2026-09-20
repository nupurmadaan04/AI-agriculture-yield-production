"""
Sensitivity Analysis Engine.

Quantifies model output elasticity with respect to controlled input feature perturbations
(-20%, -10%, 0%, +10%, +20%) across supported agricultural variables.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from backend.services.ml_service import ml_service
from backend.services.forecast_service import forecast_service
from backend.utils.data_loader import data_loader
from src.scenario_engine import SUPPORTED_SCENARIO_FEATURES

SENSITIVITY_DISCLAIMER = (
    "Sensitivity represents model response to controlled input perturbation and "
    "should not be interpreted as causal elasticity or definitive agronomic response."
)

PERTURBATION_STEPS = [-20.0, -10.0, 0.0, 10.0, 20.0]


class SensitivityAnalysisEngine:
    """
    Computes feature sensitivity curves and elasticity rankings.
    """

    def analyze_sensitivity(
        self,
        state: str,
        district: Optional[str] = None,
        horizon: int = 1,
        features_to_test: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Runs systematic perturbations across supported features.
        """
        state_code, state_name = ml_service.resolve_state(state)
        df = data_loader.dataframe

        # Baseline values
        d_matches = df[df['State Code'] == state_code]
        if district and district.strip() and district.lower() != 'all':
            sub = d_matches[d_matches['Dist Name'].str.lower() == district.strip().lower()]
            if not sub.empty:
                d_matches = sub

        latest_year = int(d_matches['Year'].max()) if not d_matches.empty else 2017
        base_area = float(d_matches['RICE AREA (1000 ha)'].median()) if not d_matches.empty else 100.0
        base_defaults = ml_service.get_district_agronomic_defaults(state_code, district, rice_area=base_area)

        base_feat_dict = {
            'Year': latest_year + horizon,
            'State Code': state_code,
            'RICE AREA (1000 ha)': base_area,
            'TOTAL_CROPPED_AREA': base_defaults['total_cropped_area'],
            'RICE_AREA_SHARE': base_defaults['rice_area_share'],
            'WHEAT AREA (1000 ha)': base_defaults['wheat_area'],
            'COTTON AREA (1000 ha)': base_defaults['cotton_area'],
            'SUGARCANE AREA (1000 ha)': base_defaults['sugarcane_area'],
            'RICE_YIELD_LAG1': base_defaults['rice_yield_lag1'],
            'RICE_YIELD_ROLL3': base_defaults['rice_yield_roll3']
        }

        pipe = forecast_service.get_pipeline()
        scaler = pipe.named_steps['scaler']
        model = pipe.named_steps['model']
        feature_names = [
            'Year', 'State Code', 'RICE AREA (1000 ha)', 'TOTAL_CROPPED_AREA',
            'RICE_AREA_SHARE', 'WHEAT AREA (1000 ha)', 'COTTON AREA (1000 ha)',
            'SUGARCANE AREA (1000 ha)', 'RICE_YIELD_LAG1', 'RICE_YIELD_ROLL3'
        ]

        # Calculate unperturbed baseline prediction
        base_df = pd.DataFrame([base_feat_dict])[feature_names]
        base_scaled = scaler.transform(base_df)
        base_pred = float(model.predict(base_scaled)[0])

        perturbable_cols = {
            'RICE AREA (1000 ha)': 'Rice Area',
            'TOTAL_CROPPED_AREA': 'Total Cropped Area',
            'RICE_AREA_SHARE': 'Rice Area Share',
            'WHEAT AREA (1000 ha)': 'Wheat Area',
            'COTTON AREA (1000 ha)': 'Cotton Area',
            'SUGARCANE AREA (1000 ha)': 'Sugarcane Area',
            'RICE_YIELD_LAG1': 'Prior Year Yield (t-1)',
            'RICE_YIELD_ROLL3': '3-Yr Rolling Mean Yield'
        }

        results = []

        for col, col_label in perturbable_cols.items():
            base_val = float(base_feat_dict[col])
            steps_data = []

            for p_pct in PERTURBATION_STEPS:
                perturbed_dict = base_feat_dict.copy()
                if p_pct == 0.0:
                    perturbed_val = base_val
                else:
                    perturbed_val = max(0.0, base_val * (1.0 + p_pct / 100.0))
                    if col == 'RICE_AREA_SHARE':
                        perturbed_val = min(1.0, perturbed_val)

                perturbed_dict[col] = perturbed_val

                # Recompute share if rice area changed
                if col == 'RICE AREA (1000 ha)':
                    tot = perturbed_dict['TOTAL_CROPPED_AREA']
                    perturbed_dict['RICE_AREA_SHARE'] = min(1.0, perturbed_val / max(tot, 0.1))

                p_df = pd.DataFrame([perturbed_dict])[feature_names]
                p_scaled = scaler.transform(p_df)
                p_yield = float(model.predict(p_scaled)[0])

                delta = round(p_yield - base_pred, 1)
                pct_change = round((delta / base_pred * 100.0) if base_pred > 0 else 0.0, 2)

                steps_data.append({
                    'perturbation_pct': p_pct,
                    'perturbed_input_value': round(perturbed_val, 2),
                    'predicted_yield': round(p_yield, 1),
                    'yield_delta': delta,
                    'yield_percent_change': pct_change
                })

            # Calculate elasticity: (|ΔYield(+20%)| + |ΔYield(-20%)|) / 2
            y_plus20 = next(s['predicted_yield'] for s in steps_data if s['perturbation_pct'] == 20.0)
            y_minus20 = next(s['predicted_yield'] for s in steps_data if s['perturbation_pct'] == -20.0)
            swing = abs(y_plus20 - y_minus20)
            elasticity = round(swing / max(base_pred, 1.0) * 100.0, 2)

            results.append({
                'feature_key': col,
                'feature_name': col_label,
                'baseline_value': round(base_val, 2),
                'elasticity_index': elasticity,
                'perturbation_responses': steps_data
            })

        # Rank features by elasticity index
        results.sort(key=lambda x: x['elasticity_index'], reverse=True)
        for rank, item in enumerate(results, 1):
            item['sensitivity_rank'] = rank

        return {
            'location': f"{state_name}" + (f" - {district}" if district else ""),
            'horizon': horizon,
            'baseline_prediction': round(base_pred, 1),
            'perturbation_steps': PERTURBATION_STEPS,
            'features_analyzed': len(results),
            'most_sensitive_feature': results[0]['feature_name'] if results else 'N/A',
            'sensitivity_matrix': results,
            'scientific_disclaimer': SENSITIVITY_DISCLAIMER
        }


sensitivity_analysis_engine = SensitivityAnalysisEngine()
