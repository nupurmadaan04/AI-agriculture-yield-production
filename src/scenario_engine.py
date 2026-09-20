"""
Scenario Simulation & What-If Analysis Engine.

Enables comparative modeling of agricultural input modifications against baseline
empirical district profiles using the validated exogenous pre-season model pipeline
and multi-horizon forecaster.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from backend.utils.data_loader import data_loader
from backend.services.ml_service import ml_service
from backend.services.forecast_service import forecast_service

SCIENTIFIC_SCENARIO_DISCLAIMER = (
    "Scenario outputs represent hypothetical model simulations under modified input assumptions. "
    "They must never be interpreted as guaranteed future outcomes or causal conclusions."
)

SUPPORTED_SCENARIO_FEATURES = {
    'rice_area': 'Cultivated Rice Area (1000 ha)',
    'total_cropped_area': 'Total Cropped Area (1000 ha)',
    'rice_area_share': 'Rice Cropland Allocation Share',
    'wheat_area': 'Wheat Area (1000 ha)',
    'cotton_area': 'Cotton Area (1000 ha)',
    'sugarcane_area': 'Sugarcane Area (1000 ha)',
    'historical_yield_lag': 'Prior Year Yield Lag (Kg per ha)',
    'rolling_yield': '3-Year Rolling Mean Yield (Kg per ha)'
}

# Bounds for input modifications (percentages or bounds)
SCENARIO_BOUNDS = {
    'rice_area_pct': (-30.0, 30.0),
    'total_cropped_area_pct': (-30.0, 30.0),
    'wheat_area_pct': (-50.0, 50.0),
    'cotton_area_pct': (-50.0, 50.0),
    'sugarcane_area_pct': (-50.0, 50.0),
    'historical_yield_lag_pct': (-30.0, 30.0),
    'rolling_yield_pct': (-30.0, 30.0)
}

SCENARIO_ARCHETYPES = {
    'baseline': {
        'name': 'Baseline Scenario',
        'description': 'No intervention or input modifications (delta = 0).',
        'deltas': {}
    },
    'conservative_improvement': {
        'name': 'Conservative Improvement',
        'description': 'Modest agricultural improvements (+5% rice acreage allocation, +5% lag productivity).',
        'deltas': {
            'rice_area_pct': 5.0,
            'historical_yield_lag_pct': 5.0,
            'rolling_yield_pct': 3.0
        }
    },
    'moderate_improvement': {
        'name': 'Moderate Improvement',
        'description': 'Balanced expansion and intensification (+12% rice acreage, +10% lag productivity).',
        'deltas': {
            'rice_area_pct': 12.0,
            'historical_yield_lag_pct': 10.0,
            'rolling_yield_pct': 7.0
        }
    },
    'stress_scenario': {
        'name': 'Climate / Market Stress Scenario',
        'description': 'Adverse conditions simulating resource contraction (-15% rice acreage, -15% lag productivity).',
        'deltas': {
            'rice_area_pct': -15.0,
            'historical_yield_lag_pct': -15.0,
            'rolling_yield_pct': -10.0
        }
    },
    'custom': {
        'name': 'Custom Scenario',
        'description': 'User-defined bounded parameter adjustments.',
        'deltas': {}
    }
}


class ScenarioEngine:
    """
    Deterministic scenario execution engine.
    """

    @staticmethod
    def get_supported_features() -> Dict[str, str]:
        """Returns the dictionary of officially supported scenario features."""
        return SUPPORTED_SCENARIO_FEATURES.copy()

    @staticmethod
    def get_scenario_templates() -> Dict[str, Any]:
        """Returns pre-configured scenario archetypes."""
        return SCENARIO_ARCHETYPES.copy()

    @staticmethod
    def validate_scenario_modifications(modifications: Dict[str, Any]) -> Tuple[Dict[str, float], List[str]]:
        """
        Validates feature changes. Rejects or flags unsupported features.
        Returns (valid_modifications, unsupported_warnings).
        """
        valid_mods: Dict[str, float] = {}
        unsupported: List[str] = []

        unsupported_known = ['fertilizer_change', 'irrigation_change', 'climate_stress', 'rainfall_change']

        for k, v in modifications.items():
            if k in SUPPORTED_SCENARIO_FEATURES or k in SCENARIO_BOUNDS:
                try:
                    valid_mods[k] = float(v)
                except (ValueError, TypeError):
                    continue
            else:
                unsupported.append(k)

        return valid_mods, unsupported

    validate_modifications = validate_scenario_modifications

    @staticmethod
    def get_archetype_modifications(scenario_type: str, base_features: Dict[str, float]) -> Dict[str, float]:
        """Calculates modified feature dictionary according to archetype percentage deltas."""
        archetype = SCENARIO_ARCHETYPES.get(scenario_type, SCENARIO_ARCHETYPES['baseline'])
        deltas = archetype.get('deltas', {})
        modified = base_features.copy()

        if 'rice_area_pct' in deltas:
            pct = deltas['rice_area_pct']
            for k in ['RICE AREA (1000 ha)', 'rice_area']:
                if k in modified:
                    modified[k] = round(modified[k] * (1.0 + pct / 100.0), 2)

        if 'historical_yield_lag_pct' in deltas:
            pct = deltas['historical_yield_lag_pct']
            for k in ['RICE_YIELD_LAG1', 'historical_yield_lag']:
                if k in modified:
                    modified[k] = round(modified[k] * (1.0 + pct / 100.0), 2)

        if 'rolling_yield_pct' in deltas:
            pct = deltas['rolling_yield_pct']
            for k in ['RICE_YIELD_ROLL3', 'rolling_yield']:
                if k in modified:
                    modified[k] = round(modified[k] * (1.0 + pct / 100.0), 2)

        return modified

    @staticmethod
    def compute_deltas(
        baseline_pred: float,
        scenario_pred: float,
        baseline_risk: float = 0.0,
        scenario_risk: float = 0.0,
        baseline_spread: float = 0.0,
        scenario_spread: float = 0.0
    ) -> Dict[str, Any]:
        """Calculates absolute and percentage changes between baseline and scenario."""
        yield_delta = round(scenario_pred - baseline_pred, 2)
        yield_pct = round((yield_delta / baseline_pred * 100.0) if baseline_pred > 0 else 0.0, 2)
        risk_delta = round(scenario_risk - baseline_risk, 1)
        spread_delta = round(scenario_spread - baseline_spread, 2)

        return {
            'yield_delta_kg_ha': yield_delta,
            'yield_percent_change': yield_pct,
            'risk_delta': risk_delta,
            'spread_delta_kg_ha': spread_delta,
            'direction': 'positive' if yield_delta > 0 else ('negative' if yield_delta < 0 else 'neutral'),
            'risk_direction': 'increased' if risk_delta > 0 else ('decreased' if risk_delta < 0 else 'unchanged')
        }

    @staticmethod
    def identify_changed_features(
        baseline_features: Dict[str, float],
        scenario_features: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Compares feature values to isolate modified inputs."""
        changed = []
        name_map = SUPPORTED_SCENARIO_FEATURES

        for k, b_val in baseline_features.items():
            s_val = scenario_features.get(k, b_val)
            if abs(s_val - b_val) > 1e-4:
                diff = s_val - b_val
                pct = (diff / b_val * 100.0) if b_val > 0 else 0.0
                changed.append({
                    'feature_key': k,
                    'feature_name': name_map.get(k, k),
                    'baseline_value': round(b_val, 2),
                    'scenario_value': round(s_val, 2),
                    'absolute_change': round(diff, 2),
                    'percent_change': round(pct, 1)
                })
        return changed


scenario_engine = ScenarioEngine()
