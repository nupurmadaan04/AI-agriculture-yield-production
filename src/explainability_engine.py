"""
Explainability Engine for Agricultural Decision Intelligence.

Provides dynamic, mathematically grounded model interpretability:
- Global Feature Importance (Model-Native vs. Out-of-Sample Permutation Importance)
- Local Prediction Attribution (Marginal Contribution & Directional Decomposition)
- Controlled Feature Sensitivity Analysis (-10% to +10% domain sweeps)
Strictly adheres to non-causal statistical interpretations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'Models'
DATA_PATH = BASE_DIR / 'Datasets' / 'rice_data_outlier_removed.csv'

FEATURE_LABEL_MAP: Dict[str, str] = {
    'RICE_YIELD_LAG1': 'Previous-Season Rice Yield (t-1)',
    'RICE_YIELD_ROLL3': '3-Year Historical Baseline Yield',
    'TOTAL_CROPPED_AREA': 'Total Cultivated Cropland Area',
    'RICE_AREA_SHARE': 'Rice Land Allocation Share',
    'RICE AREA (1000 ha)': 'Cultivated Rice Area',
    'WHEAT AREA (1000 ha)': 'Wheat Cropland Area',
    'COTTON AREA (1000 ha)': 'Cotton Cropland Area',
    'SUGARCANE AREA (1000 ha)': 'Sugarcane Cropland Area',
    'Year': 'Agricultural Survey Year',
    'State Code': 'State Regional Baseline'
}

DEFAULT_FEATURE_ORDER = [
    'Year', 'State Code', 'RICE AREA (1000 ha)', 'TOTAL_CROPPED_AREA',
    'RICE_AREA_SHARE', 'WHEAT AREA (1000 ha)', 'COTTON AREA (1000 ha)',
    'SUGARCANE AREA (1000 ha)', 'RICE_YIELD_LAG1', 'RICE_YIELD_ROLL3'
]


class ExplainabilityEngine:
    """Core mathematical engine for model explainability and feature attribution."""

    def __init__(self, model_path: Optional[Path] = None, feature_manifest_path: Optional[Path] = None):
        self.model_path = model_path or (MODELS_DIR / 'forecasting_pipeline.pkl')
        self.feature_manifest_path = feature_manifest_path or (MODELS_DIR / 'geographic_feature_manifest.json')
        self._pipeline = None
        self._feature_names = None
        self._cached_global_importance = None
        self._reference_medians = None

    def _load_pipeline_and_features(self):
        if self._pipeline is not None:
            return self._pipeline, self._feature_names

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}")

        self._pipeline = joblib.load(self.model_path)

        # Extract features from pipeline or fallback
        if self.feature_manifest_path.exists():
            try:
                with open(self.feature_manifest_path, 'r') as f:
                    manifest = json.load(f)
                self._feature_names = manifest.get('features', DEFAULT_FEATURE_ORDER)
            except Exception:
                self._feature_names = DEFAULT_FEATURE_ORDER
        else:
            self._feature_names = DEFAULT_FEATURE_ORDER

        return self._pipeline, self._feature_names

    def get_reference_baseline(self) -> Dict[str, float]:
        """Computes or loads reference median feature vector across the panel dataset."""
        if self._reference_medians is not None:
            return self._reference_medians

        defaults = {
            'Year': 2017.0,
            'State Code': 12.0,
            'RICE AREA (1000 ha)': 250.0,
            'TOTAL_CROPPED_AREA': 600.0,
            'RICE_AREA_SHARE': 0.45,
            'WHEAT AREA (1000 ha)': 180.0,
            'COTTON AREA (1000 ha)': 40.0,
            'SUGARCANE AREA (1000 ha)': 25.0,
            'RICE_YIELD_LAG1': 2850.0,
            'RICE_YIELD_ROLL3': 2800.0
        }

        if DATA_PATH.exists():
            try:
                df = pd.read_csv(DATA_PATH)
                for f in DEFAULT_FEATURE_ORDER:
                    if f in df.columns and np.issubdtype(df[f].dtype, np.number):
                        defaults[f] = float(df[f].median())
            except Exception:
                pass

        self._reference_medians = defaults
        return self._reference_medians

    def compute_global_feature_importance(self, n_repeats: int = 5) -> Dict[str, Any]:
        """
        Computes model-native importance and holdout permutation importance.
        Compares rankings and flags methodological disagreements.
        """
        if self._cached_global_importance is not None:
            return self._cached_global_importance

        pipeline, feature_names = self._load_pipeline_and_features()

        # 1. Model-native feature importance
        regressor = pipeline.named_steps.get('regressor', pipeline)
        if hasattr(regressor, 'feature_importances_'):
            native_importances = regressor.feature_importances_
        else:
            native_importances = np.ones(len(feature_names)) / len(feature_names)

        total_native = np.sum(native_importances) if np.sum(native_importances) > 0 else 1.0
        norm_native = native_importances / total_native

        # 2. Out-of-Sample Permutation Importance on test data
        perm_importances = np.zeros(len(feature_names))
        if DATA_PATH.exists():
            try:
                df = pd.read_csv(DATA_PATH)
                # Use holdout out-of-time evaluation slice (Year > 2015)
                test_df = df[df['Year'] > 2015] if 'Year' in df.columns else df.tail(300)
                if len(test_df) < 50:
                    test_df = df.tail(300)

                # Prepare feature matrix
                X_test = pd.DataFrame()
                for feat in feature_names:
                    if feat in test_df.columns:
                        X_test[feat] = test_df[feat]
                    else:
                        X_test[feat] = self.get_reference_baseline().get(feat, 0.0)

                y_test = test_df['RICE YIELD (Kg per ha)'].values if 'RICE YIELD (Kg per ha)' in test_df.columns else np.zeros(len(test_df))

                baseline_preds = pipeline.predict(X_test)
                baseline_mse = mean_squared_error(y_test, baseline_preds)

                for idx, feat in enumerate(feature_names):
                    losses = []
                    for _ in range(n_repeats):
                        X_perm = X_test.copy()
                        X_perm[feat] = np.random.permutation(X_perm[feat].values)
                        perm_preds = pipeline.predict(X_perm)
                        perm_mse = mean_squared_error(y_test, perm_preds)
                        losses.append(max(0.0, perm_mse - baseline_mse))
                    perm_importances[idx] = float(np.mean(losses))

                total_perm = np.sum(perm_importances) if np.sum(perm_importances) > 0 else 1.0
                norm_perm = perm_importances / total_perm
            except Exception:
                norm_perm = norm_native.copy()
        else:
            norm_perm = norm_native.copy()

        # Build feature items
        features_list = []
        for i, feat in enumerate(feature_names):
            features_list.append({
                'feature': feat,
                'feature_label': FEATURE_LABEL_MAP.get(feat, feat),
                'native_importance': round(float(norm_native[i]), 4),
                'permutation_importance': round(float(norm_perm[i]), 4),
                'native_rank': 0,
                'permutation_rank': 0,
                'rank_agreement': True
            })

        # Assign ranks
        features_list.sort(key=lambda x: x['native_importance'], reverse=True)
        for rank, item in enumerate(features_list, 1):
            item['native_rank'] = rank

        perm_sorted = sorted(features_list, key=lambda x: x['permutation_importance'], reverse=True)
        for rank, item in enumerate(perm_sorted, 1):
            item['permutation_rank'] = rank
            item['rank_agreement'] = abs(item['native_rank'] - item['permutation_rank']) <= 2

        # Overall summary
        result = {
            'model_id': 'exogenous_rf_forecaster',
            'model_name': 'Exogenous Random Forest Forecaster',
            'version': '2.1.0',
            'dataset_version': 'ICRISAT 1966–2017 Cleaned Panel',
            'explanation_method': 'Model-Native Gini Impurity & Out-of-Sample Permutation Importance',
            'total_features_evaluated': len(feature_names),
            'top_feature': features_list[0]['feature'],
            'top_feature_label': features_list[0]['feature_label'],
            'features': features_list,
            'scientific_disclaimer': (
                'Feature importances describe model decision split frequencies and loss sensitivities '
                'within historical training distributions. They do not constitute agronomic causal efficacy.'
            )
        }

        self._cached_global_importance = result
        return result

    def explain_local_prediction(
        self,
        features: Dict[str, float],
        entity: Optional[str] = None,
        year: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Decomposes a specific model prediction into directional feature contributions.
        Compares against the empirical baseline reference median.
        """
        pipeline, feature_names = self._load_pipeline_and_features()
        baseline_dict = self.get_reference_baseline()

        # 1. Compute Full Prediction
        input_row = pd.DataFrame([{f: features.get(f, baseline_dict.get(f, 0.0)) for f in feature_names}])
        prediction = float(pipeline.predict(input_row)[0])

        # 2. Compute Baseline Reference Prediction
        baseline_row = pd.DataFrame([{f: baseline_dict.get(f, 0.0) for f in feature_names}])
        baseline_prediction = float(pipeline.predict(baseline_row)[0])

        # 3. Marginal Feature Contribution via Reference Perturbation
        contributions: List[Dict[str, Any]] = []
        total_delta = prediction - baseline_prediction

        for feat in feature_names:
            val = float(features.get(feat, baseline_dict.get(feat, 0.0)))
            base_val = float(baseline_dict.get(feat, 0.0))

            # One-feature substitution into baseline
            perturbed_row = baseline_row.copy()
            perturbed_row[feat] = val
            perturbed_pred = float(pipeline.predict(perturbed_row)[0])
            marginal_delta = perturbed_pred - baseline_prediction

            direction = 'POSITIVE' if marginal_delta >= 0 else 'NEGATIVE'

            contributions.append({
                'feature': feat,
                'feature_label': FEATURE_LABEL_MAP.get(feat, feat),
                'feature_value': val,
                'baseline_value': base_val,
                'contribution_kg_ha': round(marginal_delta, 2),
                'contribution_direction': direction,
                'relative_influence_pct': 0.0  # Normalized below
            })

        # Normalize relative influence
        sum_abs_deltas = sum(abs(c['contribution_kg_ha']) for c in contributions) or 1.0
        for c in contributions:
            c['relative_influence_pct'] = round((abs(c['contribution_kg_ha']) / sum_abs_deltas) * 100.0, 1)

        # Sort contributions by absolute magnitude
        contributions.sort(key=lambda x: abs(x['contribution_kg_ha']), reverse=True)

        positive_drivers = [c for c in contributions if c['contribution_direction'] == 'POSITIVE']
        negative_drivers = [c for c in contributions if c['contribution_direction'] == 'NEGATIVE']

        return {
            'entity': entity or 'Custom Agronomic Query',
            'year': year or int(features.get('Year', 2017)),
            'prediction_kg_ha': round(prediction, 2),
            'baseline_reference_kg_ha': round(baseline_prediction, 2),
            'prediction_delta_kg_ha': round(total_delta, 2),
            'model_version': '2.1.0',
            'dataset_version': 'ICRISAT 1966–2017 Panel',
            'explanation_method': 'Marginal Reference Perturbation Attribution',
            'top_positive_features': [p['feature_label'] for p in positive_drivers[:3]],
            'top_negative_features': [n['feature_label'] for n in negative_drivers[:3]],
            'feature_contributions': contributions,
            'scientific_disclaimer': (
                'Feature contributions quantify how input deviations shift model output relative to median reference values. '
                'They reflect empirical statistical associations in historical data, not agronomic causation.'
            )
        }

    def compute_feature_sensitivity(
        self,
        base_features: Dict[str, float],
        target_features: Optional[List[str]] = None,
        steps: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Executes controlled feature sensitivity sweeps across discrete perturbation steps:
        [-10%, -5%, 0%, +5%, +10%] while enforcing domain boundaries.
        """
        pipeline, feature_names = self._load_pipeline_and_features()
        baseline_dict = self.get_reference_baseline()

        target_features = target_features or ['RICE_YIELD_LAG1', 'RICE AREA (1000 ha)', 'RICE_AREA_SHARE', 'TOTAL_CROPPED_AREA']
        steps = steps or [-0.10, -0.05, 0.0, 0.05, 0.10]

        # Base prediction
        base_row = pd.DataFrame([{f: base_features.get(f, baseline_dict.get(f, 0.0)) for f in feature_names}])
        base_pred = float(pipeline.predict(base_row)[0])

        curves: Dict[str, List[Dict[str, Any]]] = {}

        for feat in target_features:
            if feat not in feature_names:
                continue

            orig_val = float(base_features.get(feat, baseline_dict.get(feat, 0.0)))
            feat_curve = []

            for delta in steps:
                perturbed_val = orig_val * (1.0 + delta)
                # Enforce physical non-negative boundaries
                perturbed_val = max(0.0, perturbed_val)

                row = base_row.copy()
                row[feat] = perturbed_val
                pred = float(pipeline.predict(row)[0])
                pred_delta = pred - base_pred
                rel_delta_pct = (pred_delta / base_pred * 100.0) if base_pred > 0 else 0.0

                feat_curve.append({
                    'step_pct': round(delta * 100.0, 1),
                    'perturbed_value': round(perturbed_val, 2),
                    'predicted_yield_kg_ha': round(pred, 2),
                    'prediction_delta_kg_ha': round(pred_delta, 2),
                    'relative_delta_pct': round(rel_delta_pct, 2)
                })

            curves[feat] = feat_curve

        return {
            'model_id': 'exogenous_rf_forecaster',
            'model_version': '2.1.0',
            'base_prediction_kg_ha': round(base_pred, 2),
            'tested_features': target_features,
            'perturbation_steps_pct': [round(s * 100.0, 1) for s in steps],
            'sensitivity_curves': curves,
            'scientific_disclaimer': (
                'Perturbations describe model response curves across modified inputs and must not be interpreted as causal intervention impacts.'
            )
        }


explainability_engine = ExplainabilityEngine()
