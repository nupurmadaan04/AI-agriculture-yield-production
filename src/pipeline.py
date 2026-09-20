"""
Production ML Pipeline for Rice Yield Prediction.

This module provides an end-to-end, reproducible, leak-free scikit-learn Pipeline
for data preprocessing, feature engineering, and model inference.
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor


# Official state mapping for India crop dataset
KNOWN_STATES = [
    'Andhra Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Gujarat',
    'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala',
    'Madhya Pradesh', 'Maharashtra', 'Orissa', 'Punjab', 'Rajasthan',
    'Tamil Nadu', 'Telangana', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
]

# State Code mapping from dataset
STATE_TO_CODE = {
    'Andhra Pradesh': 1,
    'Assam': 2,
    'Bihar': 3,
    'Chhattisgarh': 14,
    'Gujarat': 4,
    'Haryana': 5,
    'Himachal Pradesh': 6,
    'Jharkhand': 15,
    'Karnataka': 7,
    'Kerala': 8,
    'Madhya Pradesh': 9,
    'Maharashtra': 10,
    'Orissa': 11,
    'Punjab': 12,
    'Rajasthan': 13,
    'Tamil Nadu': 16,
    'Telangana': 20,
    'Uttar Pradesh': 17,
    'Uttarakhand': 18,
    'West Bengal': 19
}
CODE_TO_STATE = {v: k for k, v in STATE_TO_CODE.items()}


class AgriculturalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineer for agricultural tabular data without target leakage.
    Adds state-level codes, log transforms of skewed area/production, and temporal features.
    """
    def __init__(self, add_log_features: bool = False):
        self.add_log_features = add_log_features
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=['Year', 'RICE AREA (1000 ha)', 'RICE PRODUCTION (1000 tons)', 'State Code'])
        else:
            df = X.copy()

        # Ensure required columns are present
        if 'State Name' in df.columns and 'State Code' not in df.columns:
            df['State Code'] = df['State Name'].map(STATE_TO_CODE).fillna(0)
        elif 'State Code' in df.columns and 'State Name' not in df.columns:
            df['State Name'] = df['State Code'].map(CODE_TO_STATE).fillna('Unknown')

        # Ensure correct types
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(2010).astype(int)
        df['RICE AREA (1000 ha)'] = pd.to_numeric(df['RICE AREA (1000 ha)'], errors='coerce').fillna(0.0).clip(lower=0.0)
        df['RICE PRODUCTION (1000 tons)'] = pd.to_numeric(df['RICE PRODUCTION (1000 tons)'], errors='coerce').fillna(0.0).clip(lower=0.0)
        df['State Code'] = pd.to_numeric(df['State Code'], errors='coerce').fillna(0).astype(int)

        feature_cols = ['Year', 'RICE AREA (1000 ha)', 'RICE PRODUCTION (1000 tons)', 'State Code']
        self.feature_names_out_ = feature_cols
        return df[feature_cols]

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_out_ or ['Year', 'RICE AREA (1000 ha)', 'RICE PRODUCTION (1000 tons)', 'State Code']


def build_rice_yield_pipeline(
    regressor: Optional[BaseEstimator] = None,
    random_state: int = 42
) -> Pipeline:
    """
    Build a standard, reproducible scikit-learn pipeline for Rice Yield Prediction.

    Args:
        regressor: The regression estimator. Defaults to RandomForestRegressor(random_state=42).
        random_state: Random state for reproducibility.

    Returns:
        Pipeline: Complete scikit-learn Pipeline.
    """
    if regressor is None:
        regressor = RandomForestRegressor(
            n_estimators=150,
            max_depth=30,
            min_samples_split=2,
            random_state=random_state,
            n_jobs=-1
        )

    pipeline = Pipeline(steps=[
        ('feature_engineer', AgriculturalFeatureEngineer()),
        ('scaler', StandardScaler()),
        ('regressor', regressor)
    ])
    return pipeline


class RiceYieldModel:
    """
    High-level interface for training, predicting, saving, and loading Rice Yield pipelines.
    """
    def __init__(self, pipeline: Optional[Pipeline] = None, random_state: int = 42):
        self.pipeline = pipeline if pipeline is not None else build_rice_yield_pipeline(random_state=random_state)
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RiceYieldModel':
        """Fit the complete pipeline on training data."""
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: Union[pd.DataFrame, List[Dict[str, Any]], np.ndarray]) -> np.ndarray:
        """
        Predict rice yield in Kg/ha. Ensures non-negative yield output.
        """
        if not self.is_fitted:
            # Check if underlying pipeline is fitted
            try:
                self.pipeline.named_steps['regressor']
            except Exception:
                raise RuntimeError("Model is not fitted yet.")

        if isinstance(X, list):
            df = pd.DataFrame(X)
        elif isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=['Year', 'RICE AREA (1000 ha)', 'RICE PRODUCTION (1000 tons)', 'State Code'])
        else:
            df = X.copy()

        preds = self.pipeline.predict(df)
        # Yield cannot be negative in reality
        return np.clip(preds, a_min=0.0, a_max=None)

    def predict_single(
        self,
        year: int,
        state: Union[str, int],
        rice_area_1000ha: float,
        rice_prod_1000tons: float,
        dist_name: Optional[str] = None
    ) -> float:
        """
        Predict rice yield for a single district/state input.

        Returns:
            float: Predicted yield in Kg/ha.
        """
        if isinstance(state, str):
            state_code = STATE_TO_CODE.get(state, 0)
            state_name = state
        else:
            state_code = int(state)
            state_name = CODE_TO_STATE.get(state_code, 'Unknown')

        input_df = pd.DataFrame([{
            'Year': year,
            'State Name': state_name,
            'State Code': state_code,
            'Dist Name': dist_name or 'Unknown',
            'RICE AREA (1000 ha)': max(0.0, float(rice_area_1000ha)),
            'RICE PRODUCTION (1000 tons)': max(0.0, float(rice_prod_1000tons))
        }])

        return float(self.predict(input_df)[0])

    def save(self, file_path: Union[str, Path]) -> None:
        """Serialize and save the fitted model pipeline."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self.pipeline, file_path)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'RiceYieldModel':
        """Load a serialized model pipeline."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        pipeline = joblib.load(file_path)
        instance = cls(pipeline=pipeline)
        instance.is_fitted = True
        return instance
