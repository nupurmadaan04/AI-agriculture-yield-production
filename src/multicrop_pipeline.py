"""
Multi-Crop Feature Pipeline Module
==================================
Reusable zero-leakage pre-season lag feature generator.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd


class CropFeaturePipeline:
    """Zero-leakage pre-season lag feature generator."""

    FEATURE_NAMES = [
        "yield_lag_1",
        "yield_lag_2",
        "yield_rolling_3yr_mean",
        "area_lag_1",
        "state_encoded",
        "year",
    ]

    def __init__(self):
        self.state_to_code: Dict[str, int] = {}
        self.crop_train_mean: float = 0.0
        self.dist_train_means: Dict[str, float] = {}

    def fit(self, train_df: pd.DataFrame):
        """Fits categorical encoders and historical imputation statistics strictly on train data."""
        states = sorted(train_df["state"].dropna().unique())
        self.state_to_code = {st: idx for idx, st in enumerate(states)}
        clean_train = train_df.dropna(subset=["yield_kg_ha"])
        self.crop_train_mean = float(clean_train["yield_kg_ha"].mean()) if not clean_train.empty else 0.0
        self.dist_train_means = {
            k: float(v) for k, v in clean_train.groupby("district")["yield_kg_ha"].mean().to_dict().items()
            if pd.notna(v)
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Constructs shifted lag features ensuring zero same-period lookahead."""
        df_out = df.copy().sort_values(by=["district", "year"])

        # Lag 1: Shift target by 1 within district
        df_out["yield_lag_1"] = df_out.groupby("district")["yield_kg_ha"].shift(1)

        # Lag 2: Shift target by 2 within district
        df_out["yield_lag_2"] = df_out.groupby("district")["yield_kg_ha"].shift(2)

        # 3-Year Rolling Mean: computed strictly on shifted lag
        df_out["yield_rolling_3yr_mean"] = (
            df_out.groupby("district")["yield_kg_ha"]
            .shift(1)
            .rolling(window=3, min_periods=1)
            .mean()
        )

        # Area Lag 1: Shift area by 1 within district
        df_out["area_lag_1"] = df_out.groupby("district")["area_ha"].shift(1)

        # State Encoding
        df_out["state_encoded"] = df_out["state"].map(self.state_to_code).fillna(-1).astype(int)

        # Impute missing lags with historical district/crop train mean
        def get_clean_lag(r):
            val = r["yield_lag_1"]
            if pd.isna(val) or val <= 0:
                d_val = self.dist_train_means.get(r["district"], self.crop_train_mean)
                return d_val if pd.notna(d_val) and d_val > 0 else self.crop_train_mean
            return val

        df_out["yield_lag_1"] = df_out.apply(get_clean_lag, axis=1).fillna(self.crop_train_mean)
        df_out["yield_lag_2"] = df_out["yield_lag_2"].fillna(df_out["yield_lag_1"]).fillna(self.crop_train_mean)
        df_out["yield_rolling_3yr_mean"] = df_out["yield_rolling_3yr_mean"].fillna(df_out["yield_lag_1"]).fillna(self.crop_train_mean)
        df_out["area_lag_1"] = df_out["area_lag_1"].fillna(0.0)

        return df_out
