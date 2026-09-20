"""
Multi-Crop Exogenous Feature Ablation Engine (Day 22).

Executes systematic 5-tier ablation experiments (EXP-22A through EXP-22E)
across identical expanding walk-forward folds (2014–2017) against Model A (Historical ML)
and Model C (Statistical Baseline) for all 14 evaluated commodities.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

EVALUATED_CROPS = [
    "Oilseeds", "Chickpea", "Kharif Sorghum", "Minor Pulses",
    "Maize", "Wheat", "Sugarcane", "Rice", "Sesamum",
    "Pigeonpea", "Rapeseed and Mustard", "Groundnut", "Sorghum", "Pearl Millet"
]

FOLDS = [
    {"fold_id": 1, "test_year": 2014, "train_max_year": 2013},
    {"fold_id": 2, "test_year": 2015, "train_max_year": 2014},
    {"fold_id": 3, "test_year": 2016, "train_max_year": 2015},
    {"fold_id": 4, "test_year": 2017, "train_max_year": 2016},
]

ABLATION_EXPERIMENTS = [
    {
        "experiment_id": "EXP-22A",
        "name": "Historical Only (Model A)",
        "feature_group": "HISTORICAL",
        "features": [
            "yield_lag_1", "yield_lag_2", "yield_rolling_3yr_mean",
            "yield_rolling_3yr_std", "area_lag_1", "area_rolling_3yr_mean",
            "district_encoded"
        ]
    },
    {
        "experiment_id": "EXP-22B",
        "name": "Historical + Rainfall",
        "feature_group": "HIST_PLUS_RAINFALL",
        "features": [
            "yield_lag_1", "yield_lag_2", "yield_rolling_3yr_mean",
            "yield_rolling_3yr_std", "area_lag_1", "area_rolling_3yr_mean",
            "district_encoded",
            "preseason_rainfall_total", "preseason_rainfall_anomaly", "rainfall_lag1_total"
        ]
    },
    {
        "experiment_id": "EXP-22C",
        "name": "Historical + Temperature",
        "feature_group": "HIST_PLUS_TEMP",
        "features": [
            "yield_lag_1", "yield_lag_2", "yield_rolling_3yr_mean",
            "yield_rolling_3yr_std", "area_lag_1", "area_rolling_3yr_mean",
            "district_encoded",
            "preseason_temp_mean", "preseason_temp_max", "preseason_temp_anomaly"
        ]
    },
    {
        "experiment_id": "EXP-22D",
        "name": "Historical + Rainfall + Temperature",
        "feature_group": "HIST_PLUS_WEATHER",
        "features": [
            "yield_lag_1", "yield_lag_2", "yield_rolling_3yr_mean",
            "yield_rolling_3yr_std", "area_lag_1", "area_rolling_3yr_mean",
            "district_encoded",
            "preseason_rainfall_total", "preseason_rainfall_anomaly", "rainfall_lag1_total",
            "preseason_temp_mean", "preseason_temp_max", "preseason_temp_anomaly"
        ]
    },
    {
        "experiment_id": "EXP-22E",
        "name": "Historical + All Exogenous (Model B)",
        "feature_group": "HIST_PLUS_ALL_EXO",
        "features": [
            "yield_lag_1", "yield_lag_2", "yield_rolling_3yr_mean",
            "yield_rolling_3yr_std", "area_lag_1", "area_rolling_3yr_mean",
            "district_encoded",
            "preseason_rainfall_total", "preseason_rainfall_anomaly", "rainfall_lag1_total",
            "preseason_temp_mean", "preseason_temp_max", "preseason_temp_anomaly",
            "preseason_soil_moisture", "preseason_dry_spell_days",
            "preseason_aridity_index", "irrigation_ratio_lag1"
        ]
    }
]

# Crop selected algorithm mapping from Day 19/21
CROP_ALGORITHM_MAP = {
    "Oilseeds": "RandomForestRegressor",
    "Chickpea": "GradientBoostingRegressor",
    "Kharif Sorghum": "RandomForestRegressor",
    "Minor Pulses": "GradientBoostingRegressor",
    "Maize": "RandomForestRegressor",
    "Wheat": "GradientBoostingRegressor",
    "Sugarcane": "GradientBoostingRegressor",
    "Rice": "RandomForestRegressor",
    "Sesamum": "RandomForestRegressor",
    "Pigeonpea": "RandomForestRegressor",
    "Rapeseed and Mustard": "RandomForestRegressor",
    "Groundnut": "RandomForestRegressor",
    "Sorghum": "RandomForestRegressor",
    "Pearl Millet": "RandomForestRegressor",
}


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)


class ExogenousAblationEngine:
    """Runs walk-forward ablation experiments and evaluates feature contributions."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.processed_path = self.base_dir / "Datasets" / "processed" / "exogenous_features.csv"
        self.metadata_dir = self.base_dir / "Datasets" / "metadata"

    def prepare_crop_dataset(self, crop: str) -> pd.DataFrame:
        df = pd.read_csv(self.processed_path)
        crop_df = df[df["crop"] == crop].copy()
        crop_df = crop_df.sort_values(["district", "year"]).reset_index(drop=True)

        # Compute autoregressive lag features
        crop_df["yield_lag_1"] = crop_df.groupby("district")["yield_kg_ha"].shift(1)
        crop_df["yield_lag_2"] = crop_df.groupby("district")["yield_kg_ha"].shift(2)
        crop_df["yield_rolling_3yr_mean"] = (
            crop_df.groupby("district")["yield_kg_ha"]
            .shift(1)
            .rolling(3, min_periods=1)
            .mean()
        )
        crop_df["yield_rolling_3yr_std"] = (
            crop_df.groupby("district")["yield_kg_ha"]
            .shift(1)
            .rolling(3, min_periods=1)
            .std()
            .fillna(0.0)
        )
        crop_df["area_lag_1"] = crop_df.groupby("district")["area_ha"].shift(1)
        crop_df["area_rolling_3yr_mean"] = (
            crop_df.groupby("district")["area_ha"]
            .shift(1)
            .rolling(3, min_periods=1)
            .mean()
        )

        le = LabelEncoder()
        crop_df["district_encoded"] = le.fit_transform(crop_df["district"].astype(str))

        return crop_df

    def run_all_ablations(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes all 5 ablation tiers across all 14 crops and 4 folds.
        """
        fold_results: List[Dict[str, Any]] = []

        print(f"[Exogenous Ablation] Starting evaluation across {len(EVALUATED_CROPS)} crops...")

        for crop in EVALUATED_CROPS:
            print(f"  -> Processing Crop: {crop}")
            crop_df = self.prepare_crop_dataset(crop)
            algo = CROP_ALGORITHM_MAP.get(crop, "RandomForestRegressor")

            for fold in FOLDS:
                fold_id = fold["fold_id"]
                test_year = fold["test_year"]
                train_max_year = fold["train_max_year"]

                train_mask = (crop_df["year"] >= 1966) & (crop_df["year"] <= train_max_year)
                test_mask = crop_df["year"] == test_year

                train_data = crop_df[train_mask].dropna(subset=["yield_kg_ha", "yield_lag_1"])
                test_data = crop_df[test_mask].dropna(subset=["yield_kg_ha", "yield_lag_1"])

                if len(train_data) == 0 or len(test_data) == 0:
                    continue

                y_train = train_data["yield_kg_ha"].values
                y_test = test_data["yield_kg_ha"].values

                # Compute Model C Baseline (Historical District Mean on Train)
                dist_means = train_data.groupby("district")["yield_kg_ha"].mean().to_dict()
                overall_mean = float(np.mean(y_train))
                baseline_preds = test_data["district"].map(dist_means).fillna(overall_mean).values

                base_mae = float(mean_absolute_error(y_test, baseline_preds))
                base_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_preds)))
                base_r2 = float(r2_score(y_test, baseline_preds)) if np.var(y_test) > 0 else 0.0

                # Train Model A first to establish historical reference
                hist_features = ABLATION_EXPERIMENTS[0]["features"]
                X_train_hist = train_data[hist_features].fillna(0.0).values
                X_test_hist = test_data[hist_features].fillna(0.0).values

                if algo == "RandomForestRegressor":
                    model_hist = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                else:
                    model_hist = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)

                model_hist.fit(X_train_hist, y_train)
                hist_preds = model_hist.predict(X_test_hist)
                hist_mae = float(mean_absolute_error(y_test, hist_preds))

                # Now evaluate each ablation experiment
                for exp in ABLATION_EXPERIMENTS:
                    exp_id = exp["experiment_id"]
                    exp_name = exp["name"]
                    features = exp["features"]

                    X_train = train_data[features].fillna(0.0).values
                    X_test = test_data[features].fillna(0.0).values

                    if algo == "RandomForestRegressor":
                        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                    else:
                        model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)

                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    mae = float(mean_absolute_error(y_test, preds))
                    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                    r2 = float(r2_score(y_test, preds)) if np.var(y_test) > 0 else 0.0
                    mape = compute_mape(y_test, preds)
                    smape = compute_smape(y_test, preds)

                    # Improvements
                    imp_vs_base_pct = round(((base_mae - mae) / base_mae) * 100.0, 2)
                    imp_vs_hist_pct = round(((hist_mae - mae) / hist_mae) * 100.0, 2) if exp_id != "EXP-22A" else 0.0

                    win_vs_base = bool(mae < base_mae)
                    win_vs_hist = bool(mae < hist_mae) if exp_id != "EXP-22A" else False

                    fold_results.append({
                        "crop": crop,
                        "fold_id": fold_id,
                        "test_year": test_year,
                        "experiment_id": exp_id,
                        "experiment_name": exp_name,
                        "feature_group": exp["feature_group"],
                        "n_features": len(features),
                        "algorithm": algo,
                        "test_records": len(y_test),
                        "mae": round(mae, 2),
                        "rmse": round(rmse, 2),
                        "r2": round(r2, 4),
                        "mape": round(mape, 2),
                        "smape": round(smape, 2),
                        "baseline_mae": round(base_mae, 2),
                        "historical_mae": round(hist_mae, 2),
                        "improvement_vs_baseline_pct": imp_vs_base_pct,
                        "improvement_vs_historical_pct": imp_vs_hist_pct,
                        "win_vs_baseline": win_vs_base,
                        "win_vs_historical": win_vs_hist
                    })

        df_folds = pd.DataFrame(fold_results)
        fold_csv = self.metadata_dir / "exogenous_fold_results.csv"
        df_folds.to_csv(fold_csv, index=False)

        # Aggregate summary per crop & experiment
        summary_rows: List[Dict[str, Any]] = []
        for (crop, exp_id), g in df_folds.groupby(["crop", "experiment_id"]):
            exp_name = g["experiment_name"].iloc[0]
            feature_group = g["feature_group"].iloc[0]
            algo = g["algorithm"].iloc[0]
            n_feat = g["n_features"].iloc[0]

            mean_mae = round(float(g["mae"].mean()), 2)
            median_mae = round(float(g["mae"].median()), 2)
            mean_rmse = round(float(g["rmse"].mean()), 2)
            mean_r2 = round(float(g["r2"].mean()), 4)
            mean_mape = round(float(g["mape"].mean()), 2)
            mean_smape = round(float(g["smape"].mean()), 2)

            base_mean_mae = round(float(g["baseline_mae"].mean()), 2)
            hist_mean_mae = round(float(g["historical_mae"].mean()), 2)

            mean_imp_base = round(float(g["improvement_vs_baseline_pct"].mean()), 2)
            median_imp_base = round(float(g["improvement_vs_baseline_pct"].median()), 2)
            mean_imp_hist = round(float(g["improvement_vs_historical_pct"].mean()), 2)

            win_rate_base = round((g["win_vs_baseline"].sum() / len(g)) * 100.0, 1)
            win_rate_hist = round((g["win_vs_historical"].sum() / len(g)) * 100.0, 1) if exp_id != "EXP-22A" else 0.0

            summary_rows.append({
                "crop": crop,
                "experiment_id": exp_id,
                "experiment_name": exp_name,
                "feature_group": feature_group,
                "algorithm": algo,
                "n_features": n_feat,
                "mean_mae": mean_mae,
                "median_mae": median_mae,
                "mean_rmse": mean_rmse,
                "mean_r2": mean_r2,
                "mean_mape": mean_mape,
                "mean_smape": mean_smape,
                "baseline_mean_mae": base_mean_mae,
                "historical_mean_mae": hist_mean_mae,
                "mean_improvement_vs_baseline_pct": mean_imp_base,
                "median_improvement_vs_baseline_pct": median_imp_base,
                "mean_improvement_vs_historical_pct": mean_imp_hist,
                "win_rate_vs_baseline": win_rate_base,
                "win_rate_vs_historical": win_rate_hist
            })

        df_summary = pd.DataFrame(summary_rows)
        summary_csv = self.metadata_dir / "exogenous_ablation_results.csv"
        df_summary.to_csv(summary_csv, index=False)

        print(f"[Exogenous Ablation] Saved fold results to {fold_csv} and ablation summary to {summary_csv}")
        return df_folds, df_summary


if __name__ == "__main__":
    engine = ExogenousAblationEngine()
    df_f, df_s = engine.run_all_ablations()
    print("Ablation completed.")
