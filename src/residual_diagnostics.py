"""
Residual Diagnostics, Systematic Bias Detection & Empirical Intervals Engine (Day 23).

Decomposes prediction errors into directional bias, tail spreads, yield quantiles,
district distributions, and evaluates empirical ensemble coverage (P10-P90).
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

from src.exogenous_ablation import (
    EVALUATED_CROPS,
    CROP_ALGORITHM_MAP,
    FOLDS,
    ExogenousAblationEngine
)
from src.strategy_evaluation import OPERATIONAL_POLICIES


class ResidualDiagnosticsEngine:
    """Executes granular residual diagnostics and bias classification."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.metadata_dir = self.base_dir / "Datasets" / "metadata"
        self.ablation_engine = ExogenousAblationEngine(self.base_dir)

    def run_full_residual_diagnostics(self) -> Dict[str, pd.DataFrame]:
        """
        Calculates all residual diagnostics, quantile error regimes, district-level errors,
        systematic bias metrics, and empirical ensemble intervals.
        """
        hist_features = [
            "yield_lag_1", "yield_lag_2", "yield_rolling_3yr_mean",
            "yield_rolling_3yr_std", "area_lag_1", "area_rolling_3yr_mean",
            "district_encoded"
        ]

        overall_diagnostics: List[Dict[str, Any]] = []
        year_analysis: List[Dict[str, Any]] = []
        district_analysis: List[Dict[str, Any]] = []
        bias_records: List[Dict[str, Any]] = []
        interval_records: List[Dict[str, Any]] = []

        print("[Residual Diagnostics] Processing multi-fold residuals across 14 crops...")

        for crop in EVALUATED_CROPS:
            crop_df = self.ablation_engine.prepare_crop_dataset(crop)
            algo = CROP_ALGORITHM_MAP.get(crop, "RandomForestRegressor")
            policy = OPERATIONAL_POLICIES.get(crop, {"policy_type": "STATISTICAL_BASELINE_PRIMARY"})

            all_residuals: List[float] = []
            all_abs_errors: List[float] = []
            all_actuals: List[float] = []
            all_preds: List[float] = []

            # Multi-fold accumulation
            crop_eval_records: List[Dict[str, Any]] = []

            for fold in FOLDS:
                fold_id = fold["fold_id"]
                test_year = fold["test_year"]
                train_max_year = fold["train_max_year"]

                train_data = crop_df[(crop_df["year"] >= 1966) & (crop_df["year"] <= train_max_year)].dropna(subset=["yield_kg_ha", "yield_lag_1"])
                test_data = crop_df[crop_df["year"] == test_year].dropna(subset=["yield_kg_ha", "yield_lag_1"]).copy()

                if len(train_data) == 0 or len(test_data) == 0:
                    continue

                y_train = train_data["yield_kg_ha"].values
                y_test = test_data["yield_kg_ha"].values

                # Fit model / baselines
                dist_means = train_data.groupby("district")["yield_kg_ha"].mean().to_dict()
                overall_mean = float(np.mean(y_train))
                base_preds = test_data["district"].map(dist_means).fillna(overall_mean).values

                X_train = train_data[hist_features].fillna(0.0).values
                X_test = test_data[hist_features].fillna(0.0).values

                if algo == "RandomForestRegressor":
                    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                else:
                    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)

                model.fit(X_train, y_train)
                ml_preds = model.predict(X_test)

                # Generate operational predictions
                if policy["policy_type"] == "PRIMARY_ML_WITH_FALLBACK":
                    preds = ml_preds
                elif policy["policy_type"] == "BASELINE_PRIMARY_CONDITIONAL_ML":
                    q25, q75 = np.percentile(y_train, [25, 75])
                    preds = np.array([
                        (0.5 * ml_preds[i] + 0.5 * base_preds[i]) if (q25 <= dist_means.get(r["district"], overall_mean) <= q75) else base_preds[i]
                        for i, (_, r) in enumerate(test_data.iterrows())
                    ])
                else:
                    preds = base_preds

                # Residual = predicted - actual
                residuals = preds - y_test
                abs_errors = np.abs(residuals)

                test_data["predicted_yield"] = preds
                test_data["residual"] = residuals
                test_data["abs_error"] = abs_errors
                test_data["fold_id"] = fold_id

                all_residuals.extend(residuals.tolist())
                all_abs_errors.extend(abs_errors.tolist())
                all_actuals.extend(y_test.tolist())
                all_preds.extend(preds.tolist())

                # Year-specific analysis
                mean_res_yr = float(np.mean(residuals))
                med_res_yr = float(np.median(residuals))
                mae_yr = float(np.mean(abs_errors))
                rmse_yr = float(np.sqrt(np.mean(residuals ** 2)))

                year_analysis.append({
                    "crop": crop,
                    "year": test_year,
                    "fold_id": fold_id,
                    "records": len(y_test),
                    "mean_residual": round(mean_res_yr, 2),
                    "median_residual": round(med_res_yr, 2),
                    "mae": round(mae_yr, 2),
                    "rmse": round(rmse_yr, 2),
                    "p90_abs_error": round(float(np.percentile(abs_errors, 90)), 2),
                    "temporal_regime": "DROUGHT_SHOCK" if test_year in [2014, 2015] else ("RECOVERY_SHOCK" if test_year == 2016 else "NORMAL_HARVEST")
                })

                # Empirical Ensemble Interval Calculation (Tree Quantiles for RF)
                if algo == "RandomForestRegressor":
                    tree_preds = np.array([tree.predict(X_test) for tree in model.estimators_])  # (n_trees, n_samples)
                    p10 = np.percentile(tree_preds, 10, axis=0)
                    p50 = np.percentile(tree_preds, 50, axis=0)
                    p90 = np.percentile(tree_preds, 90, axis=0)

                    in_interval = (y_test >= p10) & (y_test <= p90)
                    cov_pct = float(np.mean(in_interval) * 100.0)
                    avg_width = float(np.mean(p90 - p10))

                    interval_records.append({
                        "crop": crop,
                        "test_year": test_year,
                        "fold_id": fold_id,
                        "algorithm": "RandomForestRegressor",
                        "records_evaluated": len(y_test),
                        "empirical_coverage_pct": round(cov_pct, 1),
                        "average_interval_width": round(avg_width, 1),
                        "interval_type": "EMPIRICAL_ENSEMBLE_SPREAD",
                        "calibration_label": "Empirical ensemble interval, not a statistically calibrated prediction interval."
                    })

                for _, row in test_data.iterrows():
                    crop_eval_records.append({
                        "district": row["district"],
                        "state": row["state"],
                        "residual": row["residual"],
                        "abs_error": row["abs_error"],
                        "actual": row["yield_kg_ha"],
                        "predicted": row["predicted_yield"]
                    })

            # District-level aggregations (retaining N >= 3)
            df_crop_records = pd.DataFrame(crop_eval_records)
            if not df_crop_records.empty:
                for dist, g in df_crop_records.groupby("district"):
                    n_dist = len(g)
                    if n_dist >= 3:
                        district_analysis.append({
                            "crop": crop,
                            "district": dist,
                            "state": g["state"].iloc[0],
                            "observations": n_dist,
                            "mean_residual": round(float(g["residual"].mean()), 2),
                            "median_residual": round(float(g["residual"].median()), 2),
                            "mae": round(float(g["abs_error"].mean()), 2),
                            "rmse": round(float(np.sqrt((g["residual"] ** 2).mean())), 2),
                            "p90_error": round(float(np.percentile(g["abs_error"], 90)), 2),
                            "is_high_error_district": bool(g["abs_error"].mean() > np.mean(all_abs_errors) * 1.5)
                        })

            # Overall Crop Residual Statistics
            all_res_arr = np.array(all_residuals)
            all_abs_arr = np.array(all_abs_errors)
            all_act_arr = np.array(all_actuals)

            mean_res = float(np.mean(all_res_arr))
            med_res = float(np.median(all_res_arr))
            std_res = float(np.std(all_res_arr))
            mae_total = float(np.mean(all_abs_arr))
            rmse_total = float(np.sqrt(np.mean(all_res_arr ** 2)))
            p25_abs = float(np.percentile(all_abs_arr, 25))
            p50_abs = float(np.percentile(all_abs_arr, 50))
            p75_abs = float(np.percentile(all_abs_arr, 75))
            p90_abs = float(np.percentile(all_abs_arr, 90))
            p95_abs = float(np.percentile(all_abs_arr, 95))

            # Quantile breakdown (Q1 to Q4 by actual yield)
            q_edges = np.percentile(all_act_arr, [0, 25, 50, 75, 100])
            mae_q1 = float(np.mean(all_abs_arr[(all_act_arr >= q_edges[0]) & (all_act_arr <= q_edges[1])]))
            mae_q2 = float(np.mean(all_abs_arr[(all_act_arr > q_edges[1]) & (all_act_arr <= q_edges[2])]))
            mae_q3 = float(np.mean(all_abs_arr[(all_act_arr > q_edges[2]) & (all_act_arr <= q_edges[3])]))
            mae_q4 = float(np.mean(all_abs_arr[(all_act_arr > q_edges[3]) & (all_act_arr <= q_edges[4])]))

            overall_diagnostics.append({
                "crop": crop,
                "total_eval_samples": len(all_res_arr),
                "mean_residual": round(mean_res, 2),
                "median_residual": round(med_res, 2),
                "std_residual": round(std_res, 2),
                "mae": round(mae_total, 2),
                "rmse": round(rmse_total, 2),
                "p25_abs_error": round(p25_abs, 2),
                "p50_abs_error": round(p50_abs, 2),
                "p75_abs_error": round(p75_abs, 2),
                "p90_abs_error": round(p90_abs, 2),
                "p95_abs_error": round(p95_abs, 2),
                "mae_q1_lowest_yield": round(mae_q1, 2),
                "mae_q2_lower_mid_yield": round(mae_q2, 2),
                "mae_q3_upper_mid_yield": round(mae_q3, 2),
                "mae_q4_highest_yield": round(mae_q4, 2)
            })

            # Systematic Bias Detection
            mean_actual_yield = float(np.mean(all_act_arr))
            normalized_mean_error_pct = (mean_res / max(1.0, mean_actual_yield)) * 100.0

            # Deterministic rule: |NME| > 3.0% indicates bias
            if normalized_mean_error_pct > 3.0:
                bias_status = "OVER_PREDICTION_BIAS"
                bias_desc = f"Systematic over-prediction of +{normalized_mean_error_pct:.2f}% relative to actual yield mean."
            elif normalized_mean_error_pct < -3.0:
                bias_status = "UNDER_PREDICTION_BIAS"
                bias_desc = f"Systematic under-prediction of {normalized_mean_error_pct:.2f}% relative to actual yield mean."
            else:
                bias_status = "NO_CLEAR_BIAS"
                bias_desc = f"Unbiased residuals within acceptable tolerance ({normalized_mean_error_pct:.2f}% deviation)."

            bias_records.append({
                "crop": crop,
                "mean_actual_yield": round(mean_actual_yield, 2),
                "mean_residual": round(mean_res, 2),
                "median_residual": round(med_res, 2),
                "normalized_mean_error_pct": round(normalized_mean_error_pct, 2),
                "bias_status": bias_status,
                "bias_description": bias_desc,
                "bias_threshold_rule": "OVER if NME > +3%, UNDER if NME < -3%, else NO_CLEAR_BIAS"
            })

        # Save all artifacts
        df_overall = pd.DataFrame(overall_diagnostics)
        df_year = pd.DataFrame(year_analysis)
        df_dist = pd.DataFrame(district_analysis)
        df_bias = pd.DataFrame(bias_records)
        df_int = pd.DataFrame(interval_records)

        p_overall = self.metadata_dir / "residual_diagnostics.csv"
        p_year = self.metadata_dir / "residual_year_analysis.csv"
        p_dist = self.metadata_dir / "residual_district_analysis.csv"
        p_bias = self.metadata_dir / "prediction_bias_analysis.csv"
        p_int = self.metadata_dir / "empirical_interval_analysis.csv"

        df_overall.to_csv(p_overall, index=False)
        df_year.to_csv(p_year, index=False)
        df_dist.to_csv(p_dist, index=False)
        df_bias.to_csv(p_bias, index=False)
        df_int.to_csv(p_int, index=False)

        print("[Residual Diagnostics] Exported all 5 residual and bias diagnostic files.")
        return {
            "overall": df_overall,
            "year": df_year,
            "district": df_dist,
            "bias": df_bias,
            "intervals": df_int
        }


if __name__ == "__main__":
    engine = ResidualDiagnosticsEngine()
    engine.run_full_residual_diagnostics()
