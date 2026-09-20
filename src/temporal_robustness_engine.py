"""
Temporal Walk-Forward Validation Engine (Day 20)
================================================
Executes chronological walk-forward validation across expanding windows
for all 14 MODEL_READY crops to assess model stability, baseline superiority,
and empirical robustness with zero test contamination.
"""

import os
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

from src.multicrop_pipeline import CropFeaturePipeline


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Symmetric Mean Absolute Percentage Error (%) safely."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true)
    mask = denom > 1e-6
    if not np.any(mask):
        return 0.0
    return float(np.mean(diff[mask] / denom[mask]) * 100.0)


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Absolute Percentage Error (%) safely."""
    mask = np.abs(y_true) > 1e-6
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


class TemporalWalkForwardEngine:
    """Executes expanding-window walk-forward validation across multiple historical origins."""

    def __init__(self, data_path: str = "Datasets/processed/agricultural_panel.csv",
                 readiness_path: str = "Datasets/metadata/crop_model_readiness.csv"):
        self.data_path = data_path
        self.readiness_path = readiness_path
        self.df = pd.read_csv(data_path)
        self.readiness_df = pd.read_csv(readiness_path)
        self.model_ready_crops = self.readiness_df[
            self.readiness_df["readiness_status"] == "MODEL_READY"
        ]["crop"].tolist()

    def get_valid_folds(self, crop_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Determines valid expanding-window walk-forward folds."""
        years = sorted(crop_df["year"].unique())
        # We need at least 2010 for lags, 2011-2013 for initial training, test starting 2014
        folds = []
        test_years = [y for y in [2014, 2015, 2016, 2017] if y in years]

        for idx, test_yr in enumerate(test_years, 1):
            train_years = [y for y in years if 2011 <= y < test_yr]
            if len(train_years) >= 2:
                folds.append({
                    "fold_id": idx,
                    "train_years": train_years,
                    "test_year": test_yr,
                })
        return folds

    def evaluate_baselines_on_fold(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Evaluates statistical baselines on a specific temporal fold."""
        # 1. Historical District Mean
        train_clean = train_df.dropna(subset=["yield_kg_ha"])
        overall_mean = float(train_clean["yield_kg_ha"].mean()) if not train_clean.empty else 0.0
        dist_means = train_clean.groupby("district")["yield_kg_ha"].mean().to_dict()
        
        # Clean district mean map ensuring no NaNs
        cleaned_dist_means = {k: v for k, v in dist_means.items() if pd.notna(v)}
        pred_mean = test_df["district"].map(cleaned_dist_means).fillna(overall_mean).values
        pred_mean = np.nan_to_num(pred_mean, nan=overall_mean)

        # 2. Naive Persistence (t-1 yield)
        pred_persistence = np.nan_to_num(test_df["yield_lag_1"].fillna(overall_mean).values, nan=overall_mean)

        # 3. Linear District Trend
        pred_trend_list = []
        for _, row in test_df.iterrows():
            d = row["district"]
            d_hist = train_clean[train_clean["district"] == d].sort_values("year")
            if len(d_hist) >= 3 and d_hist["yield_kg_ha"].nunique() > 1:
                x = d_hist["year"].values
                y = d_hist["yield_kg_ha"].values
                try:
                    slope, intercept = np.polyfit(x, y, 1)
                    p = slope * row["year"] + intercept
                    if pd.isna(p) or np.isinf(p):
                        p = cleaned_dist_means.get(d, overall_mean)
                except Exception:
                    p = cleaned_dist_means.get(d, overall_mean)
                pred_trend_list.append(max(float(p), 0.0))
            else:
                pred_trend_list.append(cleaned_dist_means.get(d, overall_mean))
        pred_trend = np.nan_to_num(np.array(pred_trend_list, dtype=float), nan=overall_mean)

        y_true = test_df["yield_kg_ha"].values

        baselines = {}
        for name, preds in [
            ("Historical District Mean", pred_mean),
            ("Naive Persistence", pred_persistence),
            ("Linear District Trend", pred_trend),
        ]:
            mae = float(mean_absolute_error(y_true, preds))
            rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
            r2 = float(r2_score(y_true, preds)) if np.var(y_true) > 1e-6 else 0.0
            mape = compute_mape(y_true, preds)
            smape = compute_smape(y_true, preds)
            residuals = y_true - preds

            baselines[name] = {
                "predictions": preds,
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "r2": round(r2, 4),
                "mape": round(mape, 2),
                "smape": round(smape, 2),
                "residuals": residuals,
                "mean_residual": round(float(np.mean(residuals)), 2),
                "std_residual": round(float(np.std(residuals)), 2),
            }
        return baselines

    def run_walk_forward_for_crop(self, crop: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Runs walk-forward validation across all folds for a given crop."""
        crop_df = self.df[self.df["crop"] == crop].copy().sort_values(["district", "year"])
        folds = self.get_valid_folds(crop_df)

        fold_results = []
        feature_importances_rf = []
        feature_importances_gb = []

        for fold in folds:
            fold_id = fold["fold_id"]
            train_years = fold["train_years"]
            test_year = fold["test_year"]

            # Filter data strictly by years
            train_raw = crop_df[crop_df["year"].isin(train_years)].copy()
            # Test slice includes historical context up to test_year for lag calculation
            panel_up_to_test = crop_df[crop_df["year"] <= test_year].copy()

            # Preprocessing strictly on training slice
            pipeline = CropFeaturePipeline()
            pipeline.fit(train_raw)

            # Transform panel up to test year
            panel_trans = pipeline.transform(panel_up_to_test)
            train_proc = panel_trans[panel_trans["year"].isin(train_years)].dropna(subset=["yield_kg_ha"])
            test_proc = panel_trans[panel_trans["year"] == test_year].dropna(subset=["yield_kg_ha"])
            train_raw_clean = train_raw.dropna(subset=["yield_kg_ha"])

            X_train = np.nan_to_num(train_proc[CropFeaturePipeline.FEATURE_NAMES].values, nan=0.0)
            y_train = train_proc["yield_kg_ha"].values
            X_test = np.nan_to_num(test_proc[CropFeaturePipeline.FEATURE_NAMES].values, nan=0.0)
            y_test = test_proc["yield_kg_ha"].values

            # Evaluate Baselines
            baselines = self.evaluate_baselines_on_fold(train_raw_clean, test_proc)
            best_baseline_name = min(baselines, key=lambda k: baselines[k]["mae"])
            best_base = baselines[best_baseline_name]

            # Train Candidate 1: Random Forest
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            pred_rf = rf.predict(X_test)
            mae_rf = float(mean_absolute_error(y_test, pred_rf))
            rmse_rf = float(np.sqrt(mean_squared_error(y_test, pred_rf)))
            r2_rf = float(r2_score(y_test, pred_rf)) if np.var(y_test) > 1e-6 else 0.0
            mape_rf = compute_mape(y_test, pred_rf)
            smape_rf = compute_smape(y_test, pred_rf)
            res_rf = y_test - pred_rf
            feature_importances_rf.append(rf.feature_importances_)

            # Train Candidate 2: Gradient Boosting
            gb = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                min_samples_leaf=4,
                random_state=42,
            )
            gb.fit(X_train, y_train)
            pred_gb = gb.predict(X_test)
            mae_gb = float(mean_absolute_error(y_test, pred_gb))
            rmse_gb = float(np.sqrt(mean_squared_error(y_test, pred_gb)))
            r2_gb = float(r2_score(y_test, pred_gb)) if np.var(y_test) > 1e-6 else 0.0
            mape_gb = compute_mape(y_test, pred_gb)
            smape_gb = compute_smape(y_test, pred_gb)
            res_gb = y_test - pred_gb
            feature_importances_gb.append(gb.feature_importances_)

            # Record metrics per candidate on this fold
            candidates = {
                "Historical District Mean": baselines["Historical District Mean"],
                "Naive Persistence": baselines["Naive Persistence"],
                "Linear District Trend": baselines["Linear District Trend"],
                "RandomForestRegressor": {
                    "mae": round(mae_rf, 2),
                    "rmse": round(rmse_rf, 2),
                    "r2": round(r2_rf, 4),
                    "mape": round(mape_rf, 2),
                    "smape": round(smape_rf, 2),
                    "residuals": res_rf,
                    "mean_residual": round(float(np.mean(res_rf)), 2),
                    "std_residual": round(float(np.std(res_rf)), 2),
                },
                "GradientBoostingRegressor": {
                    "mae": round(mae_gb, 2),
                    "rmse": round(rmse_gb, 2),
                    "r2": round(r2_gb, 4),
                    "mape": round(mape_gb, 2),
                    "smape": round(smape_gb, 2),
                    "residuals": res_gb,
                    "mean_residual": round(float(np.mean(res_gb)), 2),
                    "std_residual": round(float(np.std(res_gb)), 2),
                },
            }

            for model_name, m_info in candidates.items():
                is_ml = model_name in ["RandomForestRegressor", "GradientBoostingRegressor"]
                win_vs_base = m_info["mae"] < best_base["mae"] if is_ml else False
                imp_vs_base = ((best_base["mae"] - m_info["mae"]) / best_base["mae"] * 100.0) if is_ml else 0.0

                fold_results.append({
                    "crop": crop,
                    "fold_id": fold_id,
                    "train_start_year": min(train_years),
                    "train_end_year": max(train_years),
                    "test_year": test_year,
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "model": model_name,
                    "mae": m_info["mae"],
                    "rmse": m_info["rmse"],
                    "r2": m_info["r2"],
                    "mape": m_info["mape"],
                    "smape": m_info["smape"],
                    "best_baseline_model": best_baseline_name,
                    "best_baseline_mae": best_base["mae"],
                    "win_vs_baseline": win_vs_base,
                    "mae_improvement_pct": round(imp_vs_base, 2),
                    "mean_residual": m_info["mean_residual"],
                    "std_residual": m_info["std_residual"],
                })

        # Calculate crop-level robustness summary across folds
        df_f = pd.DataFrame(fold_results)
        summary = self._summarize_crop_robustness(crop, df_f, feature_importances_rf, feature_importances_gb)
        return fold_results, summary

    def _summarize_crop_robustness(self, crop: str, df_f: pd.DataFrame,
                                   fi_rf: List[np.ndarray], fi_gb: List[np.ndarray]) -> Dict[str, Any]:
        """Summarizes stability, win rates, and assigns status per crop."""
        models = df_f["model"].unique()
        model_summaries = {}

        for m in models:
            m_rows = df_f[df_f["model"] == m]
            fold_count = len(m_rows)
            maes = m_rows["mae"].values
            rmses = m_rows["rmse"].values
            r2s = m_rows["r2"].values
            mapes = m_rows["mape"].values
            smapes = m_rows["smape"].values

            mean_mae = float(np.mean(maes))
            median_mae = float(np.median(maes))
            std_mae = float(np.std(maes))
            min_mae = float(np.min(maes))
            max_mae = float(np.max(maes))

            mean_rmse = float(np.mean(rmses))
            std_rmse = float(np.std(rmses))
            mean_r2 = float(np.mean(r2s))
            std_r2 = float(np.std(r2s))
            mean_mape = float(np.mean(mapes))
            std_mape = float(np.std(mapes))

            # Baseline comparisons
            win_count = int(m_rows["win_vs_baseline"].sum())
            loss_count = fold_count - win_count
            win_rate = float(win_count / fold_count * 100.0) if fold_count > 0 else 0.0

            imp_pcts = m_rows["mae_improvement_pct"].values
            mean_imp = float(np.mean(imp_pcts))
            median_imp = float(np.median(imp_pcts))
            std_imp = float(np.std(imp_pcts))

            mean_base_mae = float(m_rows["best_baseline_mae"].mean())

            model_summaries[m] = {
                "fold_count": fold_count,
                "mean_mae": round(mean_mae, 2),
                "median_mae": round(median_mae, 2),
                "std_mae": round(std_mae, 2),
                "min_mae": round(min_mae, 2),
                "max_mae": round(max_mae, 2),
                "mean_rmse": round(mean_rmse, 2),
                "std_rmse": round(std_rmse, 2),
                "mean_r2": round(mean_r2, 4),
                "std_r2": round(std_r2, 4),
                "mean_mape": round(mean_mape, 2),
                "std_mape": round(std_mape, 2),
                "baseline_mean_mae": round(mean_base_mae, 2),
                "win_count": win_count,
                "loss_count": loss_count,
                "win_rate": round(win_rate, 2),
                "mean_mae_improvement": round(mean_imp, 2),
                "median_mae_improvement": round(median_imp, 2),
                "std_mae_improvement": round(std_imp, 2),
            }

        # Determine best overall candidate
        rf_summ = model_summaries["RandomForestRegressor"]
        gb_summ = model_summaries["GradientBoostingRegressor"]
        base_summ = model_summaries["Historical District Mean"]

        best_ml_name = "RandomForestRegressor" if rf_summ["mean_mae"] <= gb_summ["mean_mae"] else "GradientBoostingRegressor"
        best_ml_summ = model_summaries[best_ml_name]

        # Determine Day 20 Status
        # ROBUST_ACCEPTED: win_rate >= 75% AND mean_mae_improvement > 0
        # SPLIT_SENSITIVE: win_rate between 25% and 75% or Day 19 was accepted but fails on some folds
        # BASELINE_PREFERRED: win_rate < 25%
        if best_ml_summ["win_rate"] >= 75.0 and best_ml_summ["mean_mae_improvement"] > 0:
            robustness_status = "ROBUST_ACCEPTED"
            recommendation = (f"ML model ({best_ml_name}) demonstrates consistent temporal superiority "
                              f"across {best_ml_summ['win_count']}/{best_ml_summ['fold_count']} walk-forward folds "
                              f"with mean MAE improvement of +{best_ml_summ['mean_mae_improvement']}%.")
        elif best_ml_summ["win_rate"] >= 25.0:
            robustness_status = "SPLIT_SENSITIVE"
            recommendation = (f"ML model ({best_ml_name}) is split-sensitive; outperformed baseline in "
                              f"{best_ml_summ['win_count']}/{best_ml_summ['fold_count']} folds but lacks uniform multi-origin stability.")
        else:
            robustness_status = "BASELINE_PREFERRED"
            recommendation = (f"Statistical baseline ({base_summ['baseline_mean_mae']} kg/ha MAE) is superior to ML "
                              f"across {best_ml_summ['loss_count']}/{best_ml_summ['fold_count']} temporal forecasting folds.")

        # Robustness Score (0 - 100)
        # 40 pts: Win Rate
        # 25 pts: Stability (Low Std/Mean ratio)
        # 20 pts: Mean Improvement %
        # 15 pts: Base readiness score
        win_score = (best_ml_summ["win_rate"] / 100.0) * 40.0
        cv_ratio = best_ml_summ["std_mae"] / max(best_ml_summ["mean_mae"], 1.0)
        stab_score = max(0.0, (1.0 - min(cv_ratio, 1.0))) * 25.0
        imp_score = min(max(best_ml_summ["mean_mae_improvement"] * 2.0, 0.0), 20.0)
        readiness_base = 15.0  # Already passed MODEL_READY checks

        robustness_score = round(win_score + stab_score + imp_score + readiness_base, 1)

        # Feature Importance Stability
        feat_names = CropFeaturePipeline.FEATURE_NAMES
        mean_fi_rf = np.mean(fi_rf, axis=0) if len(fi_rf) > 0 else np.zeros(len(feat_names))
        std_fi_rf = np.std(fi_rf, axis=0) if len(fi_rf) > 0 else np.zeros(len(feat_names))

        feature_stability = [
            {
                "feature": feat_names[i],
                "mean_importance": round(float(mean_fi_rf[i]), 4),
                "std_importance": round(float(std_fi_rf[i]), 4),
            }
            for i in range(len(feat_names))
        ]

        return {
            "crop": crop,
            "best_model": best_ml_name if robustness_status == "ROBUST_ACCEPTED" else "Historical District Mean",
            "evaluated_ml_model": best_ml_name,
            "robustness_status": robustness_status,
            "robustness_score": robustness_score,
            "recommendation": recommendation,
            "models": model_summaries,
            "feature_stability": feature_stability,
        }

    def execute_full_walk_forward(self) -> Dict[str, Any]:
        """Runs walk-forward evaluation across all 14 MODEL_READY crops and serializes outputs."""
        all_fold_rows = []
        crop_summaries = []

        print(f"[Temporal Walk-Forward] Commencing evaluation across {len(self.model_ready_crops)} crops...")

        for crop in self.model_ready_crops:
            print(f"  Evaluating {crop}...")
            folds_data, summary = self.run_walk_forward_for_crop(crop)
            all_fold_rows.extend(folds_data)
            crop_summaries.append(summary)

        # 1. Save Fold Results CSV
        df_folds = pd.DataFrame(all_fold_rows)
        fold_csv_path = "Datasets/metadata/multicrop_fold_results.csv"
        df_folds.to_csv(fold_csv_path, index=False)
        print(f"[Saved] {fold_csv_path} ({len(df_folds)} rows)")

        # 2. Save Temporal Robustness CSV
        robustness_rows = []
        for s in crop_summaries:
            m_name = s["evaluated_ml_model"]
            m_summ = s["models"][m_name]
            robustness_rows.append({
                "crop": s["crop"],
                "model": m_name,
                "fold_count": m_summ["fold_count"],
                "mean_mae": m_summ["mean_mae"],
                "median_mae": m_summ["median_mae"],
                "std_mae": m_summ["std_mae"],
                "mean_rmse": m_summ["mean_rmse"],
                "std_rmse": m_summ["std_rmse"],
                "mean_r2": m_summ["mean_r2"],
                "std_r2": m_summ["std_r2"],
                "baseline_mae": m_summ["baseline_mean_mae"],
                "mean_mae_improvement": m_summ["mean_mae_improvement"],
                "median_mae_improvement": m_summ["median_mae_improvement"],
                "win_rate": m_summ["win_rate"],
                "status": s["robustness_status"],
            })

        df_rob = pd.DataFrame(robustness_rows)
        rob_csv_path = "Datasets/metadata/multicrop_temporal_robustness.csv"
        df_rob.to_csv(rob_csv_path, index=False)
        print(f"[Saved] {rob_csv_path} ({len(df_rob)} rows)")

        # 3. Save Model Robustness Scores CSV
        score_rows = [
            {
                "crop": s["crop"],
                "robustness_score": s["robustness_score"],
                "robustness_status": s["robustness_status"],
                "best_model": s["best_model"],
                "win_rate": s["models"][s["evaluated_ml_model"]]["win_rate"],
                "mean_mae_improvement": s["models"][s["evaluated_ml_model"]]["mean_mae_improvement"],
                "std_mae": s["models"][s["evaluated_ml_model"]]["std_mae"],
                "recommendation": s["recommendation"],
            }
            for s in crop_summaries
        ]
        df_scores = pd.DataFrame(score_rows)
        score_csv_path = "Datasets/metadata/model_robustness_scores.csv"
        df_scores.to_csv(score_csv_path, index=False)
        print(f"[Saved] {score_csv_path} ({len(df_scores)} rows)")

        # 4. Update Model Registry without destroying Day 19 lineage
        registry_path = "Models/multicrop/model_registry.json"
        if os.path.exists(registry_path):
            with open(registry_path, "r", encoding="utf-8") as f:
                reg_data = json.load(f)
        else:
            reg_data = {"version": "2.0.0", "models": {}}

        reg_data["validation_version"] = "2.0.0 (Walk-Forward Robustness)"
        for s in crop_summaries:
            c = s["crop"]
            if c in reg_data.get("models", {}):
                reg_data["models"][c]["day20_robustness"] = {
                    "robustness_status": s["robustness_status"],
                    "robustness_score": s["robustness_score"],
                    "walk_forward_folds": s["models"][s["evaluated_ml_model"]]["fold_count"],
                    "baseline_win_rate": s["models"][s["evaluated_ml_model"]]["win_rate"],
                    "mean_mae": s["models"][s["evaluated_ml_model"]]["mean_mae"],
                    "std_mae": s["models"][s["evaluated_ml_model"]]["std_mae"],
                    "mean_mae_improvement": s["models"][s["evaluated_ml_model"]]["mean_mae_improvement"],
                }

        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(reg_data, f, indent=2)
        print(f"[Updated] {registry_path} with Day 20 validation results.")

        return {
            "total_crops": len(crop_summaries),
            "total_folds_evaluated": len(df_folds),
            "summaries": crop_summaries,
        }


if __name__ == "__main__":
    engine = TemporalWalkForwardEngine()
    results = engine.execute_full_walk_forward()
    print("\n" + "="*80)
    print("DAY 20 WALK-FORWARD VALIDATION SUMMARY")
    print("="*80)
    for s in results["summaries"]:
        m_name = s["evaluated_ml_model"]
        m_s = s["models"][m_name]
        print(f"{s['crop']:22s} | Status: {s['robustness_status']:18s} | Score: {s['robustness_score']:4.1f} | "
              f"Win Rate: {m_s['win_rate']:5.1f}% ({m_s['win_count']}/{m_s['fold_count']}) | Mean MAE: {m_s['mean_mae']:6.2f} (Base: {m_s['baseline_mean_mae']:6.2f})")
