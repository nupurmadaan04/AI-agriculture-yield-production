"""
Multi-Crop Error Diagnosis Module (Day 21)
===========================================
Executes multi-origin walk-forward error diagnosis across all 14 MODEL_READY crops.
Performs comprehensive multi-level error decomposition:
1. Crop & Fold Level Stability (MAE, RMSE, R2, MAPE, SMAPE, MAE_CV)
2. Win Consistency (Wins, Losses, Win Rate, Mean/Median Gain, Worst Degradation)
3. Error Distribution & Quantiles (P25, P50, P75, P90, P95, Normalized Errors)
4. Yield Regime Decomposition (Low <= Q25, Normal Q25-Q75, High >= Q75)
5. Temporal Error Analysis (Year-by-Year ML vs Baseline, High-Error Regimes)
6. District Error Analysis (N >= 3 observations, Best ML, Worst ML, High Error Districts)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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


class MultiCropErrorDiagnosisEngine:
    """Detailed error diagnosis and regime decomposition engine across walk-forward folds."""

    def __init__(
        self,
        data_path: str = "Datasets/processed/agricultural_panel.csv",
        readiness_path: str = "Datasets/metadata/crop_model_readiness.csv",
        model_results_path: str = "Datasets/metadata/multicrop_model_results.csv",
    ):
        self.data_path = data_path
        self.readiness_path = readiness_path
        self.df = pd.read_csv(data_path)
        self.readiness_df = pd.read_csv(readiness_path)
        self.model_ready_crops = self.readiness_df[
            self.readiness_df["readiness_status"] == "MODEL_READY"
        ]["crop"].tolist()

        # Determine chosen ML algorithm per crop from Day 19/20
        self.crop_best_algos = {}
        if os.path.exists(model_results_path):
            mr_df = pd.read_csv(model_results_path)
            for crop in self.model_ready_crops:
                c_rows = mr_df[mr_df["crop"] == crop]
                if not c_rows.empty and "ml_winner" in c_rows.columns:
                    best_algo = c_rows.iloc[0]["ml_winner"]
                    self.crop_best_algos[crop] = "RandomForestRegressor" if "RandomForest" in str(best_algo) else "GradientBoostingRegressor"
                else:
                    self.crop_best_algos[crop] = "RandomForestRegressor"
        else:
            for crop in self.model_ready_crops:
                self.crop_best_algos[crop] = "RandomForestRegressor"

    def get_valid_folds(self, crop_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Constructs expanding-window walk-forward folds."""
        years = sorted(crop_df["year"].unique())
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
        """Calculates statistical baseline predictions strictly from training data."""
        train_clean = train_df.dropna(subset=["yield_kg_ha"])
        overall_mean = float(train_clean["yield_kg_ha"].mean()) if not train_clean.empty else 0.0
        dist_means = {k: v for k, v in train_clean.groupby("district")["yield_kg_ha"].mean().to_dict().items() if pd.notna(v)}

        pred_mean = np.nan_to_num(test_df["district"].map(dist_means).fillna(overall_mean).values, nan=overall_mean)
        pred_persistence = np.nan_to_num(test_df["yield_lag_1"].fillna(overall_mean).values, nan=overall_mean)

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
                        p = dist_means.get(d, overall_mean)
                except Exception:
                    p = dist_means.get(d, overall_mean)
                pred_trend_list.append(max(float(p), 0.0))
            else:
                pred_trend_list.append(dist_means.get(d, overall_mean))
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
            baselines[name] = {
                "predictions": preds,
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "r2": round(r2, 4),
                "mape": round(mape, 2),
                "smape": round(smape, 2),
            }
        return baselines

    def run_crop_diagnosis(self, crop: str) -> Dict[str, Any]:
        """Runs full error diagnosis across all walk-forward folds for a crop."""
        crop_df = self.df[self.df["crop"] == crop].copy().sort_values(["district", "year"])
        folds = self.get_valid_folds(crop_df)
        algo_name = self.crop_best_algos.get(crop, "RandomForestRegressor")

        fold_records = []
        obs_records = []

        for fold in folds:
            fold_id = fold["fold_id"]
            train_years = fold["train_years"]
            test_year = fold["test_year"]

            train_raw = crop_df[crop_df["year"].isin(train_years)].copy()
            panel_up_to_test = crop_df[crop_df["year"] <= test_year].copy()

            pipeline = CropFeaturePipeline()
            pipeline.fit(train_raw)

            panel_trans = pipeline.transform(panel_up_to_test)
            train_proc = panel_trans[panel_trans["year"].isin(train_years)].dropna(subset=["yield_kg_ha"])
            test_proc = panel_trans[panel_trans["year"] == test_year].dropna(subset=["yield_kg_ha"])
            train_raw_clean = train_raw.dropna(subset=["yield_kg_ha"])

            X_train = np.nan_to_num(train_proc[CropFeaturePipeline.FEATURE_NAMES].values, nan=0.0)
            y_train = train_proc["yield_kg_ha"].values
            X_test = np.nan_to_num(test_proc[CropFeaturePipeline.FEATURE_NAMES].values, nan=0.0)
            y_test = test_proc["yield_kg_ha"].values

            # Baselines
            baselines = self.evaluate_baselines_on_fold(train_raw_clean, test_proc)
            best_baseline_name = min(baselines, key=lambda k: baselines[k]["mae"])
            best_base = baselines[best_baseline_name]

            # Fit ML model
            if algo_name == "GradientBoostingRegressor":
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=4,
                    min_samples_leaf=4,
                    random_state=42,
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_leaf=4,
                    random_state=42,
                    n_jobs=-1,
                )

            model.fit(X_train, y_train)
            pred_ml = model.predict(X_test)
            mae_ml = float(mean_absolute_error(y_test, pred_ml))
            rmse_ml = float(np.sqrt(mean_squared_error(y_test, pred_ml)))
            r2_ml = float(r2_score(y_test, pred_ml)) if np.var(y_test) > 1e-6 else 0.0
            mape_ml = compute_mape(y_test, pred_ml)
            smape_ml = compute_smape(y_test, pred_ml)

            base_mae = best_base["mae"]
            base_rmse = best_base["rmse"]
            base_r2 = best_base["r2"]
            base_mape = best_base["mape"]
            base_smape = best_base["smape"]

            diff_mae = round(base_mae - mae_ml, 2)
            pct_improvement = round(((base_mae - mae_ml) / base_mae) * 100.0, 2) if base_mae > 0 else 0.0
            is_win = mae_ml < base_mae

            fold_records.append({
                "crop": crop,
                "fold_id": fold_id,
                "test_year": test_year,
                "train_samples": len(train_proc),
                "test_samples": len(test_proc),
                "ml_model": algo_name,
                "ml_mae": round(mae_ml, 2),
                "ml_rmse": round(rmse_ml, 2),
                "ml_r2": round(r2_ml, 4),
                "ml_mape": round(mape_ml, 2),
                "ml_smape": round(smape_ml, 2),
                "baseline_model": best_baseline_name,
                "baseline_mae": round(base_mae, 2),
                "baseline_rmse": round(base_rmse, 2),
                "baseline_r2": round(base_r2, 4),
                "baseline_mape": round(base_mape, 2),
                "baseline_smape": round(base_smape, 2),
                "mae_diff": diff_mae,
                "mae_improvement_pct": pct_improvement,
                "win": is_win,
            })

            # Record observation-level details
            base_preds = best_base["predictions"]
            for i, (_, row) in enumerate(test_proc.iterrows()):
                y_t = float(row["yield_kg_ha"])
                y_m = float(pred_ml[i])
                y_b = float(base_preds[i])
                err_m = abs(y_t - y_m)
                err_b = abs(y_t - y_b)
                obs_records.append({
                    "crop": crop,
                    "year": test_year,
                    "district": row["district"],
                    "state": row["state"],
                    "actual_yield": y_t,
                    "ml_pred": y_m,
                    "baseline_pred": y_b,
                    "ml_abs_error": err_m,
                    "baseline_abs_error": err_b,
                    "ml_won_obs": err_m < err_b,
                })

        obs_df = pd.DataFrame(obs_records)
        fold_df = pd.DataFrame(fold_records)

        # 1. Summary across folds
        ml_maes = fold_df["ml_mae"].values
        base_maes = fold_df["baseline_mae"].values
        improvements = fold_df["mae_improvement_pct"].values

        total_folds = len(fold_records)
        win_count = int(np.sum(fold_df["win"]))
        loss_count = total_folds - win_count
        win_rate = round((win_count / total_folds) * 100.0, 2) if total_folds > 0 else 0.0

        ml_mae_mean = round(float(np.mean(ml_maes)), 2)
        ml_mae_median = round(float(np.median(ml_maes)), 2)
        ml_mae_std = round(float(np.std(ml_maes, ddof=1)), 2) if len(ml_maes) > 1 else 0.0
        ml_mae_min = round(float(np.min(ml_maes)), 2)
        ml_mae_max = round(float(np.max(ml_maes)), 2)
        ml_mae_cv = round(float(ml_mae_std / ml_mae_mean), 4) if ml_mae_mean > 0 else 0.0

        base_mae_mean = round(float(np.mean(base_maes)), 2)
        base_mae_median = round(float(np.median(base_maes)), 2)
        base_mae_std = round(float(np.std(base_maes, ddof=1)), 2) if len(base_maes) > 1 else 0.0
        base_mae_min = round(float(np.min(base_maes)), 2)
        base_mae_max = round(float(np.max(base_maes)), 2)
        base_mae_cv = round(float(base_mae_std / base_mae_mean), 4) if base_mae_mean > 0 else 0.0

        mean_gain = round(float(np.mean(improvements)), 2)
        median_gain = round(float(np.median(improvements)), 2)
        worst_degradation = round(float(np.min(improvements)), 2)
        best_gain = round(float(np.max(improvements)), 2)

        # 2. Error Quantiles & Distribution
        ml_errors = obs_df["ml_abs_error"].values
        base_errors = obs_df["baseline_abs_error"].values
        hist_median_yield = float(crop_df["yield_kg_ha"].median()) if not crop_df.empty else 1.0

        p25_ml = round(float(np.percentile(ml_errors, 25)), 2)
        p50_ml = round(float(np.percentile(ml_errors, 50)), 2)
        p75_ml = round(float(np.percentile(ml_errors, 75)), 2)
        p90_ml = round(float(np.percentile(ml_errors, 90)), 2)
        p95_ml = round(float(np.percentile(ml_errors, 95)), 2)

        p25_base = round(float(np.percentile(base_errors, 25)), 2)
        p50_base = round(float(np.percentile(base_errors, 50)), 2)
        p75_base = round(float(np.percentile(base_errors, 75)), 2)
        p90_base = round(float(np.percentile(base_errors, 90)), 2)
        p95_base = round(float(np.percentile(base_errors, 95)), 2)

        pct_under_100_ml = round(float(np.mean(ml_errors < 100) * 100.0), 2)
        pct_under_250_ml = round(float(np.mean(ml_errors < 250) * 100.0), 2)
        pct_under_500_ml = round(float(np.mean(ml_errors < 500) * 100.0), 2)
        pct_over_1000_ml = round(float(np.mean(ml_errors > 1000) * 100.0), 2)

        norm_err_p50 = round(float(p50_ml / hist_median_yield), 4) if hist_median_yield > 0 else 0.0
        norm_err_p90 = round(float(p90_ml / hist_median_yield), 4) if hist_median_yield > 0 else 0.0

        crop_summary = {
            "crop": crop,
            "selected_ml_model": algo_name,
            "total_folds": total_folds,
            "total_test_observations": len(obs_df),
            "historical_median_yield": round(hist_median_yield, 2),
            "ml_wins": win_count,
            "ml_losses": loss_count,
            "win_rate": win_rate,
            "mean_mae_improvement_pct": mean_gain,
            "median_mae_improvement_pct": median_gain,
            "worst_fold_degradation_pct": worst_degradation,
            "best_fold_improvement_pct": best_gain,
            "ml_mae_mean": ml_mae_mean,
            "ml_mae_median": ml_mae_median,
            "ml_mae_std": ml_mae_std,
            "ml_mae_min": ml_mae_min,
            "ml_mae_max": ml_mae_max,
            "ml_mae_cv": ml_mae_cv,
            "base_mae_mean": base_mae_mean,
            "base_mae_median": base_mae_median,
            "base_mae_std": base_mae_std,
            "base_mae_min": base_mae_min,
            "base_mae_max": base_mae_max,
            "base_mae_cv": base_mae_cv,
            "p25_error_ml": p25_ml,
            "p50_error_ml": p50_ml,
            "p75_error_ml": p75_ml,
            "p90_error_ml": p90_ml,
            "p95_error_ml": p95_ml,
            "p25_error_base": p25_base,
            "p50_error_base": p50_base,
            "p75_error_base": p75_base,
            "p90_error_base": p90_base,
            "p95_error_base": p95_base,
            "pct_errors_lt_100": pct_under_100_ml,
            "pct_errors_lt_250": pct_under_250_ml,
            "pct_errors_lt_500": pct_under_500_ml,
            "pct_errors_gt_1000": pct_over_1000_ml,
            "normalized_error_p50": norm_err_p50,
            "normalized_error_p90": norm_err_p90,
        }

        # 3. Yield Regimes (Low <= Q25, Normal Q25-Q75, High >= Q75)
        q25 = float(obs_df["actual_yield"].quantile(0.25))
        q75 = float(obs_df["actual_yield"].quantile(0.75))

        regime_records = []
        for regime_name, condition in [
            ("Low Yield", obs_df["actual_yield"] <= q25),
            ("Normal Yield", (obs_df["actual_yield"] > q25) & (obs_df["actual_yield"] < q75)),
            ("High Yield", obs_df["actual_yield"] >= q75),
        ]:
            subset = obs_df[condition]
            n_sub = len(subset)
            if n_sub > 0:
                sub_ml_mae = round(float(subset["ml_abs_error"].mean()), 2)
                sub_base_mae = round(float(subset["baseline_abs_error"].mean()), 2)
                sub_gain = round(((sub_base_mae - sub_ml_mae) / sub_base_mae) * 100.0, 2) if sub_base_mae > 0 else 0.0
                sub_win_rate = round(float(subset["ml_won_obs"].mean() * 100.0), 2)
                y_min = round(float(subset["actual_yield"].min()), 2)
                y_max = round(float(subset["actual_yield"].max()), 2)
            else:
                sub_ml_mae = 0.0
                sub_base_mae = 0.0
                sub_gain = 0.0
                sub_win_rate = 0.0
                y_min = 0.0
                y_max = 0.0

            regime_records.append({
                "crop": crop,
                "regime": regime_name,
                "n_observations": n_sub,
                "yield_min_kg_ha": y_min,
                "yield_max_kg_ha": y_max,
                "ml_mae": sub_ml_mae,
                "baseline_mae": sub_base_mae,
                "ml_improvement_pct": sub_gain,
                "obs_win_rate": sub_win_rate,
                "regime_status": "ML_ADVANTAGE" if sub_gain > 0 else "BASELINE_ADVANTAGE",
            })

        # 4. Temporal Year Breakdown
        year_records = []
        for _, row in fold_df.iterrows():
            yr = int(row["test_year"])
            ml_m = row["ml_mae"]
            base_m = row["baseline_mae"]
            gain = row["mae_improvement_pct"]

            if yr == 2016 and ml_m > ml_mae_mean * 1.2:
                regime_lbl = "2016 high-error regime"
            elif gain >= 5.0:
                regime_lbl = "ml_advantage"
            elif gain <= -5.0:
                regime_lbl = "baseline_advantage"
            else:
                regime_lbl = "stable_year"

            year_records.append({
                "crop": crop,
                "year": yr,
                "fold_id": int(row["fold_id"]),
                "ml_model": row["ml_model"],
                "ml_mae": ml_m,
                "ml_rmse": row["ml_rmse"],
                "baseline_model": row["baseline_model"],
                "baseline_mae": base_m,
                "baseline_rmse": row["baseline_rmse"],
                "ml_improvement_pct": gain,
                "ml_win": bool(row["win"]),
                "temporal_regime": regime_lbl,
            })

        # 5. District Error Analysis (Enforcing minimum observation threshold N >= 3)
        district_records = []
        dist_groups = obs_df.groupby(["state", "district"])
        crop_mean_err = np.mean(ml_errors)

        for (state, district), d_df in dist_groups:
            n_obs = len(d_df)
            if n_obs < 3:
                continue
            d_ml_mae = round(float(d_df["ml_abs_error"].mean()), 2)
            d_base_mae = round(float(d_df["baseline_abs_error"].mean()), 2)
            d_gain = round(((d_base_mae - d_ml_mae) / d_base_mae) * 100.0, 2) if d_base_mae > 0 else 0.0
            d_win_rate = round(float(d_df["ml_won_obs"].mean() * 100.0), 2)

            is_best = d_win_rate >= 75.0 and d_gain > 5.0
            is_worst = d_win_rate <= 25.0 or d_gain < -10.0
            is_high_error = d_ml_mae > crop_mean_err * 1.5

            district_records.append({
                "crop": crop,
                "state": state,
                "district": district,
                "observations_count": n_obs,
                "ml_mae": d_ml_mae,
                "baseline_mae": d_base_mae,
                "ml_improvement_pct": d_gain,
                "ml_win_rate": d_win_rate,
                "is_best_ml_district": is_best,
                "is_worst_ml_district": is_worst,
                "is_high_error_district": is_high_error,
            })

        return {
            "crop_summary": crop_summary,
            "regimes": regime_records,
            "years": year_records,
            "districts": district_records,
            "folds": fold_records,
        }

    def execute_all(self, output_dir: str = "Datasets/metadata") -> Dict[str, Any]:
        """Runs error diagnosis for all 14 crops and writes metadata CSVs."""
        os.makedirs(output_dir, exist_ok=True)
        all_summaries = []
        all_regimes = []
        all_years = []
        all_districts = []

        for crop in self.model_ready_crops:
            res = self.run_crop_diagnosis(crop)
            all_summaries.append(res["crop_summary"])
            all_regimes.extend(res["regimes"])
            all_years.extend(res["years"])
            all_districts.extend(res["districts"])

        summary_df = pd.DataFrame(all_summaries)
        regimes_df = pd.DataFrame(all_regimes)
        years_df = pd.DataFrame(all_years)
        districts_df = pd.DataFrame(all_districts)

        summary_df.to_csv(os.path.join(output_dir, "multicrop_error_diagnosis.csv"), index=False)
        regimes_df.to_csv(os.path.join(output_dir, "multicrop_error_regimes.csv"), index=False)
        years_df.to_csv(os.path.join(output_dir, "multicrop_year_error_analysis.csv"), index=False)
        districts_df.to_csv(os.path.join(output_dir, "multicrop_district_error_analysis.csv"), index=False)

        return {
            "crops_diagnosed": len(all_summaries),
            "total_regimes": len(all_regimes),
            "total_year_records": len(all_years),
            "total_district_records": len(all_districts),
        }


if __name__ == "__main__":
    engine = MultiCropErrorDiagnosisEngine()
    stats = engine.execute_all()
    print(f"Error diagnosis completed: {stats}")
