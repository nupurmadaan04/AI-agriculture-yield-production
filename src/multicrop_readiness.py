"""
Multi-Crop Modeling Readiness & Crop-Specific Baseline Engine
============================================================
Evaluates data volume, temporal continuity, geographic representation,
target distributions, chronological split feasibility, and baseline
forecasting performance across all 29 unified agricultural crops.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "Datasets", "processed", "agricultural_panel.csv")
CONFIG_PATH = os.path.join(BASE_DIR, "Datasets", "metadata", "model_readiness_config.json")
METADATA_DIR = os.path.join(BASE_DIR, "Datasets", "metadata")
PROCESSED_DIR = os.path.join(BASE_DIR, "Datasets", "processed")


def load_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Loads configuration criteria for model readiness evaluation."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "criteria": {
            "min_total_records": 200,
            "min_years": 5,
            "min_districts": 5,
            "min_yield_completeness": 0.90,
            "min_temporal_continuity": 0.60,
            "min_district_median_years": 4,
            "min_train_years": 4,
            "min_test_years": 2,
            "train_period": [2010, 2015],
            "test_period": [2016, 2017],
            "min_active_districts_in_test": 10,
            "max_zero_yield_rate_for_ready": 0.50,
        }
    }


class MultiCropReadinessEngine:
    """Profiles crop panel data, computes readiness metrics, and evaluates baseline models."""

    def __init__(self, dataset_path: str = DATASET_PATH, config_path: str = CONFIG_PATH):
        self.dataset_path = dataset_path
        self.config = load_config(config_path)["criteria"]
        self.df = None
        self._load_data()

    def _load_data(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Unified agricultural panel not found at {self.dataset_path}")
        self.df = pd.read_csv(self.dataset_path)
        logger.info(f"Loaded {len(self.df)} records across {self.df['crop'].nunique()} crops.")

    def profile_all_crops(self) -> pd.DataFrame:
        """Profiles observations, temporal span, geographic reach, and target distribution for each crop."""
        profiles = []
        total_districts_all = self.df["district"].nunique()
        total_states_all = self.df["state"].nunique()

        for crop_name, group in self.df.groupby("crop"):
            rec_count = len(group)
            u_states = group["state"].nunique()
            u_districts = group["district"].nunique()
            u_years = group["year"].nunique()
            min_yr = int(group["year"].min())
            max_yr = int(group["year"].max())
            yr_span = max_yr - min_yr + 1

            # Target Statistics
            yields = group["yield_kg_ha"].dropna()
            y_count = len(yields)
            y_miss_rate = 1.0 - (y_count / rec_count) if rec_count > 0 else 1.0
            zero_yield_count = (yields == 0.0).sum()
            zero_yield_rate = float(zero_yield_count / y_count) if y_count > 0 else 0.0

            # Yield Distribution (Positive Non-Zero Yields)
            pos_yields = yields[yields > 0]
            if len(pos_yields) > 0:
                y_mean = float(pos_yields.mean())
                y_median = float(pos_yields.median())
                y_std = float(pos_yields.std()) if len(pos_yields) > 1 else 0.0
                y_min = float(pos_yields.min())
                y_max = float(pos_yields.max())
                y_q1 = float(pos_yields.quantile(0.25))
                y_q3 = float(pos_yields.quantile(0.75))
                y_iqr = float(y_q3 - y_q1)
                y_cv = float(y_std / y_mean) if y_mean > 0 else 0.0
            else:
                y_mean = y_median = y_std = y_min = y_max = y_q1 = y_q3 = y_iqr = y_cv = 0.0

            # Area & Production
            areas = group["area_ha"].dropna()
            pos_areas = areas[areas > 0]
            area_mean = float(pos_areas.mean()) if len(pos_areas) > 0 else 0.0
            area_cv = float(pos_areas.std() / area_mean) if len(pos_areas) > 1 and area_mean > 0 else 0.0

            prods = group["production_tonnes"].dropna()
            pos_prods = prods[prods > 0]
            prod_mean = float(pos_prods.mean()) if len(pos_prods) > 0 else 0.0
            prod_cv = float(pos_prods.std() / prod_mean) if len(pos_prods) > 1 and prod_mean > 0 else 0.0

            # District coverage and continuity
            dist_counts = group.groupby("district")["year"].nunique()
            med_yrs_per_dist = float(dist_counts.median()) if len(dist_counts) > 0 else 0.0
            min_yrs_per_dist = int(dist_counts.min()) if len(dist_counts) > 0 else 0
            max_yrs_per_dist = int(dist_counts.max()) if len(dist_counts) > 0 else 0

            # Active Districts (districts where crop is actively produced/yield > 0 in at least 1 year)
            active_districts = group[group["yield_kg_ha"] > 0]["district"].nunique()

            profiles.append({
                "crop": crop_name,
                "total_records": rec_count,
                "unique_states": u_states,
                "unique_districts": u_districts,
                "active_districts": active_districts,
                "unique_years": u_years,
                "min_year": min_yr,
                "max_year": max_yr,
                "year_span": yr_span,
                "state_coverage_ratio": round(u_states / total_states_all, 4),
                "district_coverage_ratio": round(u_districts / total_districts_all, 4),
                "active_district_ratio": round(active_districts / total_districts_all, 4),
                "median_years_per_district": med_yrs_per_dist,
                "min_years_per_district": min_yrs_per_dist,
                "max_years_per_district": max_yrs_per_dist,
                "yield_count": y_count,
                "yield_missing_rate": round(y_miss_rate, 4),
                "zero_yield_rate": round(zero_yield_rate, 4),
                "yield_mean": round(y_mean, 2),
                "yield_median": round(y_median, 2),
                "yield_std": round(y_std, 2),
                "yield_min": round(y_min, 2),
                "yield_max": round(y_max, 2),
                "yield_q1": round(y_q1, 2),
                "yield_q3": round(y_q3, 2),
                "yield_iqr": round(y_iqr, 2),
                "yield_cv": round(y_cv, 4),
                "area_mean_ha": round(area_mean, 2),
                "area_cv": round(area_cv, 4),
                "production_mean_tonnes": round(prod_mean, 2),
                "production_cv": round(prod_cv, 4),
            })

        return pd.DataFrame(profiles).sort_values(by="total_records", ascending=False)

    def evaluate_temporal_continuity(self) -> pd.DataFrame:
        """Calculates temporal continuity ratios and district series continuity."""
        results = []
        for crop_name, group in self.df.groupby("crop"):
            dist_stats = []
            for dist, d_group in group.groupby("district"):
                years = sorted(d_group["year"].unique())
                if len(years) <= 1:
                    dist_stats.append({
                        "district": dist,
                        "years_count": len(years),
                        "continuity_ratio": 1.0 if len(years) == 1 else 0.0,
                        "status": "FRAGMENTED" if len(years) <= 1 else "CONTINUOUS"
                    })
                    continue

                possible_links = years[-1] - years[0]
                actual_links = sum(1 for i in range(len(years) - 1) if years[i + 1] - years[i] == 1)
                cont_ratio = actual_links / possible_links if possible_links > 0 else 1.0

                if actual_links == (len(years) - 1) and len(years) >= 6:
                    status = "CONTINUOUS"
                elif cont_ratio >= 0.70:
                    status = "PARTIAL"
                else:
                    status = "FRAGMENTED"

                dist_stats.append({
                    "district": dist,
                    "years_count": len(years),
                    "continuity_ratio": cont_ratio,
                    "status": status
                })

            dist_df = pd.DataFrame(dist_stats)
            d_count = len(dist_df)
            cont_count = (dist_df["status"] == "CONTINUOUS").sum()
            part_count = (dist_df["status"] == "PARTIAL").sum()
            frag_count = (dist_df["status"] == "FRAGMENTED").sum()
            med_cont = float(dist_df["continuity_ratio"].median()) if d_count > 0 else 0.0
            min_cont = float(dist_df["continuity_ratio"].min()) if d_count > 0 else 0.0
            max_cont = float(dist_df["continuity_ratio"].max()) if d_count > 0 else 0.0

            results.append({
                "crop": crop_name,
                "district_count": d_count,
                "continuous_district_count": int(cont_count),
                "partial_district_count": int(part_count),
                "fragmented_district_count": int(frag_count),
                "median_continuity": round(med_cont, 4),
                "min_continuity": round(min_cont, 4),
                "max_continuity": round(max_cont, 4),
            })

        return pd.DataFrame(results).sort_values(by="continuous_district_count", ascending=False)

    def classify_readiness(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Classifies each crop as MODEL_READY, ANALYTICS_READY, or INSUFFICIENT_DATA."""
        profiles_df = self.profile_all_crops()
        temporal_df = self.evaluate_temporal_continuity().set_index("crop")

        readiness_records = []
        target_profiles = []

        for _, row in profiles_df.iterrows():
            crop = row["crop"]
            t_info = temporal_df.loc[crop] if crop in temporal_df.index else {}

            # Evaluate Criteria
            blocking_reasons = []
            
            # Data Volume
            vol_pass = row["total_records"] >= self.config["min_total_records"]
            if not vol_pass:
                blocking_reasons.append(f"Insufficient total records ({row['total_records']} < {self.config['min_total_records']})")

            # Temporal Breadth
            yr_pass = row["unique_years"] >= self.config["min_years"]
            if not yr_pass:
                blocking_reasons.append(f"Insufficient historical years ({row['unique_years']} < {self.config['min_years']})")

            # Temporal Continuity
            med_cont = t_info.get("median_continuity", 0.0)
            cont_pass = med_cont >= self.config["min_temporal_continuity"]
            if not cont_pass:
                blocking_reasons.append(f"Low temporal continuity ({med_cont:.2f} < {self.config['min_temporal_continuity']})")

            # Target Completeness
            comp_rate = 1.0 - row["yield_missing_rate"]
            comp_pass = comp_rate >= self.config["min_yield_completeness"]
            if not comp_pass:
                blocking_reasons.append(f"Low yield completeness ({comp_rate:.2f} < {self.config['min_yield_completeness']})")

            # Active Cultivation & Zero Yield Inflation Check
            zero_rate = row["zero_yield_rate"]
            active_dist = row["active_districts"]
            active_pass = active_dist >= self.config["min_active_districts_in_test"]
            if not active_pass:
                blocking_reasons.append(f"Highly localized/sparse cultivation ({active_dist} active districts < {self.config['min_active_districts_in_test']})")

            zero_pass = zero_rate <= self.config["max_zero_yield_rate_for_ready"]
            if not zero_pass:
                blocking_reasons.append(f"Severe zero-inflation in target ({zero_rate*100:.1f}% zero yield observations)")

            # Compute Component Scores (0 to 100)
            vol_score = min(100.0, (row["total_records"] / 2469.0) * 100.0)
            temp_score = min(100.0, (row["unique_years"] / 8.0) * 100.0 * med_cont)
            geo_score = min(100.0, (row["active_districts"] / 311.0) * 100.0)
            target_score = max(0.0, (1.0 - row["yield_missing_rate"] - (row["zero_yield_rate"] * 0.5)) * 100.0)
            val_score = 100.0 if (yr_pass and active_pass and cont_pass) else 40.0
            feat_score = 100.0  # Area, Production, Yield available

            overall_score = round(
                vol_score * 0.20 +
                temp_score * 0.20 +
                geo_score * 0.25 +
                target_score * 0.20 +
                val_score * 0.15,
                1
            )

            # Determine Status
            if vol_pass and yr_pass and cont_pass and comp_pass and active_pass and zero_pass:
                status = "MODEL_READY"
                recommendation = "Support crop-specific baseline forecasting and prospective ML model development."
            elif row["total_records"] >= 100 and row["unique_years"] >= 3 and row["yield_count"] > 0:
                status = "ANALYTICS_READY"
                recommendation = "Retain for historical analytics, geographic comparisons, and descriptive reporting. Not recommended for predictive modeling due to localized or zero-inflated target."
            else:
                status = "INSUFFICIENT_DATA"
                recommendation = "Insufficient verified observations for statistical inference."

            readiness_records.append({
                "crop": crop,
                "readiness_score": overall_score,
                "readiness_status": status,
                "data_volume_score": round(vol_score, 1),
                "temporal_score": round(temp_score, 1),
                "geographic_score": round(geo_score, 1),
                "target_quality_score": round(target_score, 1),
                "validation_score": round(val_score, 1),
                "feature_score": round(feat_score, 1),
                "blocking_reasons": "; ".join(blocking_reasons) if blocking_reasons else "None (All criteria met)",
                "recommendation": recommendation
            })

            target_profiles.append({
                "crop": crop,
                "yield_mean_kg_ha": row["yield_mean"],
                "yield_median_kg_ha": row["yield_median"],
                "yield_std_kg_ha": row["yield_std"],
                "yield_min_kg_ha": row["yield_min"],
                "yield_max_kg_ha": row["yield_max"],
                "yield_iqr_kg_ha": row["yield_iqr"],
                "yield_cv": row["yield_cv"],
                "zero_yield_pct": round(row["zero_yield_rate"] * 100, 2),
                "active_districts": row["active_districts"]
            })

        readiness_df = pd.DataFrame(readiness_records).sort_values(by="readiness_score", ascending=False)
        target_profiles_df = pd.DataFrame(target_profiles)
        return readiness_df, target_profiles_df

    def evaluate_baselines(self) -> pd.DataFrame:
        """Evaluates 4 simple, zero-leakage baseline forecasting models for all crops."""
        results = []
        train_start, train_end = self.config["train_period"]
        test_start, test_end = self.config["test_period"]

        for crop_name, group in self.df.groupby("crop"):
            # Clean non-zero positive crop records
            valid_df = group[(group["yield_kg_ha"] > 0) & (group["area_ha"] > 0)].copy()

            # Chronological Split
            train_df = valid_df[(valid_df["year"] >= train_start) & (valid_df["year"] <= train_end)]
            test_df = valid_df[(valid_df["year"] >= test_start) & (valid_df["year"] <= test_end)]

            train_count = len(train_df)
            test_count = len(test_df)

            if test_count < 10 or train_count < 20:
                # Infeasible for empirical test evaluation
                results.append({
                    "crop": crop_name,
                    "model": "All Baselines",
                    "train_period": f"{train_start}-{train_end}",
                    "test_period": f"{test_start}-{test_end}",
                    "train_records": train_count,
                    "test_records": test_count,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "r2": np.nan,
                    "mape": np.nan,
                    "smape": np.nan,
                    "valid_predictions": 0,
                    "invalid_predictions": test_count,
                    "notes": "Insufficient train/test observations for out-of-time evaluation",
                    "status": "INFEASIBLE"
                })
                continue

            # Compute Historical Statistics exclusively on Training Data (Zero Leakage)
            crop_train_mean = float(train_df["yield_kg_ha"].mean())
            dist_train_means = train_df.groupby("district")["yield_kg_ha"].mean().to_dict()

            # District Linear Trends (fitted on train only)
            dist_trends = {}
            for dist, d_df in train_df.groupby("district"):
                if len(d_df) >= 2 and d_df["year"].nunique() >= 2:
                    coeffs = np.polyfit(d_df["year"], d_df["yield_kg_ha"], 1)
                    dist_trends[dist] = (coeffs[0], coeffs[1])  # slope, intercept
                else:
                    dist_trends[dist] = (0.0, float(d_df["yield_kg_ha"].mean()))

            # Evaluate Baselines on Test Set
            actuals = test_df["yield_kg_ha"].values

            # Model 1: Naive Persistence (Previous Year's Yield in same district)
            preds_naive = []
            for _, r in test_df.iterrows():
                prev_yr = r["year"] - 1
                prev_val = valid_df[(valid_df["district"] == r["district"]) & (valid_df["year"] == prev_yr)]
                if len(prev_val) > 0 and prev_val.iloc[0]["yield_kg_ha"] > 0:
                    preds_naive.append(prev_val.iloc[0]["yield_kg_ha"])
                elif r["district"] in dist_train_means:
                    preds_naive.append(dist_train_means[r["district"]])
                else:
                    preds_naive.append(crop_train_mean)
            preds_naive = np.array(preds_naive)

            # Model 2: Historical District Mean
            preds_dist_mean = np.array([
                dist_train_means.get(r["district"], crop_train_mean)
                for _, r in test_df.iterrows()
            ])

            # Model 3: Historical Crop Mean
            preds_crop_mean = np.full_like(actuals, fill_value=crop_train_mean)

            # Model 4: Linear District Trend
            preds_trend = []
            for _, r in test_df.iterrows():
                if r["district"] in dist_trends:
                    m, c = dist_trends[r["district"]]
                    val = max(0.0, m * r["year"] + c)
                    preds_trend.append(val if val > 0 else dist_train_means.get(r["district"], crop_train_mean))
                else:
                    preds_trend.append(crop_train_mean)
            preds_trend = np.array(preds_trend)

            # Metric Calculation helper
            def calc_metrics(y_true, y_pred):
                mae = float(np.mean(np.abs(y_true - y_pred)))
                rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
                mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
                smape = float(np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100.0)
                return mae, rmse, r2, mape, smape

            models = [
                ("Naive Persistence (t-1)", preds_naive, "1-year historical district lag persistence"),
                ("Historical District Mean", preds_dist_mean, "District historical mean (2010-2015)"),
                ("Historical Crop Mean", preds_crop_mean, "National historical crop mean (2010-2015)"),
                ("Linear District Trend", preds_trend, "Ordinary least squares district temporal trend"),
            ]

            for m_name, preds, notes in models:
                mae, rmse, r2, mape, smape = calc_metrics(actuals, preds)
                results.append({
                    "crop": crop_name,
                    "model": m_name,
                    "train_period": f"{train_start}-{train_end}",
                    "test_period": f"{test_start}-{test_end}",
                    "train_records": train_count,
                    "test_records": test_count,
                    "mae": round(mae, 2),
                    "rmse": round(rmse, 2),
                    "r2": round(r2, 4),
                    "mape": round(mape, 2),
                    "smape": round(smape, 2),
                    "valid_predictions": test_count,
                    "invalid_predictions": 0,
                    "notes": notes,
                    "status": "EVALUATED"
                })

        return pd.DataFrame(results)

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Runs the entire multi-crop readiness, continuity, scoring, and baseline analysis."""
        os.makedirs(METADATA_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)

        logger.info("Executing Crop Profiling & Classification...")
        readiness_df, target_profiles_df = self.classify_readiness()

        logger.info("Evaluating Temporal Continuity...")
        temporal_df = self.evaluate_temporal_continuity()

        logger.info("Evaluating Baseline Forecasting Models...")
        baselines_df = self.evaluate_baselines()

        # Save Artifacts
        readiness_path = os.path.join(METADATA_DIR, "crop_model_readiness.csv")
        readiness_df.to_csv(readiness_path, index=False)

        target_path = os.path.join(METADATA_DIR, "crop_target_profiles.csv")
        target_profiles_df.to_csv(target_path, index=False)

        temporal_path = os.path.join(METADATA_DIR, "crop_temporal_readiness.csv")
        temporal_df.to_csv(temporal_path, index=False)

        baselines_path = os.path.join(PROCESSED_DIR, "multicrop_baseline_results.csv")
        baselines_df.to_csv(baselines_path, index=False)

        summary = {
            "total_crops": len(readiness_df),
            "model_ready_count": int((readiness_df["readiness_status"] == "MODEL_READY").sum()),
            "analytics_ready_count": int((readiness_df["readiness_status"] == "ANALYTICS_READY").sum()),
            "insufficient_data_count": int((readiness_df["readiness_status"] == "INSUFFICIENT_DATA").sum()),
            "model_ready_crops": readiness_df[readiness_df["readiness_status"] == "MODEL_READY"]["crop"].tolist(),
            "analytics_ready_crops": readiness_df[readiness_df["readiness_status"] == "ANALYTICS_READY"]["crop"].tolist(),
            "insufficient_data_crops": readiness_df[readiness_df["readiness_status"] == "INSUFFICIENT_DATA"]["crop"].tolist(),
            "baselines_evaluated": len(baselines_df),
        }

        logger.info(f"Readiness Pipeline Complete: {summary['model_ready_count']} MODEL_READY, "
                    f"{summary['analytics_ready_count']} ANALYTICS_READY, "
                    f"{summary['insufficient_data_count']} INSUFFICIENT_DATA.")
        return summary


if __name__ == "__main__":
    engine = MultiCropReadinessEngine()
    summary = engine.run_full_pipeline()
    print(json.dumps(summary, indent=2))
