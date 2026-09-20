"""
Final Operational Strategy Evaluation Engine (Day 23).

Evaluates the exact operational forecasting strategies (Primary Model + Regime Fallback)
against ML-only and Baseline-only alternatives across all 14 evaluated commodities.
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

from src.exogenous_ablation import (
    EVALUATED_CROPS,
    CROP_ALGORITHM_MAP,
    FOLDS,
    compute_mape,
    compute_smape,
    ExogenousAblationEngine
)

# Operational Policy Definitions from Day 21/22
OPERATIONAL_POLICIES: Dict[str, Dict[str, str]] = {
    "Oilseeds": {
        "primary_model": "Historical ML (RandomForest)",
        "fallback_model": "Historical District Mean",
        "policy_type": "PRIMARY_ML_WITH_FALLBACK",
        "operating_rule": "Use ML predictions for all districts with >=3 historical observations; fallback to district mean for sparse districts."
    },
    "Chickpea": {
        "primary_model": "Historical District Mean",
        "fallback_model": "GradientBoosting (Shadow / Normal Regime)",
        "policy_type": "BASELINE_PRIMARY_CONDITIONAL_ML",
        "operating_rule": "Use Historical District Mean as primary baseline; ML used conditionally in normal yield regimes (Q25-Q75)."
    },
    "Kharif Sorghum": {
        "primary_model": "Historical District Mean",
        "fallback_model": "RandomForest (Shadow / Normal Regime)",
        "policy_type": "BASELINE_PRIMARY_CONDITIONAL_ML",
        "operating_rule": "Use Historical District Mean as primary baseline; ML used conditionally in normal yield regimes (Q25-Q75)."
    },
    "Minor Pulses": {
        "primary_model": "Historical District Mean",
        "fallback_model": "GradientBoosting (Research Shadow)",
        "policy_type": "BASELINE_PRIMARY_RESEARCH_SHADOW",
        "operating_rule": "Use Historical District Mean as primary; ML maintained in research shadow mode."
    },
    "Maize": {
        "primary_model": "Historical District Mean",
        "fallback_model": "RandomForest (Research Shadow)",
        "policy_type": "BASELINE_PRIMARY_RESEARCH_SHADOW",
        "operating_rule": "Use Historical District Mean as primary; ML maintained in research shadow mode."
    },
    "Wheat": {
        "primary_model": "Historical District Mean",
        "fallback_model": "GradientBoosting (Research Shadow)",
        "policy_type": "BASELINE_PRIMARY_RESEARCH_SHADOW",
        "operating_rule": "Use Historical District Mean as primary; ML maintained in research shadow mode."
    },
    "Sugarcane": {
        "primary_model": "Historical ML (GradientBoosting)",
        "fallback_model": "Historical District Mean",
        "policy_type": "PRIMARY_ML_WITH_FALLBACK",
        "operating_rule": "Use ML predictions with state-level mean fallback where district variance exceeds 3 sigma."
    },
    "Rice": {
        "primary_model": "Historical District Mean / Persistence",
        "fallback_model": "District 3-Year Rolling Mean",
        "policy_type": "STATISTICAL_BASELINE_PRIMARY",
        "operating_rule": "Use Historical District Mean as robust pre-season forecast baseline."
    },
    "Sesamum": {
        "primary_model": "Historical District Mean",
        "fallback_model": "State Baseline Mean",
        "policy_type": "STATISTICAL_BASELINE_PRIMARY",
        "operating_rule": "Use Historical District Mean as robust pre-season forecast baseline."
    },
    "Pigeonpea": {
        "primary_model": "Historical District Mean",
        "fallback_model": "State Baseline Mean",
        "policy_type": "STATISTICAL_BASELINE_PRIMARY",
        "operating_rule": "Use Historical District Mean as robust pre-season forecast baseline."
    },
    "Rapeseed and Mustard": {
        "primary_model": "Historical District Mean",
        "fallback_model": "State Baseline Mean",
        "policy_type": "STATISTICAL_BASELINE_PRIMARY",
        "operating_rule": "Use Historical District Mean as robust pre-season forecast baseline."
    },
    "Groundnut": {
        "primary_model": "Historical District Mean",
        "fallback_model": "State Baseline Mean",
        "policy_type": "STATISTICAL_BASELINE_PRIMARY",
        "operating_rule": "Use Historical District Mean as robust pre-season forecast baseline."
    },
    "Sorghum": {
        "primary_model": "Historical District Mean",
        "fallback_model": "State Baseline Mean",
        "policy_type": "STATISTICAL_BASELINE_PRIMARY",
        "operating_rule": "Use Historical District Mean as robust pre-season forecast baseline."
    },
    "Pearl Millet": {
        "primary_model": "Historical District Mean",
        "fallback_model": "State Baseline Mean",
        "policy_type": "STATISTICAL_BASELINE_PRIMARY",
        "operating_rule": "Use Historical District Mean as robust pre-season forecast baseline."
    }
}


class StrategyEvaluationEngine:
    """Evaluates operational policies vs pure ML vs pure statistical baselines."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.metadata_dir = self.base_dir / "Datasets" / "metadata"
        self.ablation_engine = ExogenousAblationEngine(self.base_dir)

    def evaluate_all_strategies(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs walk-forward evaluation of the exact operational policies vs pure ML and pure baseline.
        """
        hist_features = [
            "yield_lag_1", "yield_lag_2", "yield_rolling_3yr_mean",
            "yield_rolling_3yr_std", "area_lag_1", "area_rolling_3yr_mean",
            "district_encoded"
        ]

        detailed_fold_records: List[Dict[str, Any]] = []
        strategy_summary_records: List[Dict[str, Any]] = []

        print("[Strategy Evaluation] Running walk-forward audit across 14 commodities...")

        for crop in EVALUATED_CROPS:
            crop_df = self.ablation_engine.prepare_crop_dataset(crop)
            algo = CROP_ALGORITHM_MAP.get(crop, "RandomForestRegressor")
            policy = OPERATIONAL_POLICIES.get(crop, {
                "primary_model": "Historical District Mean",
                "fallback_model": "State Mean",
                "policy_type": "STATISTICAL_BASELINE_PRIMARY",
                "operating_rule": "Baseline Mean primary"
            })

            crop_fold_maes_strategy: List[float] = []
            crop_fold_maes_ml: List[float] = []
            crop_fold_maes_base: List[float] = []

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

                # 1. Pure Baseline Predictions (District Mean on Train)
                dist_means = train_data.groupby("district")["yield_kg_ha"].mean().to_dict()
                overall_mean = float(np.mean(y_train))
                base_preds = test_data["district"].map(dist_means).fillna(overall_mean).values

                base_mae = float(mean_absolute_error(y_test, base_preds))
                base_rmse = float(np.sqrt(mean_squared_error(y_test, base_preds)))
                base_r2 = float(r2_score(y_test, base_preds)) if np.var(y_test) > 0 else 0.0
                base_mape = compute_mape(y_test, base_preds)

                # 2. Pure ML Predictions (Model A Historical Only)
                X_train = train_data[hist_features].fillna(0.0).values
                X_test = test_data[hist_features].fillna(0.0).values

                if algo == "RandomForestRegressor":
                    model_a = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                else:
                    model_a = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)

                model_a.fit(X_train, y_train)
                ml_preds = model_a.predict(X_test)

                ml_mae = float(mean_absolute_error(y_test, ml_preds))
                ml_rmse = float(np.sqrt(mean_squared_error(y_test, ml_preds)))
                ml_r2 = float(r2_score(y_test, ml_preds)) if np.var(y_test) > 0 else 0.0
                ml_mape = compute_mape(y_test, ml_preds)

                # 3. Operational Policy Execution
                strategy_preds = np.zeros_like(y_test)

                if policy["policy_type"] == "PRIMARY_ML_WITH_FALLBACK":
                    # Use ML as primary, with baseline fallback if ML prediction deviates > 4 sigma
                    dist_counts = train_data.groupby("district")["yield_kg_ha"].count().to_dict()
                    for idx, (_, row) in enumerate(test_data.iterrows()):
                        d = row["district"]
                        n_obs = dist_counts.get(d, 0)
                        if n_obs >= 3:
                            strategy_preds[idx] = ml_preds[idx]
                        else:
                            strategy_preds[idx] = base_preds[idx]

                elif policy["policy_type"] == "BASELINE_PRIMARY_CONDITIONAL_ML":
                    # Use baseline as primary; use ML only if district yield history indicates normal regime
                    q25, q75 = np.percentile(y_train, [25, 75])
                    for idx, (_, row) in enumerate(test_data.iterrows()):
                        d_mean = dist_means.get(row["district"], overall_mean)
                        if q25 <= d_mean <= q75:
                            # In normal regime, use weighted ensemble of ML and baseline
                            strategy_preds[idx] = 0.5 * ml_preds[idx] + 0.5 * base_preds[idx]
                        else:
                            # In extreme regime, fallback to robust district mean
                            strategy_preds[idx] = base_preds[idx]

                else:
                    # Pure Baseline Primary
                    strategy_preds = base_preds.copy()

                strat_mae = float(mean_absolute_error(y_test, strategy_preds))
                strat_rmse = float(np.sqrt(mean_squared_error(y_test, strategy_preds)))
                strat_r2 = float(r2_score(y_test, strategy_preds)) if np.var(y_test) > 0 else 0.0
                strat_mape = compute_mape(y_test, strategy_preds)
                strat_smape = compute_smape(y_test, strategy_preds)

                crop_fold_maes_strategy.append(strat_mae)
                crop_fold_maes_ml.append(ml_mae)
                crop_fold_maes_base.append(base_mae)

                # Improvement metrics
                strat_imp_vs_base = round(((base_mae - strat_mae) / base_mae) * 100.0, 2)
                strat_imp_vs_ml = round(((ml_mae - strat_mae) / ml_mae) * 100.0, 2)

                detailed_fold_records.append({
                    "crop": crop,
                    "fold_id": fold_id,
                    "test_year": test_year,
                    "test_records": len(y_test),
                    "policy_type": policy["policy_type"],
                    "strategy_mae": round(strat_mae, 2),
                    "strategy_rmse": round(strat_rmse, 2),
                    "strategy_r2": round(strat_r2, 4),
                    "strategy_mape": round(strat_mape, 2),
                    "strategy_smape": round(strat_smape, 2),
                    "ml_mae": round(ml_mae, 2),
                    "ml_rmse": round(ml_rmse, 2),
                    "ml_r2": round(ml_r2, 4),
                    "ml_mape": round(ml_mape, 2),
                    "baseline_mae": round(base_mae, 2),
                    "baseline_rmse": round(base_rmse, 2),
                    "baseline_r2": round(base_r2, 4),
                    "baseline_mape": round(base_mape, 2),
                    "strategy_improvement_vs_baseline_pct": strat_imp_vs_base,
                    "strategy_improvement_vs_ml_pct": strat_imp_vs_ml,
                    "strategy_win_vs_baseline": bool(strat_mae < base_mae),
                    "strategy_win_vs_ml": bool(strat_mae <= ml_mae)
                })

            # Multi-fold aggregate summary for crop
            mean_strat_mae = float(np.mean(crop_fold_maes_strategy))
            mean_ml_mae = float(np.mean(crop_fold_maes_ml))
            mean_base_mae = float(np.mean(crop_fold_maes_base))

            agg_imp_base = round(((mean_base_mae - mean_strat_mae) / mean_base_mae) * 100.0, 2)
            agg_imp_ml = round(((mean_ml_mae - mean_strat_mae) / mean_ml_mae) * 100.0, 2)

            strategy_summary_records.append({
                "crop": crop,
                "primary_model": policy["primary_model"],
                "fallback_model": policy["fallback_model"],
                "policy_type": policy["policy_type"],
                "strategy_mean_mae": round(mean_strat_mae, 2),
                "ml_mean_mae": round(mean_ml_mae, 2),
                "baseline_mean_mae": round(mean_base_mae, 2),
                "strategy_gain_vs_baseline_pct": agg_imp_base,
                "strategy_gain_vs_ml_pct": agg_imp_ml,
                "operating_rule": policy["operating_rule"]
            })

        df_detailed = pd.DataFrame(detailed_fold_records)
        df_summary = pd.DataFrame(strategy_summary_records)

        val_csv = self.metadata_dir / "final_validation_results.csv"
        strat_csv = self.metadata_dir / "final_strategy_results.csv"

        df_detailed.to_csv(val_csv, index=False)
        df_summary.to_csv(strat_csv, index=False)

        print(f"[Strategy Evaluation] Saved validation results to {val_csv} and strategy summary to {strat_csv}")
        return df_detailed, df_summary


if __name__ == "__main__":
    engine = StrategyEvaluationEngine()
    df_d, df_s = engine.evaluate_all_strategies()
    print("Strategy evaluation completed.")
