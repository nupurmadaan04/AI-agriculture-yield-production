"""
Multi-Crop Forecasting Strategy Module (Day 21)
==============================================
Produces crop-specific operational forecasting policies and fallback architectures
based on Day 21 model selection and error diagnosis.
Updates Models/multicrop/model_registry.json preserving complete lineage history (Day 19 -> 20 -> 21).
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class MultiCropForecastingStrategyEngine:
    """Generates production forecasting policies, fallback logic, and registry updates."""

    def __init__(
        self,
        selection_path: str = "Datasets/metadata/multicrop_model_selection.csv",
        diagnosis_path: str = "Datasets/metadata/multicrop_error_diagnosis.csv",
        registry_path: str = "Models/multicrop/model_registry.json",
        results_path: str = "Datasets/metadata/multicrop_model_results.csv",
    ):
        self.selection_path = selection_path
        self.diagnosis_path = diagnosis_path
        self.registry_path = registry_path
        self.results_path = results_path

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], pd.DataFrame]:
        """Loads selection results, error diagnosis, registry, and model results."""
        sel_df = pd.read_csv(self.selection_path) if os.path.exists(self.selection_path) else pd.DataFrame()
        diag_df = pd.read_csv(self.diagnosis_path) if os.path.exists(self.diagnosis_path) else pd.DataFrame()
        with open(self.registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
        mr_df = pd.read_csv(self.results_path) if os.path.exists(self.results_path) else pd.DataFrame()
        return sel_df, diag_df, registry, mr_df

    def build_crop_strategy(
        self,
        sel_row: pd.Series,
        diag_row: pd.Series,
        mr_crop_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Formulates concrete crop forecasting strategy."""
        crop = sel_row["crop"]
        status = sel_row["day21_status"]
        win_rate = float(sel_row["win_rate"])
        mean_gain = float(sel_row["mean_mae_improvement_pct"])
        selected_ml = diag_row["selected_ml_model"]

        # Default fallback is Historical District Mean
        fallback_model = "Historical District Mean"

        if status == "ROBUST_ML":
            primary_model = selected_ml
            operating_conditions = (
                "Primary operational forecaster for pre-season yield estimation across all districts. "
                "Fallback to Historical District Mean only when district lag features are missing."
            )
            diagnostic_notes = (
                f"Demonstrated superior temporal stability across {win_rate}% of walk-forward origins "
                f"with an average MAE gain of +{mean_gain}%. Error quantiles indicate tight dispersion."
            )
            missing_gaps = "Pre-season weather indices (monsoon onset timing, cumulative precipitation)."
            evidence_required = "Maintain >70% win rate when expanding to future post-2017 origins."
            evidence_score = round(min(100.0, 60.0 + (win_rate * 0.3) + max(0.0, mean_gain * 1.5)), 1)
            evidence_interp = "Strong empirical walk-forward evidence supporting production deployment."

        elif status == "ML_WITH_CONDITIONS":
            primary_model = selected_ml
            operating_conditions = (
                "Conditional ML forecasting. Primary in normal agro-climatic conditions; "
                "switch to Historical District Mean / Naive Persistence during climate shock regimes "
                "or when lagged inputs deviate >2 standard deviations from district historical norms."
            )
            diagnostic_notes = (
                f"Moderate win rate ({win_rate}%) and positive gain (+{mean_gain}%), but elevated error "
                "observed in anomalous years (e.g. 2016 high-error regime) or extreme yield quantiles."
            )
            missing_gaps = "High-resolution soil moisture and sub-district irrigation access data."
            evidence_required = "Demonstrate error mitigation in anomalous years with climate covariates."
            evidence_score = round(min(80.0, 45.0 + (win_rate * 0.3) + max(0.0, mean_gain * 1.0)), 1)
            evidence_interp = "Moderate conditional evidence; requires operational guardrails."

        elif status == "RESEARCH_CANDIDATE":
            primary_model = fallback_model
            fallback_model = "Naive Persistence"
            operating_conditions = (
                "Deploy statistical baseline (Historical District Mean) as primary operational forecaster. "
                f"ML model ({selected_ml}) maintained in shadow/research mode for offline evaluation."
            )
            diagnostic_notes = (
                f"ML shows localized predictive signal in selected districts, but overall walk-forward "
                f"win rate is low ({win_rate}%) or mean gain is negative ({mean_gain}%)."
            )
            missing_gaps = "Direct weather observations (precipitation, temperature anomalies) and satellite NDVI."
            evidence_required = "Substantial MAE improvement over baseline in >=3 walk-forward origins with exogenous data."
            evidence_score = round(min(50.0, 30.0 + (win_rate * 0.2)), 1)
            evidence_interp = "Promising localized research signal, but insufficient for general forecasting."

        else:  # BASELINE_PREFERRED
            primary_model = fallback_model
            fallback_model = "Naive Persistence"
            operating_conditions = (
                "Statistical baseline (Historical District Mean) is the primary forecasting engine. "
                "No ML model deployed to production for this crop."
            )
            diagnostic_notes = (
                f"ML consistently degraded forecast accuracy (win rate {win_rate}%, gain {mean_gain}%). "
                "District historical yield distributions provide lower variance and superior robustness."
            )
            missing_gaps = "Crop-specific input data (fertilizer application, pest incidence, cultivar distribution)."
            evidence_required = "Complete structural feature overhaul before reconsidering ML."
            evidence_score = round(min(40.0, 20.0 + (win_rate * 0.2)), 1)
            evidence_interp = "Clear baseline superiority; ML is not currently defensible."

        return {
            "crop": crop,
            "day21_status": status,
            "primary_forecasting_model": primary_model,
            "fallback_model": fallback_model,
            "operating_conditions": operating_conditions,
            "diagnostic_notes": diagnostic_notes,
            "missing_information_gaps": missing_gaps,
            "evidence_required": evidence_required,
            "evidence_strength_score": evidence_score,
            "evidence_interpretation": evidence_interp,
        }

    def execute_all(
        self,
        output_dir: str = "Datasets/metadata",
        update_registry: bool = True,
    ) -> Dict[str, Any]:
        """Generates strategies and updates model_registry.json."""
        os.makedirs(output_dir, exist_ok=True)
        sel_df, diag_df, registry, mr_df = self.load_data()

        strategy_records = []
        diag_map = {r["crop"]: r for _, r in diag_df.iterrows()}

        for _, sel_row in sel_df.iterrows():
            crop = sel_row["crop"]
            diag_row = diag_map.get(crop, pd.Series({"selected_ml_model": "RandomForestRegressor"}))
            mr_crop = mr_df[mr_df["crop"] == crop] if not mr_df.empty else pd.DataFrame()
            strat = self.build_crop_strategy(sel_row, diag_row, mr_crop)
            strategy_records.append(strat)

            # Update Registry
            if update_registry and "models" in registry and crop in registry["models"]:
                entry = registry["models"][crop]
                day19_st = entry.get("status", "BASELINE_PREFERRED")
                day20_st = entry.get("day20_robustness", {}).get("robustness_status", "SPLIT_SENSITIVE")
                day21_st = strat["day21_status"]

                # Ensure lineage history list exists
                history = entry.get("history", [])
                # Rebuild/clean history for idempotency
                history = [h for h in history if h.get("day") not in [19, 20, 21]]
                history.append({"day": 19, "status": day19_st, "phase": "Multi-Crop Lag Modeling"})
                history.append({"day": 20, "status": day20_st, "phase": "Temporal Walk-Forward Validation"})
                history.append({"day": 21, "status": day21_st, "phase": "Error Diagnosis & Strategy Selection"})
                entry["history"] = history

                entry["day21_diagnosis"] = {
                    "day21_status": day21_st,
                    "primary_forecasting_model": strat["primary_forecasting_model"],
                    "fallback_model": strat["fallback_model"],
                    "operating_conditions": strat["operating_conditions"],
                    "diagnostic_notes": strat["diagnostic_notes"],
                    "missing_information_gaps": strat["missing_information_gaps"],
                    "evidence_required": strat["evidence_required"],
                    "evidence_strength_score": strat["evidence_strength_score"],
                    "evidence_interpretation": strat["evidence_interpretation"],
                    "win_rate": float(sel_row["win_rate"]),
                    "mean_mae_improvement_pct": float(sel_row["mean_mae_improvement_pct"]),
                    "decision_basis": sel_row["decision_basis"],
                }

        strat_df = pd.DataFrame(strategy_records)
        strat_df.to_csv(os.path.join(output_dir, "multicrop_forecasting_strategy.csv"), index=False)

        if update_registry:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2)

        return {
            "total_strategies": len(strat_df),
            "registry_updated": update_registry,
        }


if __name__ == "__main__":
    engine = MultiCropForecastingStrategyEngine()
    stats = engine.execute_all()
    print(f"Forecasting strategy completed: {stats}")
