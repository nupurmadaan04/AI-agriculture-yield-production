"""
Exogenous Multi-Crop Model Training & Registry Engine (Day 22).

Trains candidate Model B (Historical + Pre-Season Exogenous Features)
across all 14 evaluated commodities, serializes artifacts, and updates model registry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.exogenous_ablation import EVALUATED_CROPS, CROP_ALGORITHM_MAP, ABLATION_EXPERIMENTS, ExogenousAblationEngine


class ExogenousModelTrainingEngine:
    """Trains and serializes final exogenous models and records lineage."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.models_dir = self.base_dir / "Models" / "exogenous"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.base_dir / "Datasets" / "metadata"
        self.registry_path = self.base_dir / "Models" / "multicrop" / "model_registry.json"
        self.ablation_engine = ExogenousAblationEngine(self.base_dir)

    def train_and_persist_exogenous_models(self) -> Path:
        """
        Trains final full-panel models using Model B (EXP-22E feature set) and Model A (EXP-22A feature set),
        computes final crop results, and updates registry.
        """
        exo_features = ABLATION_EXPERIMENTS[4]["features"]
        hist_features = ABLATION_EXPERIMENTS[0]["features"]

        crop_results: List[Dict[str, Any]] = []

        print("[Exogenous Training] Fitting production models across 14 crops...")

        for crop in EVALUATED_CROPS:
            crop_df = self.ablation_engine.prepare_crop_dataset(crop)
            algo = CROP_ALGORITHM_MAP.get(crop, "RandomForestRegressor")

            # Final evaluation split: Train <= 2016, Test = 2017
            train_mask = (crop_df["year"] >= 1966) & (crop_df["year"] <= 2016)
            test_mask = crop_df["year"] == 2017

            train_data = crop_df[train_mask].dropna(subset=["yield_kg_ha", "yield_lag_1"])
            test_data = crop_df[test_mask].dropna(subset=["yield_kg_ha", "yield_lag_1"])

            y_train = train_data["yield_kg_ha"].values
            y_test = test_data["yield_kg_ha"].values

            # Model C: Baseline
            dist_means = train_data.groupby("district")["yield_kg_ha"].mean().to_dict()
            overall_mean = float(np.mean(y_train))
            baseline_preds = test_data["district"].map(dist_means).fillna(overall_mean).values

            base_mae = float(mean_absolute_error(y_test, baseline_preds))
            base_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_preds)))

            # Model A: Historical Only
            X_train_hist = train_data[hist_features].fillna(0.0).values
            X_test_hist = test_data[hist_features].fillna(0.0).values

            if algo == "RandomForestRegressor":
                model_a = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            else:
                model_a = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)

            model_a.fit(X_train_hist, y_train)
            hist_preds = model_a.predict(X_test_hist)
            hist_mae = float(mean_absolute_error(y_test, hist_preds))
            hist_rmse = float(np.sqrt(mean_squared_error(y_test, hist_preds)))
            hist_r2 = float(r2_score(y_test, hist_preds)) if np.var(y_test) > 0 else 0.0

            # Model B: Exogenous
            X_train_exo = train_data[exo_features].fillna(0.0).values
            X_test_exo = test_data[exo_features].fillna(0.0).values

            if algo == "RandomForestRegressor":
                model_b = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            else:
                model_b = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)

            model_b.fit(X_train_exo, y_train)
            exo_preds = model_b.predict(X_test_exo)
            exo_mae = float(mean_absolute_error(y_test, exo_preds))
            exo_rmse = float(np.sqrt(mean_squared_error(y_test, exo_preds)))
            exo_r2 = float(r2_score(y_test, exo_preds)) if np.var(y_test) > 0 else 0.0

            # Save model B artifact
            crop_clean = crop.lower().replace(" ", "_").replace("&", "and")
            model_path = self.models_dir / f"{crop_clean}_exogenous_model.pkl"
            joblib.dump({
                "crop": crop,
                "algorithm": algo,
                "feature_names": exo_features,
                "model": model_b,
                "version": "Day22_Exogenous_v1.0"
            }, model_path)

            imp_vs_base = round(((base_mae - exo_mae) / base_mae) * 100.0, 2)
            imp_vs_hist = round(((hist_mae - exo_mae) / hist_mae) * 100.0, 2)

            crop_results.append({
                "crop": crop,
                "algorithm": algo,
                "model_a_hist_mae": round(hist_mae, 2),
                "model_a_hist_rmse": round(hist_rmse, 2),
                "model_a_hist_r2": round(hist_r2, 4),
                "model_b_exo_mae": round(exo_mae, 2),
                "model_b_exo_rmse": round(exo_rmse, 2),
                "model_b_exo_r2": round(exo_r2, 4),
                "model_c_base_mae": round(base_mae, 2),
                "model_c_base_rmse": round(base_rmse, 2),
                "exogenous_gain_vs_historical_pct": imp_vs_hist,
                "exogenous_gain_vs_baseline_pct": imp_vs_base,
                "artifact_path": str(model_path.relative_to(self.base_dir)).replace("\\", "/")
            })

        df_results = pd.DataFrame(crop_results)
        out_csv = self.metadata_dir / "exogenous_crop_results.csv"
        df_results.to_csv(out_csv, index=False)

        # Update model registry
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)

            for item in crop_results:
                crop = item["crop"]
                if crop in registry.get("models", {}):
                    c_info = registry["models"][crop]
                    history = c_info.setdefault("history", [])

                    # Add Day 22 entry if not present
                    day22_entry = {
                        "phase": "Day 22: Exogenous Feature Expansion",
                        "model_a_hist_mae": item["model_a_hist_mae"],
                        "model_b_exo_mae": item["model_b_exo_mae"],
                        "model_c_base_mae": item["model_c_base_mae"],
                        "gain_vs_historical_pct": item["exogenous_gain_vs_historical_pct"],
                        "gain_vs_baseline_pct": item["exogenous_gain_vs_baseline_pct"]
                    }
                    if not any(h.get("phase") == day22_entry["phase"] for h in history):
                        history.append(day22_entry)

                    c_info["day22_exogenous"] = item

            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2)
            print(f"[Exogenous Training] Updated {self.registry_path}")

        print(f"[Exogenous Training] Saved crop results to {out_csv}")
        return out_csv


if __name__ == "__main__":
    trainer = ExogenousModelTrainingEngine()
    trainer.train_and_persist_exogenous_models()
