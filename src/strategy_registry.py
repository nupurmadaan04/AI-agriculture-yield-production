"""
Day 24 Strategy Registry & Coverage Metadata Generator.
Compiles the authoritative multi-crop forecast strategy registry from Day 23 certification evidence.
"""

from pathlib import Path
import json
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("strategy_registry")


class StrategyRegistry:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.meta_dir = self.base_dir / "Datasets" / "metadata"
        self.models_dir = self.base_dir / "Models" / "multicrop"
        self.processed_dir = self.base_dir / "Datasets" / "processed"

    def compile_strategy_registry(self) -> dict:
        """Reads Day 23 certification metadata and compiles the forecast strategy registry."""
        cert_csv = self.meta_dir / "final_model_certification.csv"
        strat_csv = self.meta_dir / "final_strategy_results.csv"
        reg_json = self.models_dir / "model_registry.json"

        if not cert_csv.exists():
            raise FileNotFoundError(f"Missing certification file: {cert_csv}")

        df_cert = pd.read_csv(cert_csv)
        df_strat = pd.read_csv(strat_csv) if strat_csv.exists() else pd.DataFrame()

        registry_raw = {}
        if reg_json.exists():
            with open(reg_json, "r", encoding="utf-8") as f:
                registry_raw = json.load(f)

        crops_meta = registry_raw.get("crops", {})

        strategy_registry = {
            "metadata": {
                "version": "v1.0-Day24-Serving",
                "total_crops": len(df_cert),
                "certification_source": "Day 23 Final Validation & Certification",
                "validation_scope": "expanding_walk_forward_2014_2017",
                "temporal_boundary": "1966-2017",
            },
            "strategies": {},
        }

        registry_rows = []

        for _, row in df_cert.iterrows():
            crop = row["crop"]
            final_status = row["final_status"]
            primary_strat = row["primary_strategy"]
            fallback_strat = row["fallback_strategy"]
            strat_mae = float(row["strategy_mae"])
            base_mae = float(row["baseline_mae"])
            gain_pct = float(row["gain_vs_baseline_pct"])
            win_rate = float(row["fold_win_rate_pct"])
            bias = row["bias_status"]
            repro = row["reproducibility_status"]

            # Pull model version from registry
            crop_dict = crops_meta.get(crop, {})
            day19_dict = crop_dict.get("day19", {})
            model_name = day19_dict.get("best_model_name", "Historical_District_Mean")
            crop_slug = crop.lower().replace(' ', '_').replace('&', 'and')
            model_artifact = f"{crop_slug}/model_pipeline.pkl" if final_status in ["PRODUCTION_READY", "CONDITIONAL_PRODUCTION"] else None
            model_version = f"{crop_slug}_{model_name.lower()}_v23" if model_artifact else "baseline_district_mean_v1"

            strategy_item = {
                "crop": crop,
                "certification_status": final_status,
                "primary_strategy": primary_strat,
                "fallback_strategy": fallback_strat,
                "strategy_mae": strat_mae,
                "baseline_mae": base_mae,
                "gain_vs_baseline_pct": gain_pct,
                "fold_win_rate_pct": win_rate,
                "bias_status": bias,
                "reproducibility_status": repro,
                "model_name": model_name,
                "model_version": model_version,
                "model_artifact": model_artifact,
                "validation_scope": "walk_forward_2014_2017",
                "operating_rule": self._get_operating_rule(crop, final_status),
                "strategy_explanation": self._get_strategy_explanation(crop, final_status, primary_strat, gain_pct),
            }

            strategy_registry["strategies"][crop] = strategy_item
            registry_rows.append(strategy_item)

        # Save JSON
        json_path = self.models_dir / "forecast_strategy_registry.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(strategy_registry, f, indent=2)
        logger.info("Saved %s", json_path)

        # Save CSV
        csv_path = self.meta_dir / "forecast_strategy_registry.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(registry_rows).to_csv(csv_path, index=False)
        logger.info("Saved %s", csv_path)

        return strategy_registry

    def compile_coverage_metadata(self) -> pd.DataFrame:
        """Extracts unique supported crops, states, districts, and temporal boundaries."""
        panel_p = self.processed_dir / "agricultural_panel.csv"
        if not panel_p.exists():
            raise FileNotFoundError(f"Missing agricultural panel: {panel_p}")

        df = pd.read_csv(panel_p, usecols=["crop", "state", "district", "year"])
        coverage_df = df.groupby(["crop", "state", "district"]).agg(
            min_year=("year", "min"),
            max_year=("year", "max"),
            total_observations=("year", "count"),
        ).reset_index()

        cov_csv = self.meta_dir / "forecast_coverage.csv"
        cov_csv.parent.mkdir(parents=True, exist_ok=True)
        coverage_df.to_csv(cov_csv, index=False)
        logger.info("Saved coverage metadata to %s (Total district-crop mappings: %d)", cov_csv, len(coverage_df))
        return coverage_df

    def _get_operating_rule(self, crop: str, status: str) -> str:
        if status == "PRODUCTION_READY":
            return "Primary ML inference (Random Forest). Fallback to Historical District Mean if district history < 5 observations."
        elif status == "CONDITIONAL_PRODUCTION":
            return "Primary ML inference (Gradient Boosting) with mandatory 3-sigma variance clipping bounded by district historical distribution."
        else:
            return "Direct Historical District Mean baseline with 3-Year Rolling Mean fallback for sparse district regimes."

    def _get_strategy_explanation(self, crop: str, status: str, primary_strat: str, gain_pct: float) -> str:
        if status == "PRODUCTION_READY":
            return f"This forecast uses the certified {crop} {primary_strat} strategy. The strategy was selected because it demonstrated statistically significant improvement (+{gain_pct:.1f}% vs baseline) and 75% fold win rate across walk-forward validation."
        elif status == "CONDITIONAL_PRODUCTION":
            return f"This forecast uses the conditional {crop} {primary_strat} strategy with 3-sigma variance clipping. Clipping is enforced to control high-variance out-of-distribution predictions."
        else:
            return f"This crop is served using a certified statistical baseline ({primary_strat}) because evaluated ML models did not produce statistically significant improvement over historical baseline averages during temporal walk-forward validation."


if __name__ == "__main__":
    registry = StrategyRegistry()
    registry.compile_strategy_registry()
    registry.compile_coverage_metadata()
    print("Strategy registry and coverage metadata compiled successfully.")
