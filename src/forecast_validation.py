"""
Day 24 Forecast Serving Validation & Determinism Benchmark.
Executes dual independent inference passes across all 14 certified crops and verifies rejection guards.
"""

from pathlib import Path
import pandas as pd
import logging

from src.strategy_registry import StrategyRegistry
from src.prediction_service import PredictionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("forecast_validation")

SAMPLE_GEOGRAPHIES = {
    "Oilseeds": ("Punjab", "Ludhiana"),
    "Sugarcane": ("Uttar Pradesh", "Meerut"),
    "Chickpea": ("Madhya Pradesh", "Indore"),
    "Kharif Sorghum": ("Maharashtra", "Solapur"),
    "Minor Pulses": ("Rajasthan", "Jaipur"),
    "Maize": ("Bihar", "Patna"),
    "Wheat": ("Punjab", "Ludhiana"),
    "Rice": ("West Bengal", "Burdwan"),
    "Sesamum": ("Gujarat", "Rajkot"),
    "Pigeonpea": ("Karnataka", "Gulbarga"),
    "Rapeseed and Mustard": ("Rajasthan", "Alwar"),
    "Groundnut": ("Gujarat", "Junagadh"),
    "Sorghum": ("Maharashtra", "Ahmednagar"),
    "Pearl Millet": ("Rajasthan", "Jodhpur"),
}


class ForecastServingValidator:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.service = PredictionService(self.base_dir)
        self.registry = StrategyRegistry(self.base_dir)

    def run_validation(self) -> pd.DataFrame:
        """Executes dual inference passes and safety checks across all commodities."""
        # Compile registry first
        self.registry.compile_strategy_registry()
        self.registry.compile_coverage_metadata()
        self.service.guard._load_metadata()

        cov_df = self.registry.compile_coverage_metadata()

        results = []

        for crop, (default_state, default_dist) in SAMPLE_GEOGRAPHIES.items():
            # Find an actual valid state and district from coverage_df
            crop_cov = cov_df[cov_df["crop"].str.lower() == crop.lower()]
            if not crop_cov.empty:
                state_val = crop_cov.iloc[0]["state"]
                dist_val = crop_cov.iloc[0]["district"]
            else:
                state_val = default_state
                dist_val = default_dist

            # Pass 1
            res1 = self.service.predict_forecast(
                crop=crop,
                state=state_val,
                district=dist_val,
                forecast_year=2017,
            )

            # Pass 2
            res2 = self.service.predict_forecast(
                crop=crop,
                state=state_val,
                district=dist_val,
                forecast_year=2017,
            )

            p1 = res1.get("prediction")
            p2 = res2.get("prediction")
            diff = abs(p1 - p2) if (p1 is not None and p2 is not None) else -1.0
            is_deterministic = (diff == 0.0)

            results.append({
                "crop": crop,
                "state": state_val,
                "district": dist_val,
                "certification_status": res1.get("certification_status"),
                "strategy": res1.get("strategy"),
                "pass_1_pred": p1,
                "pass_2_pred": p2,
                "prediction_diff": diff,
                "is_deterministic": is_deterministic,
                "fallback_used": res1.get("fallback_used"),
                "status": res1.get("status"),
                "provenance_hash": (res1.get("provenance") or {}).get("provenance_hash", ""),
            })

        # Test unsupported crop rejection
        unsupported_res = self.service.predict_forecast(
            crop="Potato",
            state="Punjab",
            district="Ludhiana",
        )
        assert unsupported_res["status"] == "REJECTED"
        assert unsupported_res["error_code"] == "UNSUPPORTED_CROP"

        # Test unsupported district rejection
        unsupported_dist_res = self.service.predict_forecast(
            crop="Oilseeds",
            state="Punjab",
            district="NonExistentDistrict123",
        )
        assert unsupported_dist_res["status"] == "REJECTED"
        assert unsupported_dist_res["error_code"] == "DISTRICT_UNSUPPORTED"

        df_results = pd.DataFrame(results)
        out_csv = self.base_dir / "Datasets" / "metadata" / "forecast_validation_results.csv"
        df_results.to_csv(out_csv, index=False)
        logger.info("Saved forecast serving validation results to %s", out_csv)

        return df_results


if __name__ == "__main__":
    validator = ForecastServingValidator()
    df = validator.run_validation()
    print("\n--- Day 24 Forecast Serving Validation Results ---")
    print(df[["crop", "certification_status", "strategy", "pass_1_pred", "is_deterministic"]].to_string(index=False))
