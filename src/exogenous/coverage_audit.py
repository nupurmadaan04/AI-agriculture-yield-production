"""
Exogenous Data Coverage & Spatial-Temporal Completeness Audit Engine (Day 22).

Calculates exact coverage percentages, missingness rates, and spatial-temporal
breadth across all 14 evaluated agricultural commodities without data fabrication.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd


class ExogenousCoverageAuditEngine:
    """Audits spatial, temporal, and variable-level data coverage across commodities."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent.parent
        self.metadata_dir = self.base_dir / "Datasets" / "metadata"

    def run_coverage_audit(
        self,
        panel_with_features_df: pd.DataFrame,
        target_crops: List[str]
    ) -> pd.DataFrame:
        """
        Audits coverage for each evaluated commodity across weather, soil, and input features.
        """
        audit_rows: List[Dict[str, Any]] = []

        weather_cols = ["preseason_rainfall_total", "preseason_temp_mean", "preseason_temp_max"]
        soil_cols = ["preseason_soil_moisture", "preseason_aridity_index"]
        all_exo_cols = weather_cols + soil_cols + ["rainfall_lag1_total", "irrigation_ratio_lag1"]

        for crop in target_crops:
            crop_df = panel_with_features_df[panel_with_features_df["crop"] == crop]
            total_records = len(crop_df)

            if total_records == 0:
                continue

            # Compute non-null rates
            weather_valid = crop_df[weather_cols].notna().all(axis=1).sum()
            soil_valid = crop_df[soil_cols].notna().all(axis=1).sum()
            all_valid = crop_df[all_exo_cols].notna().all(axis=1).sum()

            weather_cov_pct = round((weather_valid / total_records) * 100.0, 1)
            soil_cov_pct = round((soil_valid / total_records) * 100.0, 1)
            all_cov_pct = round((all_valid / total_records) * 100.0, 1)
            missing_pct = round(100.0 - all_cov_pct, 1)

            unique_dists = int(crop_df["district"].nunique())
            unique_years = int(crop_df["year"].nunique())

            if all_cov_pct >= 95.0:
                cov_status = "EXCELLENT_COVERAGE"
            elif all_cov_pct >= 85.0:
                cov_status = "ADEQUATE_COVERAGE"
            else:
                cov_status = "INSUFFICIENT_COVERAGE"

            audit_rows.append({
                "crop": crop,
                "records": total_records,
                "weather_coverage_pct": weather_cov_pct,
                "soil_coverage_pct": soil_cov_pct,
                "overall_exogenous_coverage_pct": all_cov_pct,
                "missing_pct": missing_pct,
                "district_coverage": unique_dists,
                "year_coverage": unique_years,
                "coverage_status": cov_status,
                "audit_timestamp": "2026-09-03T18:00:00Z"
            })

        df_audit = pd.DataFrame(audit_rows)
        out_csv = self.metadata_dir / "exogenous_coverage_audit.csv"
        df_audit.to_csv(out_csv, index=False)
        return df_audit


if __name__ == "__main__":
    engine = ExogenousCoverageAuditEngine()
    print("Coverage audit engine initialized.")
