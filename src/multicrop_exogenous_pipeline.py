"""
Master Multi-Crop Exogenous Pipeline Engine (Day 22).

Executes end-to-end ingestion, geographic alignment, temporal contract enforcement,
feature engineering, coverage auditing, and leakage certification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List, Tuple
import pandas as pd

from src.exogenous.source_registry import ExogenousSourceRegistry
from src.exogenous.weather_ingestion import WeatherIngestionEngine
from src.exogenous.geographic_alignment import GeographicAlignmentEngine
from src.exogenous.temporal_alignment import TemporalAlignmentEngine
from src.exogenous.feature_engineering import ExogenousFeatureEngineeringEngine
from src.exogenous.coverage_audit import ExogenousCoverageAuditEngine
from src.exogenous.leakage_audit import ExogenousLeakageAuditEngine

EVALUATED_CROPS: List[str] = [
    "Rice", "Wheat", "Kharif Sorghum", "Sorghum", "Pearl Millet",
    "Maize", "Chickpea", "Pigeonpea", "Minor Pulses", "Groundnut",
    "Sesamum", "Rapeseed and Mustard", "Oilseeds", "Sugarcane"
]


class MultiCropExogenousPipeline:
    """Master pipeline orchestrating all exogenous data integration steps."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.datasets_dir = self.base_dir / "Datasets"
        self.metadata_dir = self.datasets_dir / "metadata"
        self.processed_dir = self.datasets_dir / "processed"
        self.panel_path = self.processed_dir / "agricultural_panel.csv"

        self.source_registry = ExogenousSourceRegistry(self.base_dir)
        self.weather_ingestion = WeatherIngestionEngine(self.base_dir)
        self.geo_alignment = GeographicAlignmentEngine(self.base_dir)
        self.temporal_alignment = TemporalAlignmentEngine(self.base_dir)
        self.feature_engineering = ExogenousFeatureEngineeringEngine(self.base_dir)
        self.coverage_audit = ExogenousCoverageAuditEngine(self.base_dir)
        self.leakage_audit = ExogenousLeakageAuditEngine(self.base_dir)

    def run_pipeline(self) -> Dict[str, Any]:
        """
        Executes the full pipeline and generates all processed datasets and metadata files.
        """
        print("[Day 22 Pipeline] Step 1: Exporting Exogenous Source Registry...")
        src_csv, src_json = self.source_registry.export_registry()

        print("[Day 22 Pipeline] Step 2: Ingesting Raw Meteorological Panel...")
        raw_weather_csv = self.weather_ingestion.generate_raw_district_weather_panel()
        raw_weather_df = pd.read_csv(raw_weather_csv)

        print("[Day 22 Pipeline] Step 3: Aligning Geographies with Agricultural Panel...")
        if not self.panel_path.exists():
            raise FileNotFoundError(f"Missing agricultural panel at {self.panel_path}")
        panel_df = pd.read_csv(self.panel_path)

        # Merge weather into agricultural panel
        merged_df, geo_audit_df = self.geo_alignment.align_weather_to_agricultural_panel(
            raw_weather_df,
            panel_df
        )

        print("[Day 22 Pipeline] Step 4: Enforcing Temporal Pre-Season Contracts...")
        temp_audit_csv = self.temporal_alignment.export_temporal_audit()

        print("[Day 22 Pipeline] Step 5: Engineering Agronomic Pre-Season Features...")
        feat_reg_csv = self.feature_engineering.export_feature_registry()
        features_df = self.feature_engineering.engineer_exogenous_features(merged_df)

        # Drop any unsafe features
        safe_features_df = self.temporal_alignment.filter_safe_features(features_df)

        # Save processed exogenous dataset
        out_processed_csv = self.processed_dir / "exogenous_features.csv"
        safe_features_df.to_csv(out_processed_csv, index=False)
        print(f"[Day 22 Pipeline] Saved processed features to {out_processed_csv} ({len(safe_features_df)} rows)")

        print("[Day 22 Pipeline] Step 6: Running Coverage & Leakage Audits...")
        cov_audit_df = self.coverage_audit.run_coverage_audit(safe_features_df, EVALUATED_CROPS)
        leakage_audit_csv = self.leakage_audit.export_leakage_audit()

        summary = {
            "status": "SUCCESS",
            "total_records": len(safe_features_df),
            "total_crops_evaluated": len(EVALUATED_CROPS),
            "processed_file": str(out_processed_csv),
            "source_registry_csv": str(src_csv),
            "feature_registry_csv": str(feat_reg_csv),
            "coverage_audit_records": len(cov_audit_df),
            "geographic_matched_states": len(geo_audit_df),
        }
        print("[Day 22 Pipeline] Completed successfully.")
        return summary


if __name__ == "__main__":
    pipeline = MultiCropExogenousPipeline()
    res = pipeline.run_pipeline()
    print(json.dumps(res, indent=2))
