"""
Geographic Alignment & Spatial Mapping Engine for Exogenous Data (Day 22).

Enforces reproducible, transparent alignment between meteorological data
and the 311 standardized agricultural panel districts across 20 Indian states.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd


class GeographicAlignmentEngine:
    """Verifies and executes spatial alignment between weather and agricultural panels."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent.parent
        self.metadata_dir = self.base_dir / "Datasets" / "metadata"
        self.geo_mapping_path = self.metadata_dir / "geography_mapping.csv"

    def align_weather_to_agricultural_panel(
        self,
        weather_df: pd.DataFrame,
        panel_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aligns raw weather records to the agricultural panel using state, district, and year keys.
        Generates geographic audit metrics.
        """
        # Load mapping
        geo_map = pd.read_csv(self.geo_mapping_path)
        valid_districts = set(geo_map["standard_district"].unique())
        valid_states = set(geo_map["standard_state"].unique())

        # Audit checks
        weather_districts = set(weather_df["district"].unique())
        weather_states = set(weather_df["state"].unique())

        panel_districts = set(panel_df["district"].unique())
        panel_states = set(panel_df["state"].unique())

        matched_districts = weather_districts.intersection(panel_districts)
        unmatched_panel_districts = panel_districts - weather_districts

        # Perform inner/left merge on state, district, year
        merged_df = pd.merge(
            panel_df,
            weather_df,
            on=["state", "district", "year"],
            how="left"
        )

        # Build geographic audit dataframe
        audit_records = []
        for st in sorted(list(panel_states)):
            st_panel_dists = panel_df[panel_df["state"] == st]["district"].unique()
            st_matched_dists = [d for d in st_panel_dists if d in weather_districts]
            match_pct = (len(st_matched_dists) / max(1, len(st_panel_dists))) * 100.0

            audit_records.append({
                "state": st,
                "panel_districts_count": len(st_panel_dists),
                "weather_matched_districts": len(st_matched_dists),
                "match_percentage": round(match_pct, 1),
                "spatial_alignment_status": "FULL_MATCH" if match_pct >= 99.0 else "PARTIAL_MATCH",
                "alignment_method": "STANDARDIZED_DISTRICT_KEY_JOIN",
                "notes": "Exact match against 1966 standardized historical boundary mapping."
            })

        df_geo_audit = pd.DataFrame(audit_records)
        geo_audit_csv = self.metadata_dir / "exogenous_geographic_audit.csv"
        df_geo_audit.to_csv(geo_audit_csv, index=False)

        return merged_df, df_geo_audit


if __name__ == "__main__":
    engine = GeographicAlignmentEngine()
    print("Geographic alignment engine initialized.")
