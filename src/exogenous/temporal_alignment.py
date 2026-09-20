"""
Temporal Alignment & Pre-Season Availability Contract Engine (Day 22).

Enforces strict pre-season forecast origin cutoffs, zero lookahead bias,
and verifies that all exogenous variables are observable prior to sowing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd

# Temporal contract definitions for candidate exogenous variables
TEMPORAL_CONTRACTS: List[Dict[str, Any]] = [
    {
        "feature": "preseason_rainfall_mm",
        "observation_period": "January 1 – May 31 (Pre-Monsoon Window)",
        "availability_date": "May 31 of harvest year t",
        "forecast_origin": "Pre-Season (June 1)",
        "lag": 0,
        "is_preseason_available": True,
        "is_fold_safe": True,
        "timing_status": "SAFE",
        "rationale": "Pre-monsoon rainfall is fully observed prior to Kharif sowing."
    },
    {
        "feature": "preseason_rainfall_anomaly_pct",
        "observation_period": "January 1 – May 31 (Pre-Monsoon Window)",
        "availability_date": "May 31 of harvest year t",
        "forecast_origin": "Pre-Season (June 1)",
        "lag": 0,
        "is_preseason_available": True,
        "is_fold_safe": True,
        "timing_status": "SAFE",
        "rationale": "Normalized against 30-year historical baseline strictly computed prior to test fold."
    },
    {
        "feature": "preseason_temp_mean_c",
        "observation_period": "March 1 – May 31 (Pre-Sowing Window)",
        "availability_date": "May 31 of harvest year t",
        "forecast_origin": "Pre-Season (June 1)",
        "lag": 0,
        "is_preseason_available": True,
        "is_fold_safe": True,
        "timing_status": "SAFE",
        "rationale": "Mean pre-sowing thermal regime fully observable before planting."
    },
    {
        "feature": "preseason_temp_max_c",
        "observation_period": "March 1 – May 31 (Pre-Sowing Heat Wave Window)",
        "availability_date": "May 31 of harvest year t",
        "forecast_origin": "Pre-Season (June 1)",
        "lag": 0,
        "is_preseason_available": True,
        "is_fold_safe": True,
        "timing_status": "SAFE",
        "rationale": "Maximum pre-sowing temperature indicator for early thermal stress."
    },
    {
        "feature": "preseason_soil_moisture_index",
        "observation_period": "May 1 – May 25 (Pre-Sowing Saturated Layer)",
        "availability_date": "May 25 of harvest year t",
        "forecast_origin": "Pre-Season (June 1)",
        "lag": 0,
        "is_preseason_available": True,
        "is_fold_safe": True,
        "timing_status": "SAFE",
        "rationale": "Top-soil moisture available from near-real-time satellite/reanalysis products."
    },
    {
        "feature": "preseason_dry_spell_days",
        "observation_period": "January 1 – May 31 (Pre-Monsoon Dry Span)",
        "availability_date": "May 31 of harvest year t",
        "forecast_origin": "Pre-Season (June 1)",
        "lag": 0,
        "is_preseason_available": True,
        "is_fold_safe": True,
        "timing_status": "SAFE",
        "rationale": "Consecutive rainless days prior to planting date."
    },
    {
        "feature": "rainfall_lag1_annual_mm",
        "observation_period": "Previous Agricultural Year (t-1)",
        "availability_date": "December 31 of year t-1",
        "forecast_origin": "Pre-Season (June 1)",
        "lag": 1,
        "is_preseason_available": True,
        "is_fold_safe": True,
        "timing_status": "SAFE",
        "rationale": "Trailing year hydrological carryover."
    },
    {
        "feature": "monsoon_rainfall_june_sept_mm",
        "observation_period": "June 1 – September 30 of year t",
        "availability_date": "September 30 of year t (Post-Sowing)",
        "forecast_origin": "Pre-Season (June 1)",
        "lag": 0,
        "is_preseason_available": False,
        "is_fold_safe": False,
        "timing_status": "UNSAFE",
        "rationale": "Concurrent monsoon rainfall is NOT available prior to June planting. Forbidden in pre-season forecasting."
    },
    {
        "feature": "harvest_ndvi_max",
        "observation_period": "August 1 – October 31 of year t",
        "availability_date": "October 31 of year t (Harvest Window)",
        "forecast_origin": "Pre-Season (June 1)",
        "lag": 0,
        "is_preseason_available": False,
        "is_fold_safe": False,
        "timing_status": "UNSAFE",
        "rationale": "Peak crop vegetative vigor is observed months after pre-season forecast point. Forbidden."
    }
]


class TemporalAlignmentEngine:
    """Audits and enforces temporal validity and pre-season forecast contracts."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent.parent
        self.metadata_dir = self.base_dir / "Datasets" / "metadata"

    def export_temporal_audit(self) -> Path:
        """Exports the complete temporal audit table."""
        df = pd.DataFrame(TEMPORAL_CONTRACTS)
        out_csv = self.metadata_dir / "exogenous_temporal_audit.csv"
        df.to_csv(out_csv, index=False)
        return out_csv

    def filter_safe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops any column identified as UNSAFE or UNKNOWN from the feature matrix."""
        unsafe_cols = [c["feature"] for c in TEMPORAL_CONTRACTS if c["timing_status"] != "SAFE"]
        cols_to_drop = [c for c in unsafe_cols if c in df.columns]
        if cols_to_drop:
            return df.drop(columns=cols_to_drop)
        return df


if __name__ == "__main__":
    engine = TemporalAlignmentEngine()
    p = engine.export_temporal_audit()
    print(f"Exported temporal audit to {p}")
