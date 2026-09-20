"""
Unified Agricultural Panel Pipeline Builder.

Processes raw ICRISAT source data, extracts long-format multi-crop panel observations,
applies normalization, executes 14 data quality checks, and creates all canonical metadata registries.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.data_ingestion.ingestion_utils import compute_sha256, profile_dataframe
from src.data_ingestion.icrisat_ingestion import ICRISATIngestion, CROP_PREFIX_MAPPING
from src.data_quality.multicrop_quality import MultiCropDataQualityEngine


def build_pipeline():
    print("=== STARTING UNIFIED AGRICULTURAL PANEL BUILD ===")
    root = Path(__file__).resolve().parents[2]
    raw_icrisat_path = root / "datasets" / "raw" / "icrisat" / "ICRISAT_District_Level_Data_1966_2017_Cleaned.csv"
    processed_dir = root / "datasets" / "processed"
    metadata_dir = root / "datasets" / "metadata"
    docs_dir = root / "docs"

    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingestion & Extraction
    ingestor = ICRISATIngestion(raw_icrisat_path)
    raw_df = ingestor.ingest_raw()
    panel_df = ingestor.extract_long_panel(raw_df)

    panel_csv_path = processed_dir / "agricultural_panel.csv"
    panel_df.to_csv(panel_csv_path, index=False)
    print(f"Unified panel saved to {panel_csv_path} with {len(panel_df)} records.")

    # 2. Ingestion Manifest
    raw_hash = compute_sha256(raw_icrisat_path)
    panel_hash = compute_sha256(panel_csv_path)

    ingestion_manifest = {
        "manifest_version": "1.0.0",
        "generated_at": "2026-09-02T19:20:00Z",
        "ingested_files": [
            {
                "filename": raw_icrisat_path.name,
                "source": "ICRISAT_DLD_1966_2017",
                "download_date": "2026-09-02",
                "sha256": raw_hash,
                "row_count": int(len(raw_df)),
                "column_count": int(len(raw_df.columns)),
                "schema": list(raw_df.columns),
                "units": "Area: 1000 ha, Production: 1000 tons, Yield: kg/ha",
                "geography": "311 Districts across 20 Indian States",
                "time_range": f"{raw_df['Year'].min()}-{raw_df['Year'].max()}",
                "crop_count": len(CROP_PREFIX_MAPPING),
                "processing_status": "INGESTED_AND_TRANSFORMED"
            }
        ]
    }
    with open(metadata_dir / "ingestion_manifest.json", "w") as f:
        json.dump(ingestion_manifest, f, indent=2)

    # 3. Current Dataset Baseline Profile
    profile = {
        "dataset_name": "ICRISAT_District_Level_Data_1966_2017_Cleaned",
        "rows": int(len(raw_df)),
        "columns": int(len(raw_df.columns)),
        "crops_supported": list(CROP_PREFIX_MAPPING.values()),
        "crop_count": len(CROP_PREFIX_MAPPING),
        "states": sorted(raw_df["State Name"].dropna().unique().tolist()),
        "districts_count": int(raw_df["Dist Name"].nunique()),
        "years": [int(raw_df["Year"].min()), int(raw_df["Year"].max())],
        "total_years": int(raw_df["Year"].nunique()),
        "seasons": ["Kharif", "Rabi", "Annual"],
        "units": {
            "area": "1000 ha -> converted to ha (* 1000)",
            "production": "1000 tons -> converted to metric tonnes (* 1000)",
            "yield": "kg/ha"
        },
        "characterization": {
            "is_rice_only": False,
            "crop_level": True,
            "district_level": True,
            "state_level": True,
            "annual": True,
            "seasonal_breakdown": True,
            "has_area": True,
            "has_production": True,
            "has_yield": True
        }
    }
    with open(metadata_dir / "current_dataset_profile.json", "w") as f:
        json.dump(profile, f, indent=2)

    # 4. Crop Coverage Discovery
    crop_rows = []
    for crop_name, group in panel_df.groupby("crop"):
        # Source prefix
        source_prefix = [k for k, v in CROP_PREFIX_MAPPING.items() if v == crop_name][0]
        rec_count = len(group)
        min_y = int(group["year"].min())
        max_y = int(group["year"].max())
        st_count = int(group["state"].nunique())
        dist_count = int(group["district"].nunique())
        seasons_str = "; ".join(sorted(group["season"].unique()))
        has_a = bool(group["area_ha"].notnull().sum() > 0)
        has_p = bool(group["production_tonnes"].notnull().sum() > 0)
        has_y = bool(group["yield_kg_ha"].notnull().sum() > 0)

        crop_rows.append({
            "source": "ICRISAT_DLD_1966_2017",
            "crop_raw": source_prefix,
            "crop_standard": crop_name,
            "records": rec_count,
            "first_year": min_y,
            "last_year": max_y,
            "states": st_count,
            "districts": dist_count,
            "seasons": seasons_str,
            "area_available": has_a,
            "production_available": has_p,
            "yield_available": has_y
        })

    crop_cov_df = pd.DataFrame(crop_rows).sort_values(by="records", ascending=False)
    crop_cov_df.to_csv(metadata_dir / "crop_coverage.csv", index=False)

    # Generate docs/CROP_COVERAGE.md
    crop_md_rows = []
    for _, r in crop_cov_df.iterrows():
        crop_md_rows.append(
            f"| {r['crop_standard']} | {r['records']:,} | {r['first_year']} | {r['last_year']} | {r['states']} | {r['districts']} | {'Yes' if r['area_available'] else 'No'} | {'Yes' if r['production_available'] else 'No'} | {'Yes' if r['yield_available'] else 'No'} |"
        )

    crop_md_content = f"""# Crop Coverage Discovery Report

This report documents the verified multi-crop agricultural coverage extracted from the ICRISAT District-Level Database (1966–2017).

| Crop | Records | First Year | Last Year | States | Districts | Area Available | Production Available | Yield Available |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
""" + "\n".join(crop_md_rows) + "\n"

    with open(docs_dir / "CROP_COVERAGE.md", "w") as f:
        f.write(crop_md_content)

    # 5. Geographic Coverage Discovery
    geo_rows = []
    for state_name, group in panel_df.groupby("state"):
        geo_rows.append({
            "source": "ICRISAT_DLD_1966_2017",
            "state": state_name,
            "district_count": int(group["district"].nunique()),
            "first_year": int(group["year"].min()),
            "last_year": int(group["year"].max()),
            "record_count": int(len(group))
        })
    geo_cov_df = pd.DataFrame(geo_rows).sort_values(by="record_count", ascending=False)
    geo_cov_df.to_csv(metadata_dir / "geographic_coverage.csv", index=False)

    geo_md_rows = [
        f"| {r['state']} | {r['district_count']} | {r['first_year']} | {r['last_year']} | {r['record_count']:,} |"
        for _, r in geo_cov_df.iterrows()
    ]
    geo_md_content = f"""# Geographic Coverage Discovery Report

This report documents the geographical distribution of multi-crop agricultural observations across Indian states and districts.

- **Total States**: {panel_df['state'].nunique()}
- **Total Districts**: {panel_df['district'].nunique()}
- **Total State-Year Observations**: {panel_df[['state', 'year']].drop_duplicates().shape[0]:,}

| State | Districts | First Year | Last Year | Records |
| :--- | :--- | :--- | :--- | :--- |
""" + "\n".join(geo_md_rows) + "\n"

    with open(docs_dir / "GEOGRAPHIC_COVERAGE.md", "w") as f:
        f.write(geo_md_content)

    # 6. Temporal Coverage Discovery
    temp_rows = []
    for yr, group in panel_df.groupby("year"):
        temp_rows.append({
            "year": int(yr),
            "record_count": int(len(group)),
            "crop_count": int(group["crop"].nunique()),
            "state_count": int(group["state"].nunique()),
            "district_count": int(group["district"].nunique())
        })
    temp_cov_df = pd.DataFrame(temp_rows).sort_values(by="year")
    temp_cov_df.to_csv(metadata_dir / "temporal_coverage.csv", index=False)

    temp_md_rows = [
        f"| {r['year']} | {r['record_count']:,} | {r['crop_count']} | {r['state_count']} | {r['district_count']} |"
        for _, r in temp_cov_df.iterrows()
    ]
    temp_md_content = f"""# Temporal Coverage Discovery Report

This report documents the annual panel coverage spanning 1966 to 2017 (51 continuous agricultural years).

- **Minimum Year**: {panel_df['year'].min()}
- **Maximum Year**: {panel_df['year'].max()}
- **Total Span**: {panel_df['year'].max() - panel_df['year'].min() + 1} years
- **Continuous Years Present**: {panel_df['year'].nunique()}

| Year | Records | Crops | States | Districts |
| :--- | :--- | :--- | :--- | :--- |
""" + "\n".join(temp_md_rows) + "\n"

    with open(docs_dir / "TEMPORAL_COVERAGE.md", "w") as f:
        f.write(temp_md_content)

    # 7. Crop Mapping Registry
    crop_map_rows = [
        {"raw_crop": prefix, "standard_crop": std, "source": "ICRISAT_DLD_1966_2017", "mapping_type": "EXACT", "confidence": 1.0, "notes": "Direct prefix mapping from ICRISAT survey schema"}
        for prefix, std in CROP_PREFIX_MAPPING.items()
    ]
    pd.DataFrame(crop_map_rows).to_csv(metadata_dir / "crop_mapping.csv", index=False)

    # 8. Geography Mapping Registry
    geo_map_rows = []
    for _, row in panel_df[["state_raw", "state", "district_raw", "district"]].drop_duplicates().iterrows():
        geo_map_rows.append({
            "raw_state": row["state_raw"],
            "standard_state": row["state"],
            "raw_district": row["district_raw"],
            "standard_district": row["district"],
            "source": "ICRISAT_DLD_1966_2017",
            "mapping_status": "STANDARDIZED",
            "notes": "Title-case sanitized standard name matching 1966 baseline boundaries"
        })
    pd.DataFrame(geo_map_rows).to_csv(metadata_dir / "geography_mapping.csv", index=False)

    # 9. Unit Registry
    unit_reg = {
        "version": "1.0.0",
        "units": [
            {
                "source": "ICRISAT_DLD_1966_2017",
                "variable": "Area",
                "original_unit": "1000 ha",
                "standard_unit": "ha",
                "conversion": "x * 1000",
                "conversion_reference": "ICRISAT DLD Documentation"
            },
            {
                "source": "ICRISAT_DLD_1966_2017",
                "variable": "Production",
                "original_unit": "1000 tons",
                "standard_unit": "tonnes",
                "conversion": "x * 1000",
                "conversion_reference": "ICRISAT DLD Documentation"
            },
            {
                "source": "ICRISAT_DLD_1966_2017",
                "variable": "Yield",
                "original_unit": "Kg per ha",
                "standard_unit": "kg/ha",
                "conversion": "1:1",
                "conversion_reference": "ICRISAT DLD Documentation"
            }
        ]
    }
    with open(metadata_dir / "unit_registry.json", "w") as f:
        json.dump(unit_reg, f, indent=2)

    # 10. Duplicate Audit & Source Conflicts
    dup_mask = panel_df.duplicated(subset=["source", "state", "district", "year", "season", "crop"], keep=False)
    dup_df = panel_df[dup_mask]
    if len(dup_df) == 0:
        dup_audit_df = pd.DataFrame(columns=["record_id", "source", "state", "district", "year", "season", "crop", "duplicate_type"])
    else:
        dup_audit_df = dup_df
    dup_audit_df.to_csv(metadata_dir / "duplicate_audit.csv", index=False)

    conflicts_df = pd.DataFrame(columns=["state", "district", "year", "season", "crop", "variable", "source_a", "value_a", "source_b", "value_b", "difference", "conflict_status", "resolution"])
    conflicts_df.to_csv(metadata_dir / "source_conflicts.csv", index=False)

    # 11. Data Quality Engine Audit
    dq_engine = MultiCropDataQualityEngine(panel_df)
    dq_report = dq_engine.run_all_checks()
    with open(metadata_dir / "data_quality_report.json", "w") as f:
        json.dump(dq_report, f, indent=2)

    # 12. Dataset Manifest
    dataset_manifest = {
        "dataset_version": "AGRI_PANEL_1.0",
        "created_at": "2026-09-02T19:20:00Z",
        "sources": ["ICRISAT_DLD_1966_2017"],
        "records": int(len(panel_df)),
        "crops": sorted(panel_df["crop"].unique().tolist()),
        "crop_count": int(panel_df["crop"].nunique()),
        "states": sorted(panel_df["state"].unique().tolist()),
        "state_count": int(panel_df["state"].nunique()),
        "districts_count": int(panel_df["district"].nunique()),
        "years": [int(panel_df["year"].min()), int(panel_df["year"].max())],
        "schema": list(panel_df.columns),
        "sha256": panel_hash,
        "quality_status": dq_report["overall_status"]
    }
    with open(metadata_dir / "dataset_manifest.json", "w") as f:
        json.dump(dataset_manifest, f, indent=2)

    print("=== UNIFIED AGRICULTURAL PANEL BUILD COMPLETE ===")
    print(f"Total Records: {len(panel_df):,}")
    print(f"Crops: {panel_df['crop'].nunique()}")
    print(f"States: {panel_df['state'].nunique()}")
    print(f"Districts: {panel_df['district'].nunique()}")
    print(f"Years: {panel_df['year'].min()}–{panel_df['year'].max()}")
    print(f"Data Quality Overall Status: {dq_report['overall_status']}")


if __name__ == "__main__":
    build_pipeline()
