# 3. Dataset Properties & Provenance

### 3.1 Canonical Multi-Crop Panel Overview

The empirical foundation of this study is a longitudinal panel harmonizing district-level agricultural statistics across India:

- **Total Physical Records**: **71,601 rows**
- **Number of Verified Crops**: **29 distinct crops**
- **Geographic Coverage**: **20 Indian states** and **311 districts**
- **Active Temporal Horizon**: **2010–2017** (Normalized multi-crop panel window)
- **Historical Historical Context**: 1966–2017 (Historical ICRISAT baseline context)
- **Primary Physical File**: `Datasets/processed/agricultural_panel.csv`
- **SHA-256 Digital Signature**: `13f882d7d4617e77b6ded31c7febb55e599f3f6a13e981f23b94c4cecd47f13b`

```
+----------------------------------------------------------------------------------------------------+
|                                    CANONICAL PANEL SCHEMA                                          |
+-------------------+---------------+-------------------+--------------------------------------------+
| Column Name       | Physical Type | Measurement Unit  | Description & Provenance                   |
+-------------------+---------------+-------------------+--------------------------------------------+
| record_id         | String (UUID) | None              | Immutable unique row identifier            |
| source            | String        | None              | Authoritative source lineage tag           |
| state             | String        | None              | Normalized Indian state administrative name|
| district          | String        | None              | Harmonized district name (1966 boundaries) |
| year              | Integer       | Gregorian Year    | Agricultural harvest year (2010–2017)      |
| crop              | String        | None              | Standardized commodity name (29 crops)     |
| area_ha           | Float         | Hectares (ha)     | Total gross cultivated crop area           |
| production_tonnes | Float         | Metric Tonnes (t) | Total harvested commodity production       |
| yield_kg_ha       | Float         | Kilograms/ha      | Physical yield (production / area * 1000)  |
| created_at        | Timestamp     | ISO 8601 UTC      | Pipeline ingestion and normalization audit |
+-------------------+---------------+-------------------+--------------------------------------------+
```

### 3.2 Raw Data Sources & Provenance Classification

In compliance with empirical audit standards, all data sources referenced across the project are explicitly classified into four mutually exclusive categories:

| Source Name | Organization / Origin | Classification | Physical Repository Artifact | Role in Study |
|---|---|---|---|---|
| **ICRISAT District Level Database** | International Crops Research Institute for the Semi-Arid Tropics | **DIRECTLY INGESTED** | `Datasets/raw/icrisat/ICRISAT_District_Level_Data_1966_2017_Cleaned.csv` | Core panel containing area, production, and yield for 311 districts |
| **Directorate of Economics & Statistics (DES)** | Ministry of Agriculture & Farmers Welfare, Govt. of India | **DIRECTLY INGESTED** | Ingested via ICRISAT harmonized tables | Validation of state-level aggregate figures |
| **District Agro-Meteorology (IMD)** | India Meteorological Department | **DIRECTLY INGESTED** | `Datasets/metadata/exogenous_coverage_audit.csv` | District monthly rainfall and temperature totals used in Day 22 ablation |
| **Unified Portal for Agricultural Statistics (UPAg)** | Department of Agriculture & Farmers Welfare | **DOCUMENTED SOURCE** | Documented in `source_registry.json` | Architectural roadmap for future cloud synchronization; no live feed |
| **Open Government Data (OGD) Platform** | National Informatics Centre, India | **DOCUMENTED SOURCE** | Ingestion pipeline specifications in `src/` | Supplementary reference metadata |
| **FAOSTAT** | Food and Agriculture Organization of the UN | **METADATA ONLY** | Reference benchmark catalog | Global yield context; not utilized for model training |

### 3.3 Data Quality, Missingness, and Duplicate Audits

1. **Duplicate Records**: The canonical panel contains **0 duplicate entries** under the composite primary key `(crop, state, district, year)`.
2. **Missingness Policy**: Incomplete district-year observations where cultivated area was recorded as zero or null were filtered prior to model feature construction. Missing lag values resulting from historical gaps were handled via forward-fill within the same district or imputed using the district's historical median.
3. **Unit Harmonization**: Raw ICRISAT area (recorded in 1,000 hectares) and production (recorded in 1,000 metric tonnes) were converted to standard SI agricultural units: gross hectares ($\text{ha}$) and metric tonnes ($\text{t}$), yielding yield in kilograms per hectare ($\text{kg/ha}$).
