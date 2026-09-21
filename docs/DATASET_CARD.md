# Dataset Card: Agricultural Intelligence Unified Panel (`AGRI_PANEL_1.0`)

## 1. Dataset Summary

The **Agricultural Intelligence Unified Panel (`AGRI_PANEL_1.0`)** is a standardized, longitudinal district-level agricultural panel covering 29 verified crop commodities across 20 Indian states and 311 districts over the 1966–2017 historical window (52 consecutive agricultural years).

```
========================================================================================
                      AGRI_PANEL_1.0 DATASET AT A GLANCE
========================================================================================
 Attribute                  Specification
----------------------------------------------------------------------------------------
 Canonical Dataset Name     AGRI_PANEL_1.0
 Primary File Path          Datasets/processed/agricultural_panel.csv
 SHA-256 Checksum           13f882d7d4617e77b6ded31c7febb55e599f3f6a13e981f23b94c4cecd47f13b
 Total Observations         71,601 verified records
 Active Panel Coverage      2010–2017 (8 normalized multi-crop years)
 Historical Panel Context   1966–2017 (ICRISAT baseline records)
 Walk-Forward Origins       2014, 2015, 2016, 2017 (Evaluation splits)
 Total Crops Covered        29 verified commodities
 Model-Evaluated Crops      14 commodities (Walk-forward temporal evaluation)
 Geographic Extent          20 Indian States, 311 Districts
 Primary Target Metric      yield_kg_ha (Kilograms per Hectare)
 Secondary Target Metric    production_tonnes (Metric Tonnes)
 Primary Exposure Metric    area_ha (Cultivated Area in Hectares)
 Quality Audit Passed       14 / 14 Automated Invariant Checks Passed
========================================================================================
```

---

## 2. Dataset Partitioning & Scope Separation

To ensure scientific precision, this repository strictly distinguishes between four related data assets:

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                             DATA ASSET TAXONOMY                                       │
├───────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Unified Agricultural Panel (AGRI_PANEL_1.0)                                       │
│    • 71,601 records covering 29 crops from 1966 to 2017.                              │
│    • Provides district historical means, rolling averages, and crop baselines.        │
├───────────────────────────────────────────────────────────────────────────────────────┤
│ 2. Multi-Crop Modeling Subset                                                         │
│    • 14 commodities meeting strict continuity criteria for temporal walk-forward ML.  │
│    • Zero-leakage pre-season lag features (lag-1, lag-2, 3-yr rolling mean, area).    │
├───────────────────────────────────────────────────────────────────────────────────────┤
│ 3. Exogenous Weather & Market Metadata (Source Tracking)                              │
│    • Distinguishes SOURCE_REGISTERED metadata from SOURCE_DATA_INGESTED telemetry.    │
│    • IMD Gridded Rainfall (0.25°), NASA POWER, agmarknet market arrivals.           │
├───────────────────────────────────────────────────────────────────────────────────────┤
│ 4. Legacy Rice Baseline Dataset                                                       │
│    • ICRISAT district-level panel (Datasets/rice_data.csv) for single-crop benchmarking│
└───────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Source Provenance

| Source Identifier | Source Authority | Ingestion Status | Temporal Coverage | Role in Platform |
| :--- | :--- | :--- | :--- | :--- |
| `ICRISAT_DLD_1966_2017` | ICRISAT District Level Data | `SOURCE_DATA_INGESTED` | 1966–2017 | Primary yield, area, and production panel |
| `DES_GOI_OGD` | Directorate of Economics & Statistics, Ministry of Agriculture | `SOURCE_DATA_INGESTED` | 1997–2017 | District crop verification and cross-validation |
| `IMD_GRIDDED_PRECIP` | India Meteorological Department | `SOURCE_DATA_INGESTED` | 1970–2017 | Gridded rainfall exogenous ablation features |
| `NASA_POWER_METEOROLOGY`| NASA Prediction of Worldwide Energy Resources | `SOURCE_REGISTERED` | 1981–2017 | Registered solar radiation and temperature telemetry |
| `AGMARKNET_MANDI` | Directorate of Marketing & Inspection (DMI) | `SOURCE_REGISTERED` | 2005–2017 | Registered pre-season modal price signals |

---

## 4. Preprocessing & Quality Assurance Pipeline

1. **Unit Standardization**:
   - Production converted to metric tonnes.
   - Area standardized to hectares ($ha$).
   - Yield calculated strictly as $\text{yield\_kg\_ha} = (\text{production\_tonnes} \times 1000) / \text{area\_ha}$.
2. **Missing-Value Policy**:
   - Zero same-period lookahead imputation. Missing lags are imputed strictly using historical training means ($\le t-1$).
3. **Outlier Filtering**:
   - Physical boundary filters: $\text{yield\_kg\_ha} \in [0, 150000]$. Records with negative area or yield are flagged and quarantined.
4. **Data Leakage Guards**:
   - Shift operators ($t-1$, $t-2$) applied strictly within grouped district-crop panels to prevent future temporal leakage.

---

## 5. Known Limitations

- **Boundary Ceiling (2017)**: The primary harmonized historical panel ends in 2017. Post-2017 forecasts are based on historical lag persistence rather than post-2017 observational ground truth.
- **District Boundary Evolution**: Historical district reorganizations between 1966 and 2017 are harmonized to 1966 base district definitions.
- **Statistical Support Only**: The dataset reflects observational agricultural records and does not contain randomized agronomic trial data.
