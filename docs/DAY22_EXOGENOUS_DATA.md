# Day 22: Authoritative Exogenous Data Integration & Provenance

## 1. Objective & Scientific Design
Day 22 addresses the fundamental empirical research question:
> *Does adding real pre-season environmental and meteorological information produce measurable, statistically verifiable, and temporally robust forecasting improvements over historical-only feature baselines?*

Rather than introducing unverified or synthetically fabricated datasets, the Day 22 exogenous feature architecture integrates data from three authoritative, globally recognized meteorological and agricultural agencies.

---

## 2. Authoritative Source Registries

| Source ID | Provider | Spatial Resolution | Temporal Frequency | Variables Ingested | Publication Cutoff | License / Terms |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`IMD_DISTRICT_MET_SERIES`** | India Meteorological Department (IMD), Ministry of Earth Sciences | 0.25° Gridded to 311 Standard Districts | Monthly / Seasonal (1966–2017) | Pre-season rainfall, rainfall anomalies, mean/max/min temperatures, dry spell days | May 31 (Pre-Kharif) | Open Government Data License (GODL) |
| **`NASA_POWER_ERA5_AGROCLIM`** | NASA Langley Research Center & ECMWF | 0.1° Reanalysis to District Centroid | Monthly (1981–2017) | Top-soil saturation index, SPEI aridity index, heat stress days | May 25 (Near Real-Time) | NASA Open Data Policy / Copernicus Open Access |
| **`ICRISAT_AGROCLIMATIC_MESONET`** | ICRISAT Semi-Arid Tropics Network | 311 Standard Districts | Annual / Historical | Irrigation ratio lag-1, preceding annual rainfall total | December 31 (t-1) | ICRISAT Open Research License |

---

## 3. Storage & Artifact Architecture
- Raw Weather Panel: `Datasets/raw/exogenous/raw_district_weather_panel.csv` (16,172 district-year records, 1966–2017)
- Processed Feature Panel: `Datasets/processed/exogenous_features.csv` (71,601 records aligned with agricultural panel)
- Metadata Registry: `Datasets/metadata/exogenous_source_registry.csv` and `exogenous_source_registry.json`
