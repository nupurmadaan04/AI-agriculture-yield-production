# Multi-Crop Dataset Quality & Governance Scorecard

| Dimension | Evaluation Criteria | Status | Details |
| :--- | :--- | :--- | :--- |
| **Source Quality** | Authoritative provider with documented licensing | **COMPLETE** | ICRISAT District Level Database + GoI OGD registered |
| **Crop Diversity** | Comprehensive multi-crop representation ($>20$ crops) | **COMPLETE** | 29 verified crops across Cereals, Pulses, Oilseeds, Cash, Horticulture |
| **Temporal Coverage** | Longitudinal panel spanning continuous harvest cycles | **COMPLETE** | 8 continuous agricultural seasons (2010–2017) |
| **Geographic Coverage** | Sub-national district-level granularity ($>200$ districts) | **COMPLETE** | 311 districts across 20 major Indian states |
| **Variable Coverage** | Area, Production, and Yield available | **COMPLETE** | All three core variables standardized and verified |
| **Unit Consistency** | Exact metric units defined and conversion verified | **COMPLETE** | Standardized to `ha`, `tonnes`, and `kg/ha` |
| **Provenance** | Immutable SHA-256 manifests and source IDs | **COMPLETE** | `ingestion_manifest.json` and `dataset_manifest.json` generated |
| **Data Quality** | Comprehensive multi-dimensional quality engine | **COMPLETE** | 14/14 checks passed (`PASS` overall status) |
| **Duplicate Control** | Zero unhandled duplicate key observations | **COMPLETE** | 0 duplicate records across canonical key |
| **Conflict Detection** | Source collision tracking and resolution logging | **COMPLETE** | Clean provenance mapping in `source_conflicts.csv` |
| **Model Compatibility** | Transparent analytical and ML scope bounding | **COMPLETE** | Rice model boundaries preserved; multi-crop historical analytics live |
