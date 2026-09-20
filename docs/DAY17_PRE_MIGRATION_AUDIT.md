# Day 17 Pre-Migration Audit

## Executive Summary
This pre-migration audit documents the active dataset architecture, backend services, frontend components, model dependencies, and migration risks prior to upgrading the data foundation to a unified multi-crop agricultural data architecture.

---

## 1. Current Datasets & Physical Files

| Dataset Path | Rows | Columns | Scope & Crop Coverage | Current Role |
| :--- | :--- | :--- | :--- | :--- |
| `Datasets/rice_data_outlier_removed.csv` | 2,469 | 80 | ICRISAT panel containing 29 agricultural crop categories across 311 districts (1966–2017) | Active canonical dataset loaded by `data_loader.py` |
| `Datasets/Crops_data.csv` | 2,484 | 80 | Full un-quarantined ICRISAT panel with multi-crop area, production, and yield | Source multi-crop reference dataset |
| `Datasets/rice_data.csv` | 2,469 | 8 | Filtered 8-column subset containing only Rice variables | Legacy subset |

---

## 2. Current Schemas & Crop Availability in Storage

The active ICRISAT dataset (`Datasets/rice_data_outlier_removed.csv`) already houses comprehensive multi-crop variables in wide format:
1. **Cereals**: Rice, Wheat, Kharif Sorghum, Rabi Sorghum, Sorghum (Total), Pearl Millet (Bajra), Maize, Finger Millet (Ragi), Barley.
2. **Pulses**: Chickpea (Gram), Pigeonpea (Tur/Arhar), Minor Pulses.
3. **Oilseeds**: Groundnut, Sesamum, Rapeseed and Mustard, Safflower, Castor, Linseed, Sunflower, Soyabean, Total Oilseeds.
4. **Commercial & Cash Crops**: Sugarcane, Cotton.
5. **Horticulture & Others**: Fruits, Vegetables, Fruits & Vegetables, Potatoes, Onion, Fodder.

Each crop profile contains:
- `AREA (1000 ha)`
- `PRODUCTION (1000 tons)`
- `YIELD (Kg per ha)`

---

## 3. Current Backend Services & Crop Assumptions

- **`backend/utils/data_loader.py`**: Hardcodes validation for `RICE AREA`, `RICE PRODUCTION`, `RICE YIELD`.
- **`backend/services/agriculture_service.py`**: Serves regional and crop statistics; requires dynamic multi-crop resolver.
- **`backend/services/ml_service.py` & `forecast_service.py`**: Models are trained on `RICE YIELD (Kg per ha)`.
- **`backend/services/temporal_monitoring_service.py` & `risk_service.py`**: Default to Rice yield trends.
- **`backend/services/explainability_service.py`**: Decomposes features for the registered Rice model.
- **`backend/services/decision_intelligence_service.py`**: Accepts `crop` parameter, defaults to `"Rice"`.

---

## 4. Current Frontend Assumptions

- **`FilterBar.tsx`**: Hardcodes `Rice` in dropdown and marks Wheat/Maize as disabled.
- **`mockAgricultureData.ts`**: Static fallback data structure.
- **`AgriculturalIntelligence.tsx`**: Displays Rice forecasts.
- **`ScienceWhitepaper.tsx`**: Documents Rice modeling metrics ($R^2 = 0.7866$, $\text{MAE} = 353.01\text{ kg/ha}$).

---

## 5. Model Dependencies & Scientific Guardrails

- **Active Registered Models**:
  - `forecasting_pipeline.pkl` (Post-harvest Random Forest) -> **Rice Only**
  - `pre_season_rf_pipeline.pkl` (Pre-season baseline RF) -> **Rice Only**
  - `pre_season_exogenous_pipeline.pkl` (Advanced exogenous RF) -> **Rice Only**
  - `agricultural_anomaly_pipeline.pkl` (Isolation Forest) -> **Rice Only**
  - `spatial_cluster_pipeline.pkl` (K-Means spatial clusterer) -> **Spatial/Rainfall**
- **Safety Policy**:
  - Historical multi-crop analytics (Area, Production, Yield, Trends, Comparisons) will be enabled across all verified crops.
  - Forecasting, Scenario Simulation, and Tree SHAP XAI will remain explicitly bounded to the registered model scope (Rice) with clear UI notices: *"Forecast unavailable for this crop under the current registered model. Historical analytics remain available."*

---

## 6. Migration Risks & Mitigation Strategy

1. **Risk of API Breaking Changes**:
   - *Mitigation*: Ensure all new endpoints (`/api/agriculture/*`) are purely additive; maintain existing parameter defaults (`crop="Rice"`).
2. **Risk of Schema Leakage / Missing Columns**:
   - *Mitigation*: Provide explicit column mapping and unit standardization (`hectares`, `metric tonnes`, `kg/ha`).
3. **Risk of Silent Model Misattribution**:
   - *Mitigation*: Implement model compatibility gatekeeper in `backend/services/agriculture_service.py` and `forecast_service.py`.
4. **Risk of Test Regressions**:
   - *Mitigation*: Run `pytest -q` continuously during development to ensure 100% test pass rate across all 283 existing regression tests.
