# Day 18 Pre-Modeling Audit

## Executive Summary
This pre-modeling audit inspects the active dataset schema, existing model registry, feature representations, validation protocols, and empirical boundaries prior to executing the Multi-Crop Modeling Readiness and Baseline Evaluation framework.

---

## 1. Current Unified Dataset Inspection

- **Canonical File**: `Datasets/processed/agricultural_panel.csv`
- **Total Records**: 71,601
- **Total Columns**: 16
- **Schema**:
  - `record_id` (string, unique primary key: `{source}_{crop}_{state}_{district}_{year}_{season}`)
  - `source` (string: `ICRISAT_DLD_1966_2017`)
  - `dataset_version` (string: `AGRI_PANEL_1.0`)
  - `country` (string: `India`)
  - `state_raw` (string), `state` (string, standardized Title Case)
  - `district_raw` (string), `district` (string, standardized Title Case)
  - `year` (int: 2010–2017)
  - `season_raw` (string), `season` (string: Kharif, Rabi, Annual)
  - `crop_raw` (string), `crop` (string: standardized commodity name)
  - `area_ha` (float, hectares)
  - `production_tonnes` (float, metric tonnes)
  - `yield_kg_ha` (float, kg/ha)
- **Geographic Granularity**: 20 Indian States, 311 Administrative Districts (1966 baseline).
- **Temporal Span**: 8 continuous agricultural seasons (2010–2017).
- **Verified Crops**: 29 crop categories.

---

## 2. Existing Model Registry & Validation Protocol

- **Active Model**: `forecasting_pipeline.pkl` (Random Forest Regressor, 100 trees, max depth 12)
- **Pre-Season Baselines**: `pre_season_rf_pipeline.pkl`, `pre_season_exogenous_pipeline.pkl`
- **Anomaly Detection**: `agricultural_anomaly_pipeline.pkl` (Isolation Forest)
- **Spatial Clusters**: `spatial_cluster_pipeline.pkl` (K-Means, $k=4$)
- **Target Variable**: `RICE YIELD (Kg per ha)`
- **Evaluation Protocol**: Out-of-Time split ($\text{Train} \le 2012$, $\text{Test} = 2013-2017$)
- **Verified Model Performance (Rice Scope)**:
  - $R^2 = 0.7866$
  - $\text{MAE} = 353.01\text{ kg/ha}$
  - $\text{RMSE} = 513.11\text{ kg/ha}$
  - $\text{MAPE} = 18.04\%$
  - Calibration 80% Coverage $= 81.3\%$

---

## 3. Feature Availability & Multi-Crop Leakage Risks

1. **Wide Panel vs Long Panel Covariates**:
   - The unified long panel (`agricultural_panel.csv`) provides `area_ha`, `production_tonnes`, and `yield_kg_ha` across all 29 crops.
   - Additional climate and input features (e.g. `ANNUAL RAINFALL`, `NITROGEN CONSUMPTION`, `POTASH CONSUMPTION`) in the raw 80-column wide panel are recorded at the district level.
2. **Leakage & Simultaneity Risks**:
   - **Target Reconstruction**: In agricultural datasets, $\text{Yield} = \frac{\text{Production}}{\text{Area}}$. Using concurrent harvest-year production to forecast yield constitutes 100% target leakage.
   - **Post-Harvest vs Pre-Season**: Concurrent crop area and harvest outputs are observed only after planting/harvesting, making them invalid as pre-season predictors.
   - **Chronological Discipline**: All baseline models and feature transformations must be strictly fitted on historical training slices ($t \le t_{\text{train}}$) without future lookahead.

---

## 4. Modeling Readiness Objectives for Day 18

1. Profile all 29 crops on data completeness, target distribution, geographic breadth, and temporal continuity.
2. Formulate explicit, configurable sufficiency criteria (`model_readiness_config.json`).
3. Classify each crop into `MODEL_READY`, `ANALYTICS_READY`, or `INSUFFICIENT_DATA`.
4. Establish 4 deterministic, zero-leakage baseline models for all candidate crops:
   - Naive 1-year Persistence ($y_t = y_{t-1}$)
   - Historical District Mean ($\bar{y}_{\text{dist}}$)
   - Historical Crop Mean ($\bar{y}_{\text{crop}}$)
   - Linear District Trend ($\beta_0 + \beta_1 t$)
5. Audit Global vs Crop-Specific architectural paradigms based on empirical scale, variance, and sample balance.
