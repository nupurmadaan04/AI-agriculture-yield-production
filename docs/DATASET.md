# Multi-Crop Agricultural Dataset Documentation

## 1. Dataset Source & Context
- **Primary Source**: ICRISAT (International Crops Research Institute for the Semi-Arid Tropics) District-Level Database for Indian Agriculture.
- **Secondary Registered Source**: Ministry of Agriculture & Farmers Welfare, Open Government Data (OGD) - Directorate of Economics and Statistics (DES).
- **Benchmark Source**: FAOSTAT Crop Production Database.
- **Active Dataset Version**: **`AGRI_PANEL_1.0`**
- **Scope**: Multi-crop historical agricultural production, land use, and yield panel across India (71,601 standardized records).

---

## 2. Geographical & Multi-Crop Coverage

- **Temporal Coverage**: 2010–2017 (8 continuous agricultural seasons).
- **Geographical Scope**: 20 Indian States, 311 administrative districts (harmonized to 1966 baseline boundaries).
- **Total Unified Panel Records**: **71,601 records** (`datasets/processed/agricultural_panel.csv`).
- **Verified Crops (29 Categories)**:
  - *Cereals*: Rice, Wheat, Kharif Sorghum, Rabi Sorghum, Sorghum, Pearl Millet, Maize, Finger Millet, Barley.
  - *Pulses*: Chickpea (Gram), Pigeonpea (Tur/Arhar), Minor Pulses.
  - *Oilseeds*: Groundnut, Sesamum, Rapeseed and Mustard, Safflower, Castor, Linseed, Sunflower, Soyabean, Total Oilseeds.
  - *Commercial & Forage*: Sugarcane, Cotton, Fodder.
  - *Horticulture*: Fruits, Vegetables, Fruits and Vegetables, Potatoes, Onion.

---

## 3. Standardized Long-Panel Schema

| Field Name | Type | Standardized Unit | Description |
| :--- | :--- | :--- | :--- |
| `record_id` | String | Unique Identifier | Canonical composite record key |
| `source` | String | Categorical | Authoritative source (`ICRISAT_DLD_1966_2017`) |
| `dataset_version` | String | Semantic Version | `AGRI_PANEL_1.0` |
| `country` | String | Categorical | `India` |
| `state_raw` | String | String | Unmodified source state name |
| `state` | String | String (Title Case) | Standardized state name |
| `district_raw` | String | String | Unmodified source district name |
| `district` | String | String (Title Case) | Standardized district name |
| `year` | Integer | Year | Crop year (2010–2017) |
| `season_raw` | String | Categorical | Source season (Kharif, Rabi, Annual) |
| `season` | String | Categorical | Standardized season classification |
| `crop_raw` | String | Categorical | Source survey crop prefix |
| `crop` | String | Categorical | Standardized crop commodity name |
| `area_ha` | Float | Hectares (`ha`) | Standardized cultivated area |
| `production_tonnes`| Float | Metric Tonnes (`tonnes`) | Standardized harvest production output |
| `yield_kg_ha` | Float | Kilograms / Hectare (`kg/ha`)| Standardized crop yield per hectare |

---

## 4. Machine Learning Model Scopes & Guardrails

- **Historical Multi-Crop Analytics**: Available for all **29 verified crops** across all 20 states.
- **Predictive ML Scopes**:
  - Registered Random Forest pipelines (`forecasting_pipeline.pkl`, `pre_season_exogenous_pipeline.pkl`, etc.) are trained and validated exclusively on **Rice**.
  - Scientific validation metrics ($R^2 = 0.7866$, $\text{MAE} = 353.01\text{ kg/ha}$) pertain solely to the registered Rice model.
  - Transparent UI notices inform users when non-rice crops are selected in predictive views.
