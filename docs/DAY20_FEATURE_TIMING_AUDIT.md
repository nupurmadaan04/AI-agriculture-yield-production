# Day 20 — Feature Timing & Indirect Leakage Audit

## 1. Zero Leakage Architecture

In pre-season agricultural forecasting, predictions must be made **before the planting season begins**. Any feature derived from post-planting or simultaneous harvest data constitutes mathematical leakage.

---

## 2. Feature-by-Feature Temporal Audit

| Feature Name | Origin / Mathematical Definition | Available at Forecasting Origin ($T_{\text{forecast}}$)? | Pre-Season Valid? | Leakage Classification | Resolution in Walk-Forward Engine |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `yield_lag_1` | Shifted target: $Y_{d, t-1}$ | Yes ($t-1$ harvest completed) | **YES** | ZERO LEAKAGE | Computed via `groupby('district')['yield_kg_ha'].shift(1)`. Imputed strictly with training mean. |
| `yield_lag_2` | Shifted target: $Y_{d, t-2}$ | Yes ($t-2$ harvest completed) | **YES** | ZERO LEAKAGE | Computed via `groupby('district')['yield_kg_ha'].shift(2)`. Imputed strictly with `yield_lag_1` / train mean. |
| `yield_rolling_3yr_mean` | $\frac{1}{3}\sum_{k=1}^3 Y_{d, t-k}$ | Yes (Historical prior 3 years) | **YES** | ZERO LEAKAGE | Computed strictly on shifted series. No contemporary values included. |
| `area_lag_1` | Shifted cultivated area: $A_{d, t-1}$ | Yes ($t-1$ area completed) | **YES** | ZERO LEAKAGE | Computed via `groupby('district')['area_ha'].shift(1)`. |
| `state_encoded` | Categorical state index | Yes (Static geography) | **YES** | ZERO LEAKAGE | State mapping dictionary fit strictly on training slice of each fold. |
| `year` | Linear trend index ($t$) | Yes (Known calendar year) | **YES** | ZERO LEAKAGE | Unaltered integer year. |
| `spatial_cluster_id` | Agglomerative/K-Means cluster ID | **AUDIT WARNING** | **NO (EXCLUDED)** | INDIRECT LEAKAGE | **EXCLUDED**. Spatial clustering fitted over 2010–2017 aggregate panel would inject indirect future lookahead. |
| `production_tonnes` | Contemporary harvest: $P_{d, t}$ | No (Post-harvest output) | **NO (EXCLUDED)** | TARGET LEAKAGE | **EXCLUDED**. Direct mathematical identity ($Y = 1000 \cdot P / A$). |
| `area_ha` | Contemporary planted area: $A_{d, t}$ | No (In-season realization) | **NO (EXCLUDED)** | TIMING LEAKAGE | **EXCLUDED**. Replaced by `area_lag_1`. |

---

## 3. Spatial Cluster Lookahead Audit

During Day 20 research, an audit of `src/spatial_clustering.py` and `Models/spatial_cluster_pipeline.pkl` revealed that spatial cluster centroids were calculated across the entire 2010–2017 dataset. 

If static `spatial_cluster_id` labels are used in walk-forward folds (e.g. predicting 2014 using a cluster generated from 2010–2017 data), future yield patterns indirectly contaminate the cluster boundaries.

**Resolution**:
`CropFeaturePipeline` and `TemporalWalkForwardEngine` strictly **exclude `spatial_cluster_id`** from multi-crop forecasting features, preserving complete temporal separation.
