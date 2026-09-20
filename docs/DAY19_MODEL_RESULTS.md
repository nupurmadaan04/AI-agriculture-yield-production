# Day 19: Multi-Crop Forecasting Empirical Results & Benchmark Comparison

## 1. Overall Findings Summary

Across all **14 `MODEL_READY` crops** evaluated under the strict chronological test protocol (Train: 2011–2015, Test: 2016–2017), the empirical results are:

- **Total Crops Evaluated**: 14
- **Accepted ML Models**: 4 crops (`Sesamum`, `Maize`, `Pigeonpea`, `Sugarcane`)
- **Baseline-Preferred Crops**: 10 crops (`Minor Pulses`, `Rice`, `Chickpea`, `Wheat`, `Oilseeds`, `Rapeseed and Mustard`, `Groundnut`, `Sorghum`, `Kharif Sorghum`, `Pearl Millet`)
- **Winning ML Model Family**: Random Forest (8 wins) vs Gradient Boosting (6 wins)
- **Top MAE Improvement Crop**: `Sesamum` (+2.61% improvement over baseline)

---

## 2. Multi-Crop Forecasting Leaderboard

| Crop | Best Model | Best MAE (kg/ha) | Baseline Model | Baseline MAE | MAE Imp (%) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sesamum** | `RandomForestRegressor` | **114.62** | Historical District Mean | 117.69 | **+2.61%** | `ACCEPTED` |
| **Maize** | `RandomForestRegressor` | **744.73** | Historical District Mean | 763.91 | **+2.51%** | `ACCEPTED` |
| **Pigeonpea** | `GradientBoostingRegressor` | **295.41** | Naive Persistence ($t-1$) | 299.15 | **+1.25%** | `ACCEPTED` |
| **Sugarcane** | `GradientBoostingRegressor` | **1121.27** | Naive Persistence ($t-1$) | 1126.31 | **+0.45%** | `ACCEPTED` |
| **Rice** | `Historical District Mean` | **367.49** | Historical District Mean | 367.49 | -3.16% (RF: 379.09) | `BASELINE_PREFERRED` |
| **Pearl Millet** | `Historical District Mean` | **329.83** | Historical District Mean | 329.83 | -4.86% (RF: 345.85) | `BASELINE_PREFERRED` |
| **Groundnut** | `Historical District Mean` | **302.18** | Historical District Mean | 302.18 | -6.05% (RF: 320.45) | `BASELINE_PREFERRED` |
| **Kharif Sorghum** | `Historical District Mean` | **331.18** | Historical District Mean | 331.18 | -6.21% (RF: 351.75) | `BASELINE_PREFERRED` |
| **Sorghum** | `Historical District Mean` | **305.79** | Historical District Mean | 305.79 | -7.01% (RF: 327.23) | `BASELINE_PREFERRED` |
| **Minor Pulses** | `Historical District Mean` | **322.30** | Historical District Mean | 322.30 | -9.71% (GB: 353.59) | `BASELINE_PREFERRED` |
| **Chickpea** | `Historical District Mean` | **274.42** | Historical District Mean | 274.42 | -12.93% (GB: 309.91) | `BASELINE_PREFERRED` |
| **Rapeseed & Mustard** | `Naive Persistence (t-1)` | **224.95** | Naive Persistence ($t-1$) | 224.95 | -14.02% (GB: 256.48) | `BASELINE_PREFERRED` |
| **Oilseeds** | `Naive Persistence (t-1)` | **746.72** | Naive Persistence ($t-1$) | 746.72 | -14.16% (RF: 852.48) | `BASELINE_PREFERRED` |
| **Wheat** | `Naive Persistence (t-1)` | **396.55** | Naive Persistence ($t-1$) | 396.55 | -33.54% (GB: 529.54) | `BASELINE_PREFERRED` |

---

## 3. Key Scientific Insights

1. **Strength of Localized Statistical Baselines**:
   - In 10 out of 14 crops, simple historical district averages or last-year persistence outperformed non-linear decision tree ensembles.
   - This occurs because district-level soil quality, agro-ecological constraints, and irrigation infrastructure create persistent mean yield levels that tree models with limited panel depth cannot easily surpass without exogenous in-season weather data.
2. **Why Random Forest and Gradient Boosting Won on Maize, Sesamum, Pigeonpea, and Sugarcane**:
   - These 4 crops display non-linear shifts in regional crop area allocation (`area_lag_1`) and multi-year cyclical yield momentum that the tree models successfully leveraged.
3. **Preservation of Scientific Integrity**:
   - Rather than forcing an ML winner by manipulating test years or features, the platform honestly documents when statistical baselines are superior.
