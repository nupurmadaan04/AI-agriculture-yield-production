# Day 20 — Final Multi-Crop Scorecard & Verification Record

## 1. System-Wide Multi-Crop Modeling Status

- **Total Agricultural Crops Evaluated**: 29
- **Model-Ready Crops with Full 8-Year Panel**: 14
- **Walk-Forward Validation Folds Executed**: 280 Folds
- **Validated Candidate ML Models (`ROBUST_ACCEPTED`)**: **2** (`Chickpea`, `Oilseeds`)
- **Split-Sensitive Candidates (`SPLIT_SENSITIVE`)**: **10**
- **Statistical Baseline Preferred (`BASELINE_PREFERRED`)**: **2** (`Pigeonpea`, `Pearl Millet`)
- **Legacy Standalone Rice Model**: **Preserved & Pristine** ($R^2 = 0.7866$, $\text{MAE} = 353.01$ kg/ha)

---

## 2. Complete 14-Commodity Final Scorecard

```
========================================================================================================================
COMMODITY              ALGORITHM                 WF MAE    BASE MAE   GAIN(%)   MEDIAN(%)  WIN RATE  SCORE  STATUS
========================================================================================================================
Chickpea               GradientBoostingRegressor 257.58    263.88     +2.32%    +6.11%     75.0%     70.3   ROBUST_ACCEPTED
Oilseeds               RandomForestRegressor     465.23    521.40    +12.79%    +5.18%     75.0%     80.5   ROBUST_ACCEPTED
Sugarcane              RandomForestRegressor    1410.85   1431.02     +1.17%    +0.18%     50.0%     58.7   SPLIT_SENSITIVE
Kharif Sorghum         RandomForestRegressor     290.38    293.85     +1.30%    +0.79%     50.0%     59.9   SPLIT_SENSITIVE
Minor Pulses           GradientBoostingRegressor 329.96    315.00     -5.76%    -7.69%     50.0%     56.0   SPLIT_SENSITIVE
Wheat                  RandomForestRegressor     418.45    381.65    -13.21%   -16.04%     50.0%     55.0   SPLIT_SENSITIVE
Groundnut              RandomForestRegressor     311.92    289.68     -7.87%    -6.62%     25.0%     48.3   SPLIT_SENSITIVE
Sesamum                RandomForestRegressor     139.00    137.38     -1.19%    -2.22%     25.0%     47.6   SPLIT_SENSITIVE
Maize                  RandomForestRegressor     653.67    639.91     -2.63%    -3.05%     25.0%     47.1   SPLIT_SENSITIVE
Sorghum                RandomForestRegressor     274.04    264.16     -3.50%    -1.45%     25.0%     47.1   SPLIT_SENSITIVE
Rapeseed & Mustard     RandomForestRegressor     199.70    191.95     -5.62%    -4.42%     25.0%     46.9   SPLIT_SENSITIVE
Rice (Multi-Crop Panel)RandomForestRegressor     336.35    310.28     -8.39%   -10.76%     25.0%     46.9   SPLIT_SENSITIVE
Pearl Millet           RandomForestRegressor     274.83    259.23     -6.04%    -6.07%      0.0%     39.4   BASELINE_PREFERRED
Pigeonpea              RandomForestRegressor     293.12    276.41     -5.68%    -3.52%      0.0%     36.9   BASELINE_PREFERRED
========================================================================================================================
```

---

## 3. Metadata Files Generated

1. `Datasets/metadata/multicrop_temporal_robustness.csv` (14 commodities with walk-forward aggregates)
2. `Datasets/metadata/multicrop_fold_results.csv` (280 walk-forward fold evaluations)
3. `Datasets/metadata/model_robustness_scores.csv` (14 composite stability scores and recommendations)
4. `Models/multicrop/model_registry.json` (Updated with Day 20 robustness schema and lineage tracking)

---

## 4. API Endpoints Verified

- `GET /api/modeling/robustness` — 200 OK (14 crops)
- `GET /api/modeling/robustness/summary` — 200 OK (2 Robust Accepted, 10 Split-Sensitive, 2 Baseline Preferred)
- `GET /api/modeling/robustness/{crop}` — 200 OK
- `GET /api/modeling/robustness/{crop}/folds` — 200 OK
- `GET /api/modeling/robustness/{crop}/comparison` — 200 OK
