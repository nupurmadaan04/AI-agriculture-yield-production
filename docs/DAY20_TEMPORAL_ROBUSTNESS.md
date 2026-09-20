# Day 20 — Temporal Robustness & Walk-Forward Model Validation

## 1. Executive Summary

Day 20 establishes a rigorous, leak-free **walk-forward cross-validation protocol** across 14 model-ready agricultural commodities in the AI Agriculture Intelligence Platform. 

While Day 19 evaluated models against a single chronological holdout split (Train: 2011–2015, Test: 2016–2017), Day 20 tests whether ML models demonstrate genuine **temporal robustness** across multiple expanding historical forecasting origins (2014, 2015, 2016, and 2017).

---

## 2. Walk-Forward Validation Methodology

To replicate real-world operational pre-season forecasting, an **expanding-window walk-forward scheme** is implemented:

```
2010 (Seed) | 2011  2012  2013 | 2014 (Test Origin 1)
────────────┴──────────────────┴─────────────────────►
                  TRAIN (N≈926)        TEST (N≈304)

2010 (Seed) | 2011  2012  2013  2014 | 2015 (Test Origin 2)
────────────┴────────────────────────┴─────────────────────►
                  TRAIN (N≈1,230)      TEST (N≈310)

2010 (Seed) | 2011  2012  2013  2014  2015 | 2016 (Test Origin 3)
────────────┴──────────────────────────────┴─────────────────────►
                  TRAIN (N≈1,540)            TEST (N≈311)

2010 (Seed) | 2011  2012  2013  2014  2015  2016 | 2017 (Test Origin 4)
────────────┴────────────────────────────────────┴─────────────────────►
                  TRAIN (N≈1,851)                  TEST (N≈307)
```

### Strict Temporal Boundaries
1. **Zero Future Contamination**: Training slices only contain data strictly preceding the forecast year ($T_{\text{train}} < T_{\text{test}}$).
2. **Dynamic Preprocessing**: `CropFeaturePipeline` fits categorical encoders, state maps, and historical district/crop yield averages strictly on the training partition of each individual fold.
3. **No Target or Post-Harvest Leakage**: All simultaneous post-harvest variables (`production_tonnes`, `spatial_cluster_id`) are excluded.

---

## 3. Walk-Forward Robustness Leaderboard

| Commodity | Candidate Algorithm | Folds | Walk-Forward MAE (kg/ha) | Baseline MAE (kg/ha) | Mean MAE Gain (%) | Median MAE Gain (%) | Std Dev MAE (kg/ha) | Fold Win Rate (%) | Robustness Score | Day 20 Validated Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Chickpea** | GradientBoosting | 4 | 257.58 | 263.88 | **+2.32%** | **+6.11%** | 44.97 | **75.0% (3/4)** | **70.3** | `ROBUST_ACCEPTED` |
| **Oilseeds** | RandomForest | 4 | 465.23 | 521.40 | **+12.79%** | **+5.18%** | 176.91 | **75.0% (3/4)** | **80.5** | `ROBUST_ACCEPTED` |
| **Kharif Sorghum** | RandomForest | 4 | 290.38 | 293.85 | +1.30% | +0.79% | 31.58 | 50.0% (2/4) | 59.9 | `SPLIT_SENSITIVE` |
| **Sugarcane** | RandomForest | 4 | 1410.85 | 1431.02 | +1.17% | +0.18% | 206.16 | 50.0% (2/4) | 58.7 | `SPLIT_SENSITIVE` |
| **Minor Pulses** | GradientBoosting | 4 | 329.96 | 315.00 | -5.76% | -7.69% | 53.42 | 50.0% (2/4) | 56.0 | `SPLIT_SENSITIVE` |
| **Wheat** | RandomForest | 4 | 418.45 | 381.65 | -13.21% | -16.04% | 83.86 | 50.0% (2/4) | 55.0 | `SPLIT_SENSITIVE` |
| **Groundnut** | RandomForest | 4 | 311.92 | 289.68 | -7.87% | -6.62% | 21.01 | 25.0% (1/4) | 48.3 | `SPLIT_SENSITIVE` |
| **Sesamum** | RandomForest | 4 | 139.00 | 137.38 | -1.19% | -2.22% | 13.18 | 25.0% (1/4) | 47.6 | `SPLIT_SENSITIVE` |
| **Maize** | RandomForest | 4 | 653.67 | 639.91 | -2.63% | -3.05 | 76.39 | 25.0% (1/4) | 47.1 | `SPLIT_SENSITIVE` |
| **Sorghum** | RandomForest | 4 | 274.04 | 264.16 | -3.50% | -1.45% | 31.86 | 25.0% (1/4) | 47.1 | `SPLIT_SENSITIVE` |
| **Rapeseed & Mustard** | RandomForest | 4 | 199.70 | 191.95 | -5.62% | -4.42 | 24.88 | 25.0% (1/4) | 46.9 | `SPLIT_SENSITIVE` |
| **Rice (Multi-Crop)** | RandomForest | 4 | 336.35 | 310.28 | -8.39% | -10.76% | 41.14 | 25.0% (1/4) | 46.9 | `SPLIT_SENSITIVE` |
| **Pearl Millet** | RandomForest | 4 | 274.83 | 259.23 | -6.04% | -6.07% | 6.78 | 0.0% (0/4) | 39.4 | `BASELINE_PREFERRED` |
| **Pigeonpea** | RandomForest | 4 | 293.12 | 276.41 | -5.68% | -3.52 | 36.46 | 0.0% (0/4) | 36.9 | `BASELINE_PREFERRED` |

---

## 4. Key Scientific Insights

1. **Split-Sensitivity is Pervasive**:
   - Single-split evaluations (such as Day 19) can create an illusion of model generalizability due to favorable test-year weather or localized yield distribution alignments.
   - For example, Sesamum and Maize won on the 2016–2017 holdout split, but failed in 3 out of 4 historical walk-forward origins.
2. **True Robustness Emerges in High-Signal Regimes**:
   - **Chickpea** and **Oilseeds** demonstrate consistent, multi-origin superiority ($\ge 75\%$ win rate, positive mean and median MAE gains across diverse climate years).
3. **Zero Compromise on Integrity**:
   - In accordance with rigorous scientific protocol, split-sensitive models are explicitly downgraded rather than artificially protected.
