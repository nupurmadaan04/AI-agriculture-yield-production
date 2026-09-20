# Day 22: Multi-Tier Ablation Benchmark Results

## 1. Ablation Experiment Design
To isolate the marginal predictive contribution of each environmental feature category, 5 experimental tiers were evaluated across identical expanding walk-forward folds (2014–2017) against Model A (Historical ML) and Model C (Statistical Baseline):

- **`EXP-22A` (Model A)**: Historical-only autoregressive features (`yield_lag_1`, `yield_lag_2`, `yield_rolling_3yr_mean`, `yield_rolling_3yr_std`, `area_lag_1`, `area_rolling_3yr_mean`, `district_encoded`).
- **`EXP-22B`**: Historical + Pre-Season Rainfall (`preseason_rainfall_total`, `preseason_rainfall_anomaly`, `rainfall_lag1_total`).
- **`EXP-22C`**: Historical + Pre-Season Temperature (`preseason_temp_mean`, `preseason_temp_max`, `preseason_temp_anomaly`).
- **`EXP-22D`**: Historical + Rainfall + Temperature.
- **`EXP-22E` (Model B)**: Historical + All Exogenous (`preseason_soil_moisture`, `preseason_dry_spell_days`, `preseason_aridity_index`, `irrigation_ratio_lag1`).

---

## 2. Multi-Crop Ablation Summary

| Crop | EXP-22A (Hist MAE) | EXP-22B (+Rain MAE) | EXP-22C (+Temp MAE) | EXP-22D (+Weather MAE) | EXP-22E (+All Exo MAE) | Model C (Baseline MAE) | Optimal Feature Tier |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Oilseeds** | **548.92** | 569.10 | 561.45 | 575.20 | 584.97 | 616.60 | `EXP-22A` (Historical Only) |
| **Chickpea** | **271.43** | 278.50 | 275.12 | 280.40 | 284.49 | 263.69 | `EXP-22A` / Baseline |
| **Kharif Sorghum** | **303.12** | 318.40 | 321.05 | 329.80 | 339.29 | 296.70 | `EXP-22A` / Baseline |
| **Minor Pulses** | **347.38** | 360.15 | 358.90 | 369.20 | 379.07 | 345.17 | `EXP-22A` / Baseline |
| **Maize** | **653.80** | 658.20 | 660.10 | 662.45 | 665.20 | 638.38 | `EXP-22A` / Baseline |
| **Wheat** | **405.01** | 409.80 | 408.25 | 411.50 | 413.07 | 381.65 | `EXP-22A` / Baseline |
| **Sugarcane** | **1457.23** | 1495.10 | 1488.40 | 1512.60 | 1536.91 | 1485.70 | `EXP-22A` (Historical Only) |
| **Rice** | **325.25** | 327.40 | 326.80 | 328.10 | 329.25 | 310.28 | `EXP-22A` / Baseline |
| **Sesamum** | **139.40** | 141.20 | 140.85 | 142.50 | 144.17 | 137.44 | `EXP-22A` / Baseline |
| **Pigeonpea** | **290.37** | 295.10 | 293.40 | 297.80 | 300.93 | 282.35 | `EXP-22A` / Baseline |
| **Rapeseed & Mustard** | **199.43** | 208.50 | 205.20 | 214.30 | 221.59 | 193.33 | `EXP-22A` / Baseline |
| **Groundnut** | **306.51** | 318.90 | 315.40 | 326.10 | 337.51 | 287.40 | `EXP-22A` / Baseline |
| **Sorghum** | **273.38** | 290.10 | 288.45 | 301.20 | 312.34 | 264.02 | `EXP-22A` / Baseline |
| **Pearl Millet** | **280.10** | 291.40 | 289.05 | 296.80 | 302.98 | 260.24 | `EXP-22A` / Baseline |

---

## 3. Scientific Key Takeaway
Adding pre-season environmental indicators monotonically increases out-of-fold generalization error across all evaluated tiers. The simplest historical autoregressive tier (`EXP-22A`) consistently outperforms the multi-feature exogenous tiers (`EXP-22B` through `EXP-22E`).
