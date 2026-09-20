# Day 22: Three-Tier Model Architecture Comparison

## 1. Comparative Architecture Overview
- **Model A (Historical-Only ML)**: Supervised Tree Ensemble (`RandomForestRegressor` / `GradientBoostingRegressor`) using strictly lag-1, lag-2, rolling statistics, and district encodings.
- **Model B (Exogenous ML)**: Supervised Tree Ensemble incorporating Model A plus all 10 pre-season environmental indicators.
- **Model C (Statistical Baseline)**: Historical District Mean / Naive Persistence.

---

## 2. Multi-Fold Performance Comparison (2014–2017)

| Crop | Model A (Hist MAE) | Model B (Exo MAE) | Model C (Base MAE) | Exogenous vs Hist Gain (%) | Exogenous vs Base Gain (%) | Fold Win Rate vs Hist | Day 22 Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Oilseeds** | **548.92** | 584.97 | 616.60 | -6.57% | +5.13% | 25.0% | `NO_MEANINGFUL_GAIN` |
| **Chickpea** | 271.43 | 284.49 | **263.69** | -4.81% | -7.89% | 25.0% | `NO_MEANINGFUL_GAIN` |
| **Kharif Sorghum** | 303.12 | 339.29 | **296.70** | -11.93% | -14.35% | 25.0% | `NO_MEANINGFUL_GAIN` |
| **Minor Pulses** | 347.38 | 379.07 | **345.17** | -9.12% | -9.82% | 50.0% | `NO_MEANINGFUL_GAIN` |
| **Maize** | 653.80 | 665.20 | **638.38** | -1.74% | -4.20% | 0.0% | `NO_MEANINGFUL_GAIN` |
| **Wheat** | 405.01 | 413.07 | **381.65** | -1.99% | -8.23% | 50.0% | `NO_MEANINGFUL_GAIN` |
| **Sugarcane** | **1457.23** | 1536.91 | 1485.70 | -5.47% | -3.45% | 50.0% | `NO_MEANINGFUL_GAIN` |
| **Rice** | 325.25 | 329.25 | **310.28** | -1.23% | -6.11% | 50.0% | `NO_MEANINGFUL_GAIN` |
| **Sesamum** | 139.40 | 144.17 | **137.44** | -3.42% | -4.89% | 25.0% | `NO_MEANINGFUL_GAIN` |
| **Pigeonpea** | 290.37 | 300.93 | **282.35** | -3.64% | -6.58% | 25.0% | `NO_MEANINGFUL_GAIN` |
| **Rapeseed & Mustard** | 199.43 | 221.59 | **193.33** | -11.11% | -14.62% | 25.0% | `NO_MEANINGFUL_GAIN` |
| **Groundnut** | 306.51 | 337.51 | **287.40** | -10.11% | -17.44% | 50.0% | `NO_MEANINGFUL_GAIN` |
| **Sorghum** | 273.38 | 312.34 | **264.02** | -14.25% | -18.30% | 25.0% | `NO_MEANINGFUL_GAIN` |
| **Pearl Millet** | 280.10 | 302.98 | **260.24** | -8.17% | -16.42% | 25.0% | `NO_MEANINGFUL_GAIN` |

---

## 3. Scientific Finding: The "Negative Result" Rule
In accordance with non-negotiable scientific rule #20 and #21:
- We report **`NO_MEANINGFUL_GAIN`** for all 14 crops without fabrication or artificial threshold adjustments.
- Pre-season meteorological data prior to sowing contains insufficient predictive mutual information to supersede historical district yield persistence.
- The Day 21 strategy (Model A for Oilseeds; Baseline for 13 crops) remains the verified production strategy.
