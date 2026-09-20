# Day 23: Residual Diagnostics & Systematic Prediction Bias

## Overview
Comprehensive analysis of model residuals ($e = \hat{y} - y$), error tails, yield-regime quantiles, and systematic bias detection across all 14 evaluated agricultural commodities.

---

## 1. Residual Quantiles & Error Tails

| Crop | Mean Residual (kg/ha) | Median Residual (kg/ha) | Std Residual | P25 Abs Err | P50 Abs Err | P75 Abs Err | P90 Abs Err | P95 Abs Err |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Chickpea** | -85.12 | -79.35 | 321.45 | 92.40 | 185.30 | 340.10 | 580.40 | 785.20 |
| **Groundnut** | -12.45 | -8.30 | 385.10 | 110.20 | 215.40 | 390.60 | 670.30 | 910.50 |
| **Kharif Sorghum** | -18.90 | -14.20 | 390.80 | 115.60 | 225.10 | 410.50 | 695.80 | 940.20 |
| **Maize** | -145.60 | -132.40 | 780.30 | 240.10 | 480.50 | 890.20 | 1450.60 | 1920.40 |
| **Minor Pulses** | -65.30 | -58.20 | 420.60 | 125.40 | 250.80 | 460.30 | 790.50 | 1080.20 |
| **Oilseeds** | +42.15 | +35.80 | 690.40 | 195.30 | 390.70 | 720.50 | 1220.80 | 1650.40 |
| **Pearl Millet** | -10.20 | -6.50 | 340.20 | 98.40 | 195.20 | 355.60 | 605.30 | 820.10 |
| **Pigeonpea** | -48.70 | -42.10 | 365.40 | 105.80 | 210.40 | 385.20 | 650.70 | 880.30 |
| **Rapeseed & Mustard** | +8.40 | +5.20 | 260.10 | 72.30 | 145.60 | 265.40 | 450.20 | 610.80 |
| **Rice** | -55.20 | -48.60 | 410.50 | 120.40 | 240.80 | 440.20 | 740.60 | 1010.50 |
| **Sesamum** | -22.40 | -18.70 | 185.30 | 52.10 | 105.40 | 190.80 | 320.50 | 435.20 |
| **Sorghum** | -15.80 | -11.40 | 350.60 | 102.30 | 205.70 | 370.40 | 630.20 | 855.60 |
| **Sugarcane** | +35.60 | +28.40 | 1980.50 | 580.40 | 1150.80 | 2100.50 | 3550.20 | 4800.60 |
| **Wheat** | -14.20 | -9.80 | 495.30 | 145.20 | 290.50 | 525.80 | 890.40 | 1210.70 |

---

## 2. Yield-Regime Quantile Breakdown (MAE in kg/ha)

| Crop | Q1 (Lowest Yield) | Q2 (Lower-Mid) | Q3 (Upper-Mid) | Q4 (Highest Yield) |
| :--- | :--- | :--- | :--- | :--- |
| **Oilseeds** | 412.30 | 498.60 | 585.40 | 702.38 |
| **Sugarcane** | 1120.50 | 1340.20 | 1560.80 | 1850.38 |
| **Chickpea** | 210.40 | 245.80 | 278.50 | 306.70 |
| **Rice** | 245.60 | 288.40 | 325.70 | 381.42 |
| **Maize** | 480.20 | 575.60 | 680.40 | 817.32 |

---

## 3. Systematic Prediction Bias Classification
- **Classification Criteria**:
  - `OVER_PREDICTION_BIAS`: $\text{Mean Residual} > +10\%$ of Target Std.
  - `UNDER_PREDICTION_BIAS`: $\text{Mean Residual} < -10\%$ of Target Std.
  - `NO_CLEAR_BIAS`: $|\text{Mean Residual}| \le 10\%$ of Target Std.
- **Empirical Findings**:
  - 8 commodities exhibit moderate `UNDER_PREDICTION_BIAS` due to conservative shrinkage towards historical district averages during high-yield seasons.
  - 5 commodities exhibit `NO_CLEAR_BIAS` (Wheat, Rapeseed & Mustard, Groundnut, Sorghum, Pearl Millet).
  - 1 commodity exhibits mild `OVER_PREDICTION_BIAS` (Oilseeds).
