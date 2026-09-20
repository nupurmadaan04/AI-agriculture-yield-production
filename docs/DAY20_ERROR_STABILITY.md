# Day 20 — Error Stability & Residual Distribution Audit

## 1. Residual Variance Across Walk-Forward Folds

A robust model must exhibit stable, well-behaved error distributions across multiple test origins without extreme catastrophic tail errors.

---

## 2. Multi-Fold Error Distributions

| Commodity | Algorithm | Fold 1 MAE (2014) | Fold 2 MAE (2015) | Fold 3 MAE (2016) | Fold 4 MAE (2017) | Mean Residual (kg/ha) | Std Dev Residual (kg/ha) | Error Stability Rating |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Chickpea** | GradientBoosting | 224.51 | 240.67 | 327.91 | 237.04 | -12.45 | 44.97 | **EXCELLENT (Stable)** |
| **Oilseeds** | RandomForest | 321.40 | 415.80 | 720.60 | 403.11 | -28.10 | 176.91 | **GOOD (High Signal)** |
| **Sugarcane** | RandomForest | 1180.20 | 1350.40 | 1720.60 | 1392.20 | +42.30 | 206.16 | **MODERATE (High Inertia)** |
| **Kharif Sorghum** | RandomForest | 275.40 | 260.10 | 340.20 | 285.80 | -8.10 | 31.58 | **STABLE (Marginal ML)** |
| **Minor Pulses** | GradientBoosting | 310.20 | 325.40 | 412.80 | 271.40 | -15.20 | 53.42 | **VARIABLE** |
| **Wheat** | RandomForest | 350.10 | 390.40 | 540.20 | 393.10 | -18.40 | 83.86 | **HIGH VARIANCE** |
| **Maize** | RandomForest | 580.40 | 620.10 | 780.90 | 633.20 | +35.20 | 76.39 | **VARIABLE (Climate Sensitive)** |
| **Sesamum** | RandomForest | 130.20 | 125.40 | 160.80 | 139.60 | -4.20 | 13.18 | **STABLE (Low Gain)** |
| **Groundnut** | RandomForest | 290.40 | 310.20 | 345.80 | 301.30 | -11.00 | 21.01 | **STABLE (Baseline Superior)** |
| **Rapeseed & Mustard** | RandomForest | 180.20 | 195.40 | 240.10 | 183.10 | -7.50 | 24.88 | **STABLE (Baseline Superior)** |
| **Sorghum** | RandomForest | 250.10 | 265.40 | 325.80 | 254.90 | -9.80 | 31.86 | **STABLE (Baseline Superior)** |
| **Rice (Multi-Crop)** | RandomForest | 310.20 | 325.10 | 405.40 | 304.70 | -14.10 | 41.14 | **STABLE (Baseline Superior)** |
| **Pearl Millet** | RandomForest | 270.10 | 268.40 | 285.40 | 275.40 | +2.10 | 6.78 | **ULTRA-STABLE (Baseline Dominates)** |
| **Pigeonpea** | RandomForest | 265.10 | 280.40 | 355.20 | 271.80 | -8.40 | 36.46 | **STABLE (Baseline Superior)** |

---

## 3. Climate Shock Impact (2016 Peak Residuals)

Across almost all commodities, **Fold 3 (Test Year: 2016)** exhibits higher prediction errors. 
This reflects the widespread 2015–2016 Indian drought and monsoon delay, which caused anomalous crop yield disruptions nationwide. 

The walk-forward evaluation accurately captures this real-world operational shock and ensures models are judged on their ability to weather climate volatility.
