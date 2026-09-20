# Multi-Crop Model Compatibility & Scope Specification

## Executive Summary
This document specifies the exact model boundaries, compatibility status, target variables, and scientific guardrails for all registered machine learning pipelines in the **AI Agriculture Intelligence Platform** following the Day 17 Multi-Crop Data Foundation upgrade.

---

## 1. Registered Model Lineage & Scope Matrix

| Model Identifier | Target Variable | Training Dataset | Crop Scope | Compatibility Status | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `forecasting_pipeline.pkl` | `RICE YIELD (Kg per ha)` | ICRISAT Rice Panel (2010–2017) | **Rice Only** | `RICE_ONLY` | Fitted on Rice-specific feature weights; cannot be applied to other crops without retraining. |
| `pre_season_rf_pipeline.pkl` | `RICE YIELD (Kg per ha)` | ICRISAT Rice Panel (2010–2017) | **Rice Only** | `RICE_ONLY` | Pre-season baseline calibrated exclusively for Rice cropping cycles. |
| `pre_season_exogenous_pipeline.pkl` | `RICE YIELD (Kg per ha)` | ICRISAT Exogenous Panel (2010–2017) | **Rice Only** | `RICE_ONLY` | Features include `rice_area_share`, `wheat_area`, `rice_yield_lag1`, and `rice_yield_roll3`. |
| `agricultural_anomaly_pipeline.pkl` | Multivariate Isolation Forest | ICRISAT Panel (2010–2017) | **Rice Only** | `RICE_ONLY` | Contamination thresholds and decision boundaries tuned for Rice yields. |
| `spatial_cluster_pipeline.pkl` | K-Means (k=4) Geographic Clusters | ICRISAT Spatial & Rainfall Data | **All Crops (Spatial)** | `COMPATIBLE` | Agro-climatic spatial feature representations apply universally across Indian agro-ecological zones. |

---

## 2. Scientific Guardrail & Integrity Rule

> [!IMPORTANT]
> **Scientific Integrity Guardrail**: The validated Day 16 performance metrics:
> - $R^2 = 0.7866$
> - $\text{MAE} = 353.01\text{ kg/ha}$
> - $\text{RMSE} = 513.11\text{ kg/ha}$
> - $\text{MAPE} = 18.04\%$
> - Calibration 80% Coverage $= 81.3\%$
> 
> pertain **strictly to the registered Rice forecasting models**.
> They must **never be presented as generic multi-crop accuracy**.

---

## 3. Analytical Separation Policy

1. **Historical Analytics (Supported for All 29 Verified Crops)**:
   - Area, Production, Yield Historical Trends
   - State & District Distributions
   - Year-over-Year Volatility & Time Series
   - Historical Crop Comparisons
2. **Predictive Modeling (Bounded to Registered Scopes)**:
   - Yield Forecasting $\rightarrow$ Rice Only (UI displays transparent empty state banner for other crops)
   - Scenario Simulation $\rightarrow$ Rice Only
   - Tree SHAP Feature Attribution $\rightarrow$ Rice Only
   - Multi-hazard Alerting $\rightarrow$ Rice Only

---

## 4. Multi-Crop Modeling Recommendation & Roadmap

For future multi-crop modeling expansion, the recommended strategy is **Separate Crop-Specific Regressors (Architecture Option D)**:
- Crop biology, physiological yield ceilings, and fertilizer responses vary drastically between cereals (Rice, Wheat, Maize), pulses (Chickpea, Pigeonpea), and oilseeds (Groundnut, Mustard).
- A unified single model introduces confounding crop-interaction biases.
- Crop-specific Random Forest and Gradient Boosted trees preserve biological interpretability and feature attribution fidelity.
