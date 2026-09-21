# 7. Error Diagnostics & Spatial-Temporal Regime Analysis

### 7.1 Error Quantiles and Distribution Profiles

Residual analysis was conducted across 14 commodities over the multi-origin walk-forward testing window (origins 2014–2017). Residuals are defined signed as $e_{i, t} = \hat{y}_{i, t} - y_{i, t}$, such that positive values indicate over-prediction and negative values indicate under-prediction.

```
+----------------------------------------------------------------------------------------------------+
|                                    TABLE 2: ERROR QUANTILE PROFILES                                |
+---------------------+---------------+---------------+---------------+---------------+--------------+
| Crop Commodity      | 10th Quantile | Median (Q2)   | 90th Quantile | P90 Abs Error | Bias Status  |
|                     | (kg/ha)       | (kg/ha)       | (kg/ha)       | (kg/ha)       |              |
+---------------------+---------------+---------------+---------------+---------------+--------------+
| **Oilseeds**        | -348.2        | +42.1         | +492.6        | 1,145.8       | Over-Pred.   |
| **Sugarcane**       | -1,240.5      | -85.2         | +1,620.0      | 3,745.5       | Unbiased     |
| **Chickpea**        | -312.4        | -40.98        | +285.6        | 569.0         | Under-Pred.  |
| **Kharif Sorghum**  | -380.1        | +15.19        | +410.2        | 749.8         | Unbiased     |
| **Minor Pulses**    | -415.0        | -134.33       | +210.5        | 730.2         | Under-Pred.  |
| **Rice**            | -390.2        | -18.4         | +420.5        | 780.4         | Unbiased     |
| **Wheat**           | -480.0        | +32.1         | +510.0        | 890.2         | Unbiased     |
+---------------------+---------------+---------------+---------------+---------------+--------------+
```

### 7.2 Performance Breakdown Across Climate Regimes

Evaluating models across the 2014–2017 historical window revealed critical differences across agricultural regimes:

1. **Normal / High-Rainfall Regimes (2016–2017)**:
   In benign monsoon years, both machine learning and statistical baselines performed predictably, with errors tightly concentrated within $\pm 10\%$ of actual yields. Feature importances in tree models predominantly weighted historical lag features ($y_{t-1}$ and $\bar{y}_{t-1:t-3}$).
2. **Drought Regime Shocks (2014–2015)**:
   During the severe pan-India drought of 2015, rainfed district yields dropped precipitously (frequently $30\text{--}50\%$ below district historical averages).
   - In unconstrained machine learning models, tree ensembles trained on recent normal years failed to extrapolate downward into extreme negative yield anomalies, producing severe positive residuals ($e_{i, t} \gg 0$).
   - Statistical baselines (Historical District Mean) also over-predicted, but their linear errors were bounded by long-term regional dispersion, avoiding the erratic extrapolation spikes exhibited by unclipped GBDT trees.

### 7.3 District-Level Heterogeneity

Spatial disaggregation indicates that predictive errors are strongly associated with district irrigation infrastructure:
- **Assured Irrigation Zones** (e.g. Punjab, Haryana, Western Uttar Pradesh): Exhibits low coefficient of variation ($\text{CV} < 0.12$) and high baseline predictability ($\text{MAPE} < 10\%$).
- **Rainfed / Semi-Arid Zones** (e.g. Marathwada, Vidarbha, Rayalaseema): Exhibits high inter-annual volatility ($\text{CV} > 0.35$). Large forecast errors in these districts are associated with rainfall failure during early vegetative stages that cannot be captured by pre-season indicators.
