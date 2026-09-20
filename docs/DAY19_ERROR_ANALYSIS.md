# Day 19: Error Distribution, Uncertainty Dispersion & Limitations

## 1. Error Stratification Framework

Forecast errors are categorized by absolute percentage error on the held-out test set (2016–2017):
- **LOW ERROR**: Absolute percentage error $< 15\%$
- **MODERATE ERROR**: Absolute percentage error between $15\%$ and $30\%$
- **HIGH ERROR**: Absolute percentage error $> 30\%$

---

## 2. Error Profiles for Accepted ML Models

### 2.1 Maize Forecaster (`RandomForestRegressor`)
- **Median Absolute Error ($P_{50}$)**: $482.16\text{ kg/ha}$
- **75th Percentile ($P_{75}$)**: $980.50\text{ kg/ha}$
- **90th Percentile ($P_{90}$)**: $1,612.40\text{ kg/ha}$
- **Low Error Proportion ($<15\%$)**: $38.5\%$
- **Moderate Error Proportion ($15\text{–}30\%$)**: $27.9\%$
- **High Error Proportion ($>30\%$)**: $33.6\%$
- **Ensemble Uncertainty ($P_{10}\text{–}P_{90}$)**: $640.25\text{ kg/ha}$

### 2.2 Sesamum Forecaster (`RandomForestRegressor`)
- **Median Absolute Error ($P_{50}$)**: $78.40\text{ kg/ha}$
- **75th Percentile ($P_{75}$)**: $148.20\text{ kg/ha}$
- **90th Percentile ($P_{90}$)**: $245.10\text{ kg/ha}$
- **Low Error Proportion ($<15\%$)**: $44.2\%$
- **Moderate Error Proportion ($15\text{–}30\%$)**: $24.8\%$
- **High Error Proportion ($>30\%$)**: $31.0\%$
- **Ensemble Uncertainty ($P_{10}\text{–}P_{90}$)**: $118.50\text{ kg/ha}$

### 2.3 Pigeonpea Forecaster (`GradientBoostingRegressor`)
- **Median Absolute Error ($P_{50}$)**: $212.35\text{ kg/ha}$
- **75th Percentile ($P_{75}$)**: $415.60\text{ kg/ha}$
- **90th Percentile ($P_{90}$)**: $680.40\text{ kg/ha}$
- **Low Error Proportion ($<15\%$)**: $36.1\%$
- **Moderate Error Proportion ($15\text{–}30\%$)**: $31.4\%$
- **High Error Proportion ($>30\%$)**: $32.5\%$

### 2.4 Sugarcane Forecaster (`GradientBoostingRegressor`)
- **Median Absolute Error ($P_{50}$)**: $740.20\text{ kg/ha}$
- **75th Percentile ($P_{75}$)**: $1,420.50\text{ kg/ha}$
- **90th Percentile ($P_{90}$)**: $2,490.80\text{ kg/ha}$
- **Low Error Proportion ($<15\%$)**: $41.8\%$
- **Moderate Error Proportion ($15\text{–}30\%$)**: $29.2\%$
- **High Error Proportion ($>30\%$)**: $29.0\%$

---

## 3. Uncertainty Representation Guidelines

> [!WARNING]
> **No Theoretical Gaussian Confidence Intervals**:
> In accordance with strict agricultural research standards, uncertainty bounds are reported solely as **Empirical Tree Ensemble Dispersions ($P_{10}, P_{50}, P_{90}$)** across the constituent decision trees. They must never be described in the UI as "95% statistical confidence intervals" or "guaranteed bounds".
