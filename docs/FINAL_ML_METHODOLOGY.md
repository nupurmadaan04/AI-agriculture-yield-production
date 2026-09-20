# Machine Learning Methodology & Model Specifications

## 1. Overview & Problem Formulation
The platform models district-level rice yield ($\text{kg/ha}$) across two distinct operational regimes:
1. **Pre-Season Decision Planning**: Forecasting yield before sowing using strictly exogenous historical inputs, pre-monsoon precipitation indices, and regional baseline trends.
2. **Post-Harvest Diagnostic Assessment**: Evaluating yield performance using full-season observed inputs to benchmark district productivity and detect operational anomalies.

---

## 2. Model Architectures & Pipelines

### A. Full-Season Post-Harvest Regressor (`Models/rf_pipeline.pkl`)
- **Algorithm**: Random Forest Regressor (`n_estimators=100`, `max_depth=12`, `min_samples_leaf=3`, `random_state=42`).
- **Feature Pipeline**: `StandardScaler` fitted with canonical feature headers, followed by ensemble decision tree regression.
- **Uncertainty Estimation**: Non-parametric quantile bounds ($P_{10}$ and $P_{90}$) derived directly from individual tree prediction distributions.

### B. Pre-Season Advanced Exogenous Forecaster (`Models/pre_season_exogenous_pipeline.pkl`)
- **Algorithm**: Random Forest Regressor trained without concurrent monsoon rainfall or end-of-season inputs.
- **Input Features**: Lagged yields ($t-1, t-2$), 3-year rolling mean yield, pre-monsoon rainfall, historical fertilizer density, and regional spatial cluster identifiers.

### C. Agricultural Anomaly Detector (`Models/agricultural_anomaly_pipeline.pkl`)
- **Algorithm**: Isolation Forest (`contamination=0.05`, `n_estimators=100`).
- **Input Features**: Normalized yield residuals, fertilizer application efficiency, and precipitation deviation indices scaled via `RobustScaler`.
- **Output**: Discrete anomaly flag $\in \{-1, 1\}$ and continuous anomaly anomaly score $\in [-1.0, 1.0]$.

### D. Spatial Agro-Climatic Clusterer (`src/spatial_clustering.py`)
- **Algorithm**: K-Means Clustering ($k=4$).
- **Features**: Mean annual precipitation, historical yield stability, and fertilizer consumption intensity across 311 districts.

---

## 3. Chronological Out-of-Time Validation Results

The primary model was validated on an unobserved historical test split ($2013-2017$) preserving realistic forward-looking forecasting conditions:

| Metric | Measured Value | Interpretation |
| :--- | :--- | :--- |
| **Coefficient of Determination ($R^2$)** | **0.7866** | Explains 78.66% of yield variance across unseen test seasons |
| **Mean Absolute Error (MAE)** | **353.01 kg/ha** | Average absolute deviation from actual observed district yields |
| **Root Mean Squared Error (RMSE)** | **513.11 kg/ha** | Penalized deviation reflecting sensitivity to extreme weather years |
| **Mean Absolute Percentage Error (MAPE)** | **18.04%** | Normalized error relative to baseline district yield levels |

---

## 4. Scientific Guardrails & Generalization Boundaries
- **Historical Validity**: Metrics represent evaluation over historical ICRISAT panel partitions. They do **not** guarantee universal accuracy under unprecedented future climate shifts.
- **Non-Causal Estimations**: Model predictions denote statistical associations under historical constraints and must **never** be interpreted as causal guarantees or biological certainty.
