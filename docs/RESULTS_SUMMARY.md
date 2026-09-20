# Research & Platform Results Summary

## 1. Core Platform Summary

| Dimension | Specification / Verified Result |
| :--- | :--- |
| **Dataset** | ICRISAT District-Level Database for Indian Agriculture (1966–2017) |
| **Geographic Scope** | 20 Indian States • 311 Districts • 2,469 Clean Panel Observations |
| **Target Crop** | Rice (`RICE YIELD (Kg per ha)`) |
| **Primary Models** | Random Forest Regressors, Isolation Forest, K-Means Spatial Clustering |
| **Validation Protocol** | Chronological Out-of-Time Split (Train: $\le 2012$, Test: $2013-2017$) |
| **Validation $R^2$** | **0.7866** |
| **Validation MAE** | **353.01 kg/ha** |
| **Validation RMSE** | **513.11 kg/ha** |
| **Validation MAPE** | **18.04%** |
| **Interval Coverage** | **81.3%** empirical coverage across $P_{10} - P_{90}$ prediction deciles |
| **Temporal Monitoring** | 3-yr / 5-yr moving averages, two-sided CUSUM drift, 311 district alert tracking |
| **Geospatial Intelligence** | 4 spatial clusters, Local Moran's I spatial autocorrelation, within-state Z-scores |
| **Explainable AI** | Local Tree SHAP attributions, feature sensitivity curves, `EXP-` audit IDs |
| **Scenario Intelligence** | Bounded input perturbations, constrained SLSQP optimization, `SCN-` audit IDs |
| **Decision Intelligence** | 10-domain evidence fusion, robustness scoring, Statement $\rightarrow$ Data DAG |
| **Auditability** | Deterministic SHA-256 decision certificates (`DEC-xxxxxxxx`), Markdown/HTML exports |
| **Scientific Guardrails** | Strictly non-causal statistical decision support; 14 explicit domain limitations |
