# Research Paper Summary: Auditable Agricultural Decision Intelligence

## Abstract
Accurate agricultural yield forecasting and decision intelligence are vital for regional food security, resource allocation, and disaster mitigation. However, traditional machine learning models often suffer from optimistic bias under random train/test splits, black-box opacity, and a disconnect from auditable decision-making workflows. This work presents an end-to-end Agricultural Decision Intelligence Platform built on a 51-year panel dataset (1966–2017) from the International Crops Research Institute for the Semi-Arid Tropics (ICRISAT) covering 311 Indian districts. Using chronological out-of-time validation ($2013-2017$), our pre-season and post-harvest Random Forest pipelines achieve a verified $R^2$ of 0.7866, MAE of 353.01 kg/ha, and MAPE of 18.04%. We integrate multi-window CUSUM temporal monitoring, spatial autocorrelation clustering (Moran's I), Tree SHAP feature attributions, and constrained scenario optimization into a unified Decision Intelligence layer. Crucially, analytical statements are backed by a Statement $\rightarrow$ Model $\rightarrow$ Data provenance DAG and deterministic SHA-256 cryptographic audit certificates.

---

## 1. Introduction
Agricultural decision-making requires synthesizing diverse indicators across historical productivity, soil nutrients, climate dynamics, and regional vulnerability. We formulate an auditable decision-support architecture bridging empirical statistical modeling and policy analysis.

## 2. Problem Definition
Predicting district-level crop yields ($\text{kg/ha}$) and synthesizing multi-hazard operational priorities while avoiding false causal claims and maintaining strict cryptographic reproducibility.

## 3. Dataset
Panel data from ICRISAT across 20 Indian states, comprising 2,469 clean district-year observations with features including synthetic fertilizer consumption (N, P, K), monsoon precipitation indices, and cultivated land area.

## 4. Methodology & ML Architecture
- Pre-season exogenous models and full-season post-harvest regressors.
- Unsupervised Isolation Forest for anomaly identification.
- K-Means ($k=4$) spatial clustering for agro-ecological grouping.

## 5. Model Validation & Reliability
- Chronological Out-of-Time split ($\text{Train} \le 2012$, $\text{Test} = 2013-2017$).
- Decile spread calibration (81.3% coverage across $P_{10} - P_{90}$ bands).
- PSI feature drift tracking.

## 6. Geospatial & Temporal Intelligence
- Local Moran's I spatial autocorrelation for spatial hotspot/coldspot detection.
- Multi-window moving averages (3-yr and 5-yr) and two-sided CUSUM change-point tracking across 311 districts.

## 7. Explainability & Scenario Simulation
- Tree SHAP local feature attribution and sensitivity sweeps.
- Constrained SLSQP optimization with physical non-negativity bounds.

## 8. Decision Intelligence & Auditability
- Multi-module evidence synthesis ranking operational priorities (`HIGH`, `MODERATE`, `LOW`).
- Deterministic SHA-256 decision audit records (`DEC-xxxxxxxx`) and provenance DAGs.

## 9. Results & Limitations
The platform demonstrates strong predictive and diagnostic utility while strictly enforcing non-causal interpretation guardrails across all user-facing interfaces.
