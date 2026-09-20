# Scientific Guardrails & Analytical Limitations

> **Critical Notice**:
> The system provides **analytical decision support** and does **not** establish causal relationships, guarantee agricultural outcomes, or replace professional agronomist field evaluations.

---

## 1. Explicit Domain & Methodological Limitations

1. **Historical Dataset Boundaries**: Trained strictly on historical ICRISAT district panel data (1966–2017). Unprecedented future climatic shocks outside the historical training envelope cannot be reliably extrapolated.
2. **Geographic Coverage Limits**: Calibrated across 20 major rice-producing Indian States (311 districts). The system cannot be directly applied to uncalibrated overseas or non-rice agro-ecological zones.
3. **Temporal Resolution Constraints**: Operates at the annual district-year level. Sub-seasonal, weekly, or daily weather extremes (e.g., sudden flash floods, 3-day heatwaves during flowering) are aggregated into seasonal indices.
4. **Feature-Space Boundaries**: Covers primary inputs (NPK fertilizers, precipitation, cultivated area). Micro-nutrients, pest infestations, localized seed variety genetics, and irrigation groundwater levels are unobserved in the panel.
5. **Forecasting Uncertainty**: Yield forecasts represent expected statistical point estimates with empirical prediction spreads ($P_{10} - P_{90}$), not deterministic guarantees.
6. **Model Generalization**: Historical out-of-time accuracy ($R^2 = 0.7866$) reflects unobserved seasons within the panel period; it does not guarantee invariance under non-stationary future climate regimes.
7. **Prediction Spread Interpretation**: Interval bounds denote ensemble tree variance, not formal Bayesian posterior probabilities.
8. **Scenario Simulation Bounds**: Scenario simulations represent *hypothetical model responses* under modified input assumptions, not guaranteed real-world outcomes.
9. **Explainable AI (XAI) Attribution**: SHAP values denote *local statistical feature contributions* to the mathematical model, **not biological or agronomic causality**.
10. **Early-Warning Signal Scope**: Early warning alerts identify *statistical deviation patterns* from historical moving averages, not definitive declarations of crop failure.
11. **Spatial Abstraction Limits**: District-level spatial aggregation averages across heterogeneous micro-climates, soil varieties, and farm management practices within each district.
12. **Non-Causal Interpretation Requirement**: All associations, feature importance scores, and optimization recommendations denote empirical correlations within the trained distribution. Prohibited phrases (`caused by`, `will increase`, `will reduce`, `guarantees`, `proves`, `definitely`) are strictly barred.
13. **Decision-Support Role**: The platform produces structured analytical decision briefs to assist policy makers and agronomists, not automated autonomous farm management instructions.
14. **Data Availability Lag**: Operates on post-season published reporting cycles rather than live real-time IoT or satellite telemetry streams.
