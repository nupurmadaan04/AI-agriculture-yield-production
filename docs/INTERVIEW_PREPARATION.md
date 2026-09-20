# Technical Interview Preparation & Deep Dive

## 1. What problem does the project solve?
It solves the disconnect between raw agricultural data / isolated ML forecasts and actionable, auditable decision-making. Agronomists and policymakers need unified risk signals (monitoring, spatial context, model reliability, scenario simulation, explainability) backed by a transparent audit trail rather than black-box point predictions.

## 2. Why did you choose ICRISAT?
The ICRISAT (International Crops Research Institute for the Semi-Arid Tropics) district-level database is the gold-standard longitudinal agricultural panel in India, spanning 1966 to 2017 with standardized district boundaries, production records, input usages, and climate metrics across 51 continuous seasons.

## 3. How was the dataset processed?
We resolved administrative boundary shifts, imputed isolated missing values via localized district-level interpolation, constructed rolling lagged features ($t-1, t-2$, 3-year rolling means), normalized cultivated area using log transforms, and filtered out non-physical yield anomalies ($<100$ or $>6,000\text{ kg/ha}$).

## 4. Why chronological validation?
Agriculture exhibits non-stationary temporal trends (climate shifts, fertilizer adoption, technological improvements). Random train/test splits cause massive data leakage by training on future seasons to predict past seasons. Chronological splitting (Train: $\le 2012$, Test: $2013-2017$) accurately mirrors real-world forecasting conditions.

## 5. Why not random train/test splitting?
Random splitting yields artificially inflated metrics (e.g. $R^2 > 0.95$) due to temporal auto-correlation between consecutive years in the same district, resulting in catastrophic failure when deployed to truly unobserved future crop years.

## 6. What does $R^2 = 0.7866$ mean?
Under chronological out-of-time evaluation ($2013-2017$), our Random Forest model explains 78.66% of the variance in unseen district rice yields across India, demonstrating strong generalization without overfitting.

## 7. What does $\text{MAE} = 353.01\text{ kg/ha}$ mean?
On average, the model's yield prediction deviates by $353.01\text{ kg/ha}$ from the actual recorded harvest yield across all evaluated district seasons.

## 8. What are the limitations of the model?
The model operates on historical annual panel data without intra-seasonal daily weather telemetry, micro-nutrient details, or satellite imagery. It represents statistical correlations within the historical distribution and cannot guarantee accuracy under unprecedented future climate tipping points.

## 9. How does anomaly detection work?
We use an Isolation Forest trained on normalized yield residuals, fertilizer efficiency, and rainfall deviations scaled via `RobustScaler`. It flags multivariate anomalies that deviate from typical regional agronomic relationships.

## 10. How does geospatial clustering work?
We use K-Means clustering ($k=4$) on historical yield stability, precipitation, and input intensity to group the 311 districts into distinct agro-ecological operational zones, combined with Local Moran's I to identify spatial yield hotspots and coldspots.

## 11. How does early warning work?
We track multi-window moving averages (3-year and 5-year spans) combined with two-sided CUSUM drift detection to capture sustained negative yield departures and compute a composite hazard index categorized into `WATCH`, `WARNING`, and `EMERGENCY`.

## 12. How did you avoid fake probabilities?
We strictly avoid synthetic or uncalibrated softmax percentages. Uncertainty is represented through empirical quantile spreads ($P_{10} - P_{90}$) measured from the individual decision tree variance across the Random Forest ensemble and validated across 10 deciles.

## 13. How does explainability work?
We implement Tree SHAP (SHapley Additive exPlanations) to compute exact feature attributions for any local prediction, showing how each feature shifted the prediction relative to the historical baseline.

## 14. Why is XAI not causal?
SHAP attributions quantify the mathematical sensitivity of the fitted regression function to changes in input values within the training distribution. They do not represent biological mechanisms or prove that adjusting an input will cause a guaranteed yield change in the field.

## 15. How does scenario simulation differ from forecasting?
Forecasting estimates expected yield given observed or exogenous baseline conditions. Scenario simulation computes *hypothetical model responses* under user-defined perturbations (e.g., $+20\%$ fertilizer, $-15\%$ rainfall) to explore sensitivity and policy options.

## 16. How does multi-objective optimization work?
Using SciPy's bounded SLSQP optimizer, we solve for optimal input adjustments that maximize predicted yield while minimizing input costs and environmental overuse, subject to strict non-negativity and domain bounds.

## 17. How does decision intelligence combine evidence?
It aggregates signals across 10 analytical domains (forecast, uncertainty, anomaly, spatial risk, CUSUM drift, alerts, backtest, XAI, scenarios, reliability), ranks operational priorities, evaluates scenario option robustness, and produces a structured Decision Brief.

## 18. How does provenance work?
Every analytical conclusion in the Decision Brief is mapped via a Directed Acyclic Graph (DAG) linking the finding to the specific model version, preprocessing transformation, and raw ICRISAT panel record.

## 19. How does the audit certificate work?
We compute a canonical JSON serialization of the decision inputs, model version, and generated findings, and generate a deterministic SHA-256 hash (`DEC-xxxxxxxx`). Re-running the identical context reproduces the exact same certificate.

## 20. How would you deploy this system?
The platform is packaged into containerized multi-stage Docker images (Python 3.11 FastAPI backend and React 18 / Nginx frontend) orchestrated via Docker Compose with deep readiness probes (`/ready`) and structured JSON logging.

## 21. How would you improve it with more data?
Integrating Sentinel-2 / Landsat NDVI vegetation indices, gridded daily meteorological telemetry (ERA5 / IMD), soil organic carbon maps, and groundwater sensor data would substantially increase temporal resolution and precision.

## 22. What would you change for real-time agricultural deployment?
Transition from post-season panel updates to streaming in-season weather APIs, automated daily satellite tile ingestion, and live telemetry webhooks connected to agricultural extension advisory systems.
