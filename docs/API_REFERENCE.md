# REST API Reference & Endpoint Specifications

## 1. Overview
The backend exposes comprehensive REST endpoints built with FastAPI and Pydantic v2.

---

## 2. Endpoint Catalog

### 2.1 Core ML & Prediction
- `GET /api/health` — System and model caching health status.
- `GET /api/states` — List 20 supported agricultural states.
- `GET /api/districts/{state}` — List districts for a specified state.
- `POST /api/predict/post-harvest` — Post-harvest Random Forest prediction.
- `POST /api/predict/pre-season` — True pre-season yield prediction without production leakage.
- `POST /api/predict/advanced` — Exogenous climate and irrigation prediction.
- `POST /api/forecast` — Multi-horizon recursive autoregressive forecast.

### 2.2 Geospatial Intelligence
- `GET /api/geospatial/clusters` — Spatial clustering manifest and zones.
- `GET /api/geospatial/districts` — District coordinates and spatial features.
- `GET /api/geospatial/outliers` — Within-state spatial outlier detection.
- `GET /api/geospatial/similarity` — Regional similarity search across districts.

### 2.3 Model Reliability & Validation
- `GET /api/validation/overview` — Top-level metrics ($R^2$, MAE, RMSE, sMAPE).
- `GET /api/validation/calibration` — 10-bin decile calibration data.
- `GET /api/validation/drift` — Population Stability Index (PSI) drift report.
- `GET /api/validation/quality` — 5-pillar data quality score.
- `GET /api/validation/registry` — Versioned model registry entries.

### 2.4 Scenario Simulation & Decision Intelligence
- `POST /api/scenario/simulate` — Baseline vs hypothetical scenario simulation.
- `POST /api/scenario/compare` — Multi-scenario parallel comparison.
- `POST /api/scenario/sensitivity` — One-at-a-time (OAT) sensitivity sweeps.
- `POST /api/scenario/optimize` — Resource-constrained decision optimization.
- `GET /api/scenario/history` — Audit trail of executed simulations.

### 2.5 Real-Time Temporal Monitoring & Early Warning
- `GET /api/monitoring/overview` — Active alerts, high/critical counts, and status.
- `GET /api/monitoring/timeline` — Trajectory and multi-window rolling stats (3, 5, 8 yr).
- `GET /api/monitoring/alerts` — Prioritized alerts with state/severity filters.
- `GET /api/monitoring/alerts/{alert_id}` — Full evidence certificate and audit chain.
- `GET /api/monitoring/warning-map` — State-level warning tiers.
- `GET /api/monitoring/health` — 5-pillar continuous monitoring health certificate.
- `POST /api/monitoring/backtest` — Chronological step-forward historical backtesting.

### 2.6 Copilot Intelligence
- `POST /api/copilot/query` — Natural language query routing with deterministic tool execution and grounded evidence synthesis.

### 2.7 Explainable AI & Decision Traceability (Day 13)
- `GET /api/explainability/global` — Global feature importance comparing model-native Gini with holdout permutation importance.
- `POST /api/explainability/prediction` — Local prediction attribution deconstruction with positive/negative driver breakdown.
- `POST /api/explainability/sensitivity` — Controlled continuous parameter sweeps ([-10%, +10%]) on registered models.
- `GET /api/explainability/alert/{alert_id}` — Deconstructs an early warning alert into a multi-signal evidence explanation.
- `POST /api/explainability/scenario/{scenario_id}` — Input parameter differential analysis between baseline and scenario.
- `GET /api/explainability/audit/{explanation_id}` — Retrieves an immutable, verifiable explanation decision certificate.
- `GET /api/explainability/methodology` — Mathematical definitions and non-causal integrity boundaries.
- `GET /api/explainability/validation` — 7-point scientific validation check report.

