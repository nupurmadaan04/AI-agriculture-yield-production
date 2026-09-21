# Platform Screenshot & Golden Demonstration Package

This directory provides documented visual UI mockups, layout specifications, and a step-by-step screenshot capture checklist for the 10 core views of the platform, along with verified golden test cases.

---

## 1. Screenshot Capture Checklist & Layout Specifications

| ID | View Identifier | Target URL Route | Primary Visual Components | Key Realistic Data Displayed |
|---|---|---|---|---|
| **01** | `01_homepage` | `/` | Hero section, architecture overview, quick-start entry cards, system status indicator | System Online, 29 Crops, 311 Districts, Governed Status |
| **02** | `02_dataset_explorer` | `/data` | Longitudinal panel table, filter controls (crop/state/district), summary statistics cards | 71,601 records, 2010–2017 active window, yield/area distributions |
| **03** | `03_prediction_explorer` | `/predictions` | Searchable historical forecast table, filter by strategy, JSON export button | Historical forecasts, SHA-256 provenance hashes, model versions |
| **04** | `04_forecast` | `/forecast` | Single-district forecast form, strategy badge, point prediction, P10–P90 spread | Oilseeds / Ujjain / 2017: 1,180 kg/ha [1,020–1,340], PRODUCTION_READY |
| **05** | `05_forecast_monitoring` | `/monitoring` | Population Stability Index (PSI) drift chart, 10-bin quantile comparison, alert badges | PSI = 0.082 (Normal / Green), zero severe drift alerts |
| **06** | `06_observability` | `/observability` | Live CPU/RSS telemetry gauges, P50/P95 latency charts, readiness probe status | P95 Latency 38ms, RSS 342 MB, /ready HTTP 200 OK |
| **07** | `07_decision_intelligence`| `/decision-intelligence` | Regional acreage allocation summary, Pareto frontier chart, commodity trade-offs | Multi-crop allocation under water/land constraints |
| **08** | `08_decision_workspace` | `/decision-workspace` | Multi-evidence decision brief, scenario simulation sliders, side-by-side comparison | `[PREDICTED]` vs `[SCENARIO]`, Marginal Reference Perturbation, Tree SHAP |
| **09** | `09_provenance` | `/provenance` | Cryptographic verification modal, SHA-256 hash breakdown, input parameter tree | Immutable digital fingerprint, model version verification |
| **10** | `10_model_governance` | `/governance` | Strategy certification matrix, walk-forward fold win-rates, 3-$\sigma$ fallback rules | 1 Production ML, 1 Conditional ML, 12 Baselines |

---

## 2. Golden Demonstration Test Cases

Use these four verified geographic and commodity combinations for live demonstrations and screenshots:

### Golden Case 1: Oilseeds (Unconstrained Governed ML)
- **Crop**: Total Oilseeds
- **State**: Madhya Pradesh
- **District**: Ujjain
- **Forecast Year**: 2017
- **Expected Governed Strategy**: `PRODUCTION_READY` (RandomForestRegressor)
- **Expected UI Behavior**: Displays active ML badge; predicts ~1,180 kg/ha; empirical P10–P90 uncertainty spread [1,020–1,340 kg/ha]; Tree SHAP feature attributions active.
- **Evidence Available**: 75% walk-forward win-rate, +12.79% mean MAE gain over baseline persistence.

### Golden Case 2: Sugarcane (Conditional ML with 3-$\sigma$ Variance Clipping)
- **Crop**: Sugarcane
- **State**: Uttar Pradesh
- **District**: Muzaffarnagar
- **Forecast Year**: 2017
- **Expected Governed Strategy**: `CONDITIONAL_PRODUCTION` (GradientBoostingRegressor with 3-$\sigma$ district clipping)
- **Expected UI Behavior**: Displays conditional strategy badge; indicates active 3-$\sigma$ safety guard [$\mu \pm 3\sigma$]; predicts ~62,500 kg/ha.
- **Evidence Available**: Bounded Fold 2 drought loss (-5.78% vs -10.19% unclipped); +1.19% aggregate gain over baseline persistence.

### Golden Case 3: Rice (Certified Statistical District Mean Baseline)
- **Crop**: Rice
- **State**: Punjab
- **District**: Ludhiana
- **Forecast Year**: 2017
- **Expected Governed Strategy**: `BASELINE_PRODUCTION` (Historical District Mean)
- **Expected UI Behavior**: Displays certified baseline badge; predicts historical district mean (~3,980 kg/ha); uncertainty shows district standard deviation; feature attribution explicitly displays baseline notice.
- **Evidence Available**: Statistical baseline outperformed ML across temporal folds (ML had negative gain of -8.39%).

### Golden Case 4: Wheat (Certified Statistical District Mean Baseline)
- **Crop**: Wheat
- **State**: Haryana
- **District**: Karnal
- **Forecast Year**: 2017
- **Expected Governed Strategy**: `BASELINE_PRODUCTION` (Historical District Mean)
- **Expected UI Behavior**: Displays certified baseline badge; predicts historical district mean (~4,420 kg/ha); robust to external weather shocks.
- **Evidence Available**: Statistical baseline outperformed ML across temporal folds (ML had negative gain of -18.36%).

---

## 3. High-Fidelity UI Layout Representation

### View 04: Forecast Serving (`/forecast`)
```
+---------------------------------------------------------------------------------------------------------+
| [AI Agriculture Intelligence Platform]        Explore | Monitor | Decide | Governance & Science         |
+---------------------------------------------------------------------------------------------------------+
| PRE-SEASON DISTRICT FORECAST SERVING                                                                     |
|                                                                                                         |
| Crop: [ Total Oilseeds       ▼ ]    State: [ Madhya Pradesh    ▼ ]    District: [ Ujjain         ▼ ]    |
| Year: [ 2017                 ▼ ]    Lead Time: Pre-Season (3-6 Months Before Sowing)                    |
|                                                                                                         |
| [ GENERATE FORECAST ]                                                                                   |
+---------------------------------------------------------------------------------------------------------+
| FORECAST RESULTS & STRATEGY CERTIFICATION                                                               |
|                                                                                                         |
| Status: ● PRODUCTION_READY (Certified Machine Learning)       Latency: 34ms                             |
| Strategy: RandomForestRegressor (150 Trees, Pre-Season Lags V1)                                         |
|                                                                                                         |
| +------------------------------------+  +------------------------------------+                          |
| | PREDICTED YIELD                    |  | EMPIRICAL ENSEMBLE DISPERSION      |                          |
| | 1,180.45 kg/ha                     |  | P10: 1,024.10  |  P90: 1,342.80    |                          |
| | (+12.79% Error Gain over Baseline) |  | (Empirical spread across 150 trees)|                          |
| +------------------------------------+  +------------------------------------+                          |
|                                                                                                         |
| DIGITAL PROVENANCE FINGERPRINT:                                                                         |
| SHA256:7f4a9b2c8e1d3f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a                                 |
| [View Full Audit Record]  [Export JSON]                                                                 |
+---------------------------------------------------------------------------------------------------------+
```

### View 08: Decision Workspace (`/decision-workspace`)
```
+---------------------------------------------------------------------------------------------------------+
| DECISION WORKSPACE: MULTI-EVIDENCE SYNTHESIS & SCENARIO ANALYSIS                                        |
+---------------------------------------------------------------------------------------------------------+
| Commodity: Total Oilseeds | Region: Madhya Pradesh - Ujjain | Harvest Horizon: 2017                    |
|                                                                                                         |
| [EVIDENCE MATRIX: THREE SEPARATE SEMANTIC ENTITIES]                                                     |
|                                                                                                         |
|  Entity Type        Value         Source / Model               Semantic Role                            |
|  ---------------------------------------------------------------------------------------------          |
|  [OBSERVED]         1,145 kg/ha   ICRISAT 2016 Historical      Prior Year Ground Truth Reference         |
|  [PREDICTED]        1,180 kg/ha   RandomForest Regressor       Certified Pre-Season Point Forecast       |
|  [SCENARIO]         1,215 kg/ha   SLSQP Perturbation (+10% Area)Hypothetical What-If Simulation          |
|                                                                                                         |
| NOTE: Scenarios represent mathematical sensitivity within trained manifolds, NOT agronomic certainties.|
+---------------------------------------------------------------------------------------------------------+
```
