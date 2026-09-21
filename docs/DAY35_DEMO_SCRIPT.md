# DAY 35 — Production Demonstration Script & Golden Journey Walkthrough

## Target Duration: 5–7 Minutes
**Audience**: Technical evaluators, agricultural analysts, policy decision-makers.  
**Core Objective**: Demonstrate how the platform delivers transparent, governed, and mathematically bounded pre-season forecasts and multi-scenario decision support without ungrounded AI claims.

---

## Script Flow & Minute-by-Minute Guide

### 0:00 – 0:30: Platform Overview (Homepage)
- **URL**: `http://localhost/`
- **Action**: Open homepage. Highlight the verified platform scale in the top metrics ribbon:
  - 71,601 unified agricultural records across 20 Indian states and 311 districts (1966–2017).
  - Highlight the two analytical modes: **Pre-Season Forecasting** (out-of-sample prediction) vs. **Post-Harvest Verification** (deterministic agronomic identity check: $Yield = (Production / Area) \times 1,000$).
- **Key Talking Point**: *"This platform does not treat AI as an autonomous decision-maker. It enforces a strict certification guard where machine learning is retained only when it provably outperforms statistical baselines under expanding walk-forward validation."*

---

### 0:30 – 1:30: Pre-Season Forecasting & Strategy Selection (Prediction Explorer)
- **URL**: `http://localhost/prediction-explorer`
- **Demo Golden Case 1 (Governed ML)**:
  - **Crop**: `Oilseeds`
  - **State**: `Madhya Pradesh`
  - **District**: `Indore`
  - **Year**: `2017`
- **Action**: Click **Execute Governed Forecast**.
- **Observation**:
  - Yield prediction renders with explicit badge: `[PREDICTED]`.
  - Strategy identified as: `RandomForestRegressor` with status `[CERTIFIED_PRODUCTION]`.
  - Displayed along with historical context from the canonical ICRISAT panel.

---

### 1:30 – 2:30: Strategy Transparency: ML vs. Baseline Comparison
- **Demo Golden Case 2 (Statistical Baseline)**:
  - **Crop**: `Rice`
  - **State**: `Punjab`
  - **District**: `Ludhiana`
  - **Year**: `2017`
- **Action**: Switch crop to `Rice` and click **Execute Governed Forecast**.
- **Observation**:
  - The platform transparently switches strategy to `Historical District Mean` with status `[BASELINE_PRODUCTION]`.
  - No machine learning model is fabricated for this crop.
- **Key Talking Point**: *"For Rice in this region, historical district mean persistence demonstrated superior temporal stability and lower variance during the walk-forward validation tournament. The platform transparently defaults to the baseline rather than deploying an unverified black-box model."*

---

### 2:30 – 3:30: Temporal Validation & Empirical Uncertainty
- **Action**: Return to `Oilseeds` (Indore, MP) and inspect the **Uncertainty & Validation** cards.
- **Observation**:
  - Validation metrics displayed: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) across 2014–2017 expanding folds.
  - Uncertainty displayed: **Empirical P10–P90 Ensemble Interval** (e.g. 820 kg/ha to 1,040 kg/ha).
  - Explicit disclaimer: *"Derived empirically from decision-tree dispersion across 150 estimators; this is not a formal frequentist confidence interval."*
- **Key Talking Point**: *"We never present a single point prediction as a certainty. The empirical interval gives operational decision-makers visibility into tree-level estimator variance."*

---

### 3:30 – 4:30: Attribution & Continuous Monitoring
- **Action**: Scroll to **Feature Attribution** and **Monitoring Health**.
- **Observation**:
  - Attribution displayed as: **Model Attribution (Marginal Reference Perturbation Attribution)**.
  - Primary driver: Prior Year Yield Lag ($t-1$) followed by 3-Year Rolling Mean and Cultivated Area.
  - Non-causal notice: *"Indicates mathematical sensitivity in the trained feature space; does not represent biological causality."*
  - Monitoring card displays Population Stability Index (PSI) and Covariate Drift status: `HEALTHY (PSI < 0.10)`.

---

### 4:40 – 5:30: Multi-Scenario Decision Workspace
- **URL**: `http://localhost/decision-workspace`
- **Action**: Select `Oilseeds`, `Madhya Pradesh`, `Indore`, `2017` and click **Analyze Decision Workspace**.
- **Observation**:
  - The side-by-side matrix renders:
    1. **Baseline Forecast**: Governed pre-season estimate (`[PREDICTED]`).
    2. **Severe Drought Shock (-25% Yield Lag)**: Simulated scenario (`[SCENARIO]`).
    3. **Monsoon Abundance (+15% Yield Lag)**: Simulated scenario (`[SCENARIO]`).
    4. **Area Expansion (+20% Area)**: Simulated scenario (`[SCENARIO]`).
  - Delta yield and percentage changes are clearly presented without arbitrary ranking.
- **Key Talking Point**: *"The Decision Workspace allows analysts to stress-test hypothetical climate and land-use shifts against the certified baseline without conflating simulations with forecasts."*

---

### 5:30 – 6:30: Cryptographic Provenance & Audit Trail
- **Action**: Expand the **Provenance & Audit Trail** accordion at the bottom of the page.
- **Observation**:
  - Immutable `request_id` (e.g., `REQ-20260921-...`).
  - Cryptographic digital signature: `SHA256:7f4a...` hashing the request parameters, dataset version, and model weights.
  - Timestamp and execution duration.
- **Key Talking Point**: *"Every forecast and decision brief generated by this platform produces an immutable audit record stored in our operational telemetry and prediction audit logs."*

---

### 6:30 – 7:00: Platform Limitations & Human Decision-Maker Role
- **Action**: Review the **Platform Limitations** box:
  1. Forecasts reflect pre-planting information; in-season weather events require updated runs.
  2. Data is grounded in district-level aggregates (ICRISAT), not micro-plot sensors.
  3. Scenarios are mathematical perturbations, not agronomic guarantees.
  4. Final policy decisions remain strictly with human agronomic experts.
- **Conclusion**: *"The platform provides defensible, audited, and transparent evidence to assist human leaders in making informed agricultural policy decisions."*

---

## Demo Golden Cases Reference Table

| Case | Commodity | State | District | Year | Strategy Applied | Certification Status | Expected Uncertainty |
|---|---|---|---|---|---|---|---|
| **Case 1** | Oilseeds | Madhya Pradesh | Indore | 2017 | RandomForestRegressor | `PRODUCTION_READY` | Empirical P10–P90 available |
| **Case 2** | Sugarcane | Uttar Pradesh | Meerut | 2017 | GradientBoostingRegressor | `CONDITIONAL_PRODUCTION` | Empirical P10–P90 available |
| **Case 3** | Rice | Punjab | Ludhiana | 2017 | Historical District Mean | `BASELINE_PRODUCTION` | Explicitly marked unavailable |
| **Case 4** | Wheat | Haryana | Karnal | 2017 | Historical District Mean | `BASELINE_PRODUCTION` | Explicitly marked unavailable |
