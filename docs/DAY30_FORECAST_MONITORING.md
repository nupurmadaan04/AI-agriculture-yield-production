# Day 30: Forecast Monitoring, Drift Detection & Outcome Intelligence

## 1. Overview & Architecture

The **Forecast Monitoring & Outcome Intelligence** system represents the final operational and scientific oversight layer of the Agricultural Intelligence & Forecasting Platform. It provides end-to-end operational visibility over certified pre-season forecasting services, empirical prediction distributions, statistical covariate drift, and leak-free post-outcome evaluation.

```
+-------------------------------------------------------------------------------+
|                             OPERATIONAL LIFECYCLE                             |
|                                                                               |
|   Canonical Data (AGRI_PANEL_1.0)                                             |
|        |                                                                      |
|        v                                                                      |
|   Governed Pre-Season Forecast (Prediction Service)                           |
|        |                                                                      |
|        v                                                                      |
|   Explainability & Cryptographic Provenance (SHA-256)                         |
|        |                                                                      |
|        v                                                                      |
|   Prediction Audit Log & Operational Telemetry                                |
|        |                                                                      |
|        v                                                                      |
|   Forecast Monitoring & Statistical Drift Detection (PSI / KS)                |
|        |                                                                      |
|        v                                                                      |
|   Observed Harvest Outcome Publication (Strict Temporal Window)               |
|        |                                                                      |
|        v                                                                      |
|   Post-Outcome Evaluation (Signed Bias, MAE, Regime Decompositions)           |
|        |                                                                      |
|        v                                                                      |
|   Evidence-First Monitoring Alerts & Health Oversight                         |
+-------------------------------------------------------------------------------+
```

---

## 2. Core Functional Modules

### A. Executive System Status
Surfaces real-time monitoring state (`HEALTHY`, `WATCH`, `DRIFT_DETECTED`, `EVALUATION_UNAVAILABLE`), total operational audit events, count of verified outcome folds, and active evidence signals.

### B. Forecast Operations Telemetry
Aggregates live requests from `prediction_audit_log.csv` without synthetic data:
- Total, successful, rejected, and failed requests.
- Success rate percentage.
- Crop invocation breakdown.
- Strategy and certification tier breakdown (`PRODUCTION_READY`, `CONDITIONAL_PRODUCTION`, `BASELINE_PRODUCTION`).
- Daily operational request volume time series.

### C. Prediction Distribution Monitoring
Computes continuous statistical moments across generated predictions:
- Sample Count ($N$), Mean, Median, Standard Deviation ($\sigma$), Minimum, Maximum, and Interquartile/Decile spreads ($P10, P25, P75, P90$).
- Compares active prediction moments against historical reference distributions from `agricultural_panel.csv` (1966–2017).
- Automatically highlights distribution shifts when relative mean deviations exceed certified thresholds.

### D. Multi-Aspect Drift Monitoring
Evaluates Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) statistics across continuous feature distributions:
- Reference Window: 2010–2015 baseline.
- Evaluation Window: 2016–2017 out-of-time walk-forward test set.
- Threshold Guidelines:
  - $\text{PSI} < 0.10$: Stable / No Drift.
  - $0.10 \le \text{PSI} < 0.25$: Moderate Drift / Watch.
  - $\text{PSI} \ge 0.25$: Significant Covariate Shift.
- Monitors geographic and crop portfolio coverage stability (14 crops, 311 districts, 19 states).

---

## 3. Data Semantics & Classifications

Every metric surfaced in Day 30 is tagged with an unambiguous semantic label:

| Category | Semantic Label | Example Metric |
| :--- | :--- | :--- |
| Runtime Prediction | `PREDICTED` | Pre-season estimated yield (kg/ha) |
| Harvest Observations | `OBSERVED` | Actual observed yield from panel data |
| Post-Outcome Error | `POST_OUTCOME_EVALUATION` | Signed error ($e = \hat{y} - y$), Absolute error ($|e|$) |
| Covariate Shift | `MONITORING` | Population Stability Index (PSI), KS statistic |
| Operational Events | `MONITORING` | Request counts, success rate %, rejection rates |
| Cryptographic Hash | `PROVENANCE` | Model artifact SHA-256, dataset fingerprint |
| Baseline Distributions | `HISTORICAL_REFERENCE` | 1966–2017 district yield mean & deciles |

---

## 4. API Endpoints

- `GET /api/monitoring/summary`: Executive operational and monitoring status.
- `GET /api/monitoring/operations`: Detailed request volume, crop breakdown, and strategy telemetry.
- `GET /api/monitoring/distributions`: Empirical prediction moments vs historical baseline moments.
- `GET /api/monitoring/drift`: Continuous feature PSI/KS statistics and coverage drift.
- `GET /api/monitoring/outcomes`: Verified forecast vs observed outcome evaluations.
- `GET /api/monitoring/errors`: Stratified error breakdowns across temporal, geographic, and regime dimensions.
- `GET /api/monitoring/bias`: Directional systematic bias metrics (NME %) and threshold classifications.
- `GET /api/monitoring/forecast-alerts`: Evidence-backed operational, drift, and bias alerts.
- `GET /api/monitoring/forecast-health`: Subsystem health and record availability.
