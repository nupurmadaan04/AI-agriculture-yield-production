# Day 29: Prediction Explorer & UI Architecture

## 1. Executive Summary

The **Prediction Explorer** (`/prediction-explorer`) provides a transparent, production-grade inspection interface for the agricultural forecasting platform. It directly connects to the existing Day 24 governed forecast-serving architecture, strategy registry, certification subsystem, cryptographic provenance generator, and immutable audit logs.

The explorer answers:
> *"Why did the system produce this prediction, which strategy generated it, what empirical evidence supports that strategy, what historical context exists, and how can I trace the result back to the registered model and data?"*

---

## 2. End-to-End Architecture & Data Flow

```
React UI (/prediction-explorer)
   ↓
API Service (useForecastContext, useForecastEvidence, api.predictForecast)
   ↓
FastAPI Serving Gateway (/api/forecast/*)
   ↓
PredictionService (Day 24 Master Serving Engine)
   ↓
CertificationGuard (Validates Crop/State/District/Data Integrity)
   ↓
StrategyRegistry (Resolves ML vs Baseline Strategy & Operating Rules)
   ↓
ForecastRouter (Routes to ML Pipeline or Historical District Baseline)
   ↓
PredictionProvenanceBuilder (Generates SHA-256 Fingerprint & Metadata)
   ↓
PredictionAuditLogger (Records Immutable Audit Event)
   ↓
Prediction Explorer UI (Displays Prediction, Context, Evidence & Trace)
```

> **Strict Non-Execution Rule**: The browser client **never** executes model inference. All inference, baseline calculations, and cryptographic fingerprint generation occur exclusively in the governed Python backend.

---

## 3. UI Component Structure

1. **Scenario Input Panel**:
   - Dynamic cascading selector (`Crop` → `Supported States` → `Supported Districts` → `Target Forecast Year`).
   - Sourced from `/api/forecast/coverage` across 14 certified crops and 311 districts.
   - Pre-season feature context controls (automatically populated from historical panel or manual override).
2. **Prominent Prediction Section**:
   - Yield value formatted strictly in canonical units (`kg/ha`).
   - Governed strategy badge (`PRODUCTION_READY`, `CONDITIONAL_PRODUCTION`, `BASELINE_PRODUCTION`).
   - Target geography, model artifact version, and active fallback status.
3. **5-Point Governance Status Checklist**:
   - Dataset Integrity: `PASS (AGRI_PANEL_1.0)`
   - Model Integrity: `PASS (SHA-256 Verified)`
   - Strategy Registration: `PASS (Day 24 Registry)`
   - Geographic Coverage: `PASS (Certified District)`
   - Input Completeness: `PASS (Validated Features)`
4. **Historical Context & Baseline Comparisons**:
   - Metric comparison cards (`PREDICTED`, `HISTORICAL REFERENCE`, `PREVIOUS OBSERVATION`, `DERIVED REFERENCE`).
   - Recent historical observed harvest seasons table (Year, Yield kg/ha, Cultivated Area ha, Production tonnes).
5. **Model Validation Evidence**:
   - 4-origin expanding walk-forward validation evidence (2014–2017).
   - Mean MAE, Baseline MAE, Gain vs Baseline (%), Fold Win Rate (%).
   - Operating rules and fallback strategies.
6. **Empirical Uncertainty Panel**:
   - Displays empirical P10–P90 ensemble spread where supported (e.g. Oilseeds Random Forest ±398.81 kg/ha).
   - Mandatory disclaimer: *"This range represents an empirical ensemble spread and is not a formal confidence interval."*
7. **Model Feature Evidence (XAI)**:
   - Native tree split feature importance attribution for ML models.
   - For statistical baselines: explicitly indicates that feature attribution is not applicable.
8. **Why This Prediction? (Decision Logic)**:
   - 6-step progressive disclosure summary covering strategy selection, model used, input data, validation evidence, fallback logic, and certification.
9. **Cryptographic Provenance Record**:
   - Request ID, Model artifact hash, Dataset version, and SHA-256 fingerprint with one-click JSON copy.
10. **Lifecycle Audit Trace Timeline**:
    - 6-stage sequential verification trace (`INPUT_VALIDATION` → `CERTIFICATION_CHECK` → `STRATEGY_LOOKUP` → `INFERENCE_EXECUTION` → `PROVENANCE_GENERATION` → `AUDIT_LOGGING`).

---

## 4. Semantic Data Classifications

Every metric displayed on the Prediction Explorer is categorized with strict semantic labeling:
- **`OBSERVED`**: Real empirical yield measurements recorded in historical harvest seasons (e.g. previous year yield $t-1$).
- **`DERIVED` / `HISTORICAL_REFERENCE`**: Summary statistics calculated over historical observations (e.g. district historical mean, 3-year rolling mean).
- **`PREDICTED`**: The output of the governed pre-season forecasting model or baseline.
- **`MODEL_ATTRIBUTION`**: Feature importance metrics derived from trained decision trees.
- **`VALIDATION`**: Out-of-sample walk-forward evaluation metrics (MAE, RMSE, fold win rate).
- **`PROVENANCE`**: Cryptographic signatures and execution metadata.
