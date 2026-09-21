# DAY 36 — Platform Scientific Experiment Lineage

This document traces the complete chronological research and engineering lineage of the Agricultural Forecasting & Decision Intelligence platform across all 35 prior development milestones.

```
DATA FOUNDATION
      ↓
MODEL READINESS
      ↓
CROP-SPECIFIC MODELING
      ↓
TEMPORAL ROBUSTNESS
      ↓
MODEL SELECTION
      ↓
EXOGENOUS ABLATION
      ↓
INDEPENDENT AUDIT
      ↓
STRATEGY GOVERNANCE
      ↓
FORECAST SERVING & PROVENANCE
      ↓
OBSERVABILITY & MONITORING
      ↓
DECISION SUPPORT & WORKSPACE
```

---

## Stage-by-Stage Lineage Details

### Stage 1: Data Foundation (Days 1–17)
- **Scientific Question**: How can fragmented district agricultural records (ICRISAT / DES) across India be harmonized into a leakage-safe, analysis-ready panel?
- **Method**: Standardized district and state administrative names across 1966–2017, extracted 29 canonical crops, applied deterministic unit transformations, and separated algebraic identities ($Yield = \frac{Production}{Area} \times 1000$) from predictive modeling data.
- **Result**: Unified long panel of **71,601 records** across 20 states and 311 districts (2010–2017 active window) and single-crop Rice historical panel of 2,469 records.
- **Decision**: Reject raw concurrent production as an input to yield models to prevent 100% target leakage.
- **Authoritative Artifacts**:
  - `Datasets/processed/agricultural_panel.csv` (SHA-256: `13f882d7d4617e77b6ded31c7febb55e599f3f6a13e981f23b94c4cecd47f13b`)
  - `Datasets/metadata/dataset_manifest.json`
  - `Datasets/metadata/crop_coverage.csv`

---

### Stage 2: Model Readiness & Baseline Screening (Day 18)
- **Scientific Question**: Which of the 29 canonical crops possess sufficient sample size, geographic breadth, and temporal continuity for pre-season yield forecasting?
- **Method**: Implemented explicit sufficiency gates ($N \ge 100$ district-year observations, multi-district dispersion, temporal span $\ge 5$ years) and benchmarked against 4 zero-leakage baselines (Naive Persistence, Historical District Mean, Historical Crop Mean, Linear District Trend).
- **Result**: 14 crops classified as `MODEL_READY`; 15 crops classified as `ANALYTICS_READY` or `INSUFFICIENT_DATA`.
- **Decision**: Limit machine learning tournament evaluation strictly to the 14 model-ready commodities.
- **Authoritative Artifacts**:
  - `Datasets/metadata/crop_model_readiness.csv`
  - `docs/DAY18_PRE_MODELING_AUDIT.md`

---

### Stage 3: Crop-Specific Pre-Season Modeling (Day 19)
- **Scientific Question**: Can historical lag features ($t-1$ yield, 3-year rolling yield, cultivated area) train initial out-of-time ML models that outperform simple baselines?
- **Method**: Single chronological train/test split ($\text{Train} \le 2015$, $\text{Test} = 2016-2017$) comparing Random Forest and Gradient Boosting against district mean persistence.
- **Result**: Several crops showed initial promise on the single 2016–2017 split, while others failed to beat district mean persistence.
- **Decision**: Flag single-split results as preliminary; require multi-origin walk-forward testing to detect regime fragility.
- **Authoritative Artifacts**:
  - `Datasets/metadata/crop_target_profiles.csv`
  - `Models/forecasting_model_metadata.json`

---

### Stage 4: Temporal Robustness Evaluation (Day 20)
- **Scientific Question**: Does model performance hold across multiple distinct historical origins without future lookahead?
- **Method**: 4-fold expanding walk-forward temporal evaluation with historical origins $T \in \{2014, 2015, 2016, 2017\}$. Models train only on historical data prior to origin year $T$.
- **Result**: Revealed severe split sensitivity for multiple commodities during the 2015 pan-India drought shock. Several crops that looked acceptable in Day 19 degraded catastrophically under multi-origin testing.
- **Decision**: Disqualify single-split metrics from production claims; require fold win-rate $\ge 50\%$ and mean MAE improvement $> 0$ for production ML.
- **Authoritative Artifacts**:
  - `Datasets/metadata/multicrop_temporal_robustness.csv`
  - `Datasets/metadata/multicrop_fold_results.csv`

---

### Stage 5: Crop-Specific Model Selection & Error Diagnosis (Day 21)
- **Scientific Question**: Which algorithm (RF, GBDT, or Baseline Persistence) is optimal for each of the 14 commodities under temporal cross-validation?
- **Method**: Evaluated fold win-rate, mean MAE improvement, median improvement, worst-fold degradation, and MAE coefficient of variation (CV).
- **Result**:
  - **Oilseeds**: Passed all gates (`ROBUST_ML`, 75% win-rate, +12.79% mean gain, -2.34% worst fold).
  - **Sugarcane**: Failed unconstrained stability (`RESEARCH_CANDIDATE`, 50% win-rate, -1.60% mean gain due to extreme fold 2 variance).
  - **Rice & Wheat**: Statistical baseline persistence consistently outperformed ML (Rice: 25% win rate, -8.39% mean gain; Wheat: 50% win rate, -18.36% mean gain).
- **Decision**: Only Oilseeds qualified as unconstrained robust ML; Rice and Wheat designated for baseline production.
- **Authoritative Artifacts**:
  - `Datasets/metadata/multicrop_model_selection.csv`
  - `Datasets/metadata/multicrop_error_diagnosis.csv`

---

### Stage 6: Pre-Season Exogenous Feature Ablation (Day 22)
- **Scientific Question**: Do pre-season exogenous weather features (rainfall, temperature, drought indicators) improve forecast accuracy over historical-only models?
- **Method**: 5-tier ablation tournament across all 14 crops ($EXP-22A$ Historical, $EXP-22B$ +Rainfall, $EXP-22C$ +Temperature, $EXP-22D$ +Weather, $EXP-22E$ +All Exogenous) under 4-fold walk-forward validation.
- **Result**: **NO MEANINGFUL GAIN** across all 14 crops. Adding pre-season district weather aggregations degraded MAE or increased fold variance.
- **Decision**: Retain historical-only feature sets ($EXP-22A$) as authoritative. Constrain conclusion to tested district-level pre-season lead times without claiming weather has no biological influence.
- **Authoritative Artifacts**:
  - `Datasets/metadata/exogenous_model_selection.csv`
  - `Datasets/metadata/exogenous_ablation_results.csv`
  - `Datasets/metadata/exogenous_fold_results.csv`

---

### Stage 7: Independent Temporal Audit & Strategy Governance (Days 23–25)
- **Scientific Question**: How can models and baselines be safely routed in production while protecting against out-of-distribution regime failures?
- **Method**: Formalized a 3-tier strategy governance matrix (`PRODUCTION_READY`, `CONDITIONAL_PRODUCTION`, `BASELINE_PRODUCTION`) with automated 3-sigma variance clipping and sparse district fallback.
- **Result**:
  - Oilseeds: Certified `PRODUCTION_READY` (RandomForestRegressor).
  - Sugarcane: Certified `CONDITIONAL_PRODUCTION` (GBDT with mandatory 3-sigma district fallback, achieving **+1.19% gain** over baseline).
  - 12 Commodities (including Rice and Wheat): Certified `BASELINE_PRODUCTION` (Historical District Mean).
- **Decision**: Build strategy router enforcing automatic certification guards before inference.
- **Authoritative Artifacts**:
  - `Datasets/metadata/final_model_certification.csv`
  - `Datasets/metadata/final_strategy_results.csv`
  - `Models/multicrop/forecast_strategy_registry.json`

---

### Stage 8: Governed Forecast Serving & Cryptographic Provenance (Days 26–27)
- **Scientific Question**: How can predictions be served securely with complete auditability, reproducible parameters, and zero data leakage?
- **Method**: FastAPI prediction service with runtime parameter validation, pre-inference certification checks, 3-sigma safety bounds, SHA-256 digital provenance hashing, and append-only audit logging.
- **Result**: Sub-50ms inference with immutable lineage fingerprints (`SHA256:7f4a...`) recording feature cutoffs, strategy versions, and execution timestamps.
- **Decision**: Reject any request targeting uncertified combinations with explicit HTTP 400 rejection codes.
- **Authoritative Artifacts**:
  - `src/prediction_service.py`
  - `src/provenance_service.py`
  - `Datasets/metadata/prediction_audit_log.csv`

---

### Stage 9: Observability, Drift Detection & Outcome Intelligence (Days 28–30)
- **Scientific Question**: How can model health, covariate drift, and post-outcome errors be monitored without confusing predictions with observations?
- **Method**: Evaluated runtime telemetry (RSS, CPU, latency), Population Stability Index (PSI) drift monitoring across feature distributions, post-harvest error decomposition (signed bias $\hat{y} - y$), and evidence-first alert flags.
- **Result**: Demonstrated early warning detection of covariate shifts during anomalous years without interrupting live inference.
- **Decision**: Separate runtime monitoring events from historical validation benchmarks in UI and documentation.
- **Authoritative Artifacts**:
  - `src/monitoring_service.py`
  - `Datasets/metadata/operational_telemetry.jsonl`
  - `Datasets/metadata/historical_outcome_evaluations.csv`

---

### Stage 10: Decision Intelligence, Scenarios & Workspace (Days 31–32)
- **Scientific Question**: How can forecasts and simulated what-if scenarios be presented to human policy-makers without making autonomous or prescriptive causal claims?
- **Method**: Built interactive `/decision-workspace` presenting side-by-side matrices of baseline forecasts (`[PREDICTED]`), historical references (`[OBSERVED]`), and parameter perturbations (`[SCENARIO]`) with SLSQP optimization, non-causal attribution disclaimers, and zero subjective rankings.
- **Result**: Decision Briefs and multi-scenario comparisons deliver auditable, non-prescriptive evidence to support human agronomic expertise.
- **Decision**: Strictly disclaim all scenario deltas as mathematical perturbations within trained manifolds, never as biological or policy certainties.
- **Authoritative Artifacts**:
  - `src/decision_workspace.py`
  - `src/decision_intelligence.py`
  - `frontend/src/pages/DecisionWorkspace.tsx`

---

### Stage 11: Production Hardening, Recovery & Product Readiness (Days 33–35)
- **Scientific Question**: Can the entire architecture be containerized, secured, verified for disaster recovery, and made WCAG 2.1 AA accessible under an absolute scientific freeze?
- **Method**: OWASP security headers, non-root execution (`appuser`), fail-closed `/ready` probe (HTTP 503 on missing assets), Nginx reverse proxy, deterministic restart verification, skip-to-content bypass link, semantic labeling audit, and 89 production acceptance tests.
- **Result**: 100% test pass rate across all operational suites with zero modifications to underlying models or data.
- **Decision**: Lock scientific layer and proceed to Day 36 Final Scientific Audit.
- **Authoritative Artifacts**:
  - `tests/test_deployment_verification.py`
  - `tests/test_day35_ui_contracts.py`
  - `docs/DAY35_FINAL_STATUS.md`
