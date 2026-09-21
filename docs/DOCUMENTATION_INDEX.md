# Canonical Documentation Index

This index organizes all research, architecture, methodology, operational, and development documentation across the platform.

---

## 1. System Architecture & Operations
- [ARCHITECTURE.md](ARCHITECTURE.md) — Complete 8-layer architecture specification with exact technology stack.
- [DEPLOYMENT.md](DEPLOYMENT.md) — Local development and containerized Docker setup guide.
- [API_REFERENCE.md](API_REFERENCE.md) — Complete REST API catalog and request/response specifications.

---

## 2. Multi-Crop Data Foundation & Governance (Day 17)
- [DATASET.md](DATASET.md) — Complete multi-crop agricultural panel dataset specification (`AGRI_PANEL_1.0`, 71,601 records).
- [DATA_SOURCES.md](DATA_SOURCES.md) — Authoritative sources registry (ICRISAT, GoI OGD, FAOSTAT).
- [CROP_COVERAGE.md](CROP_COVERAGE.md) — Complete breakdown of 29 verified crop commodities.
- [GEOGRAPHIC_COVERAGE.md](GEOGRAPHIC_COVERAGE.md) — State and district longitudinal distribution.
- [TEMPORAL_COVERAGE.md](TEMPORAL_COVERAGE.md) — Multi-season annual temporal matrix (2010–2017).
- [MULTI_CROP_DATA_GOVERNANCE.md](MULTI_CROP_DATA_GOVERNANCE.md) — Unit standardization, missingness handling, and data quality standards.
- [MULTI_CROP_MODEL_COMPATIBILITY.md](MULTI_CROP_MODEL_COMPATIBILITY.md) — Model boundary specification and Rice model containment rules.
- [MULTI_CROP_DATASET_SCORECARD.md](MULTI_CROP_DATASET_SCORECARD.md) — Multi-crop data quality scorecard (14/14 checks passed).

---

## 3. Multi-Crop Modeling Readiness & Baselines (Day 18)
- [DAY18_PRE_MODELING_AUDIT.md](DAY18_PRE_MODELING_AUDIT.md) — Pre-modeling audit and dataset profile inspection.
- [MULTI_CROP_MODELING_READINESS.md](MULTI_CROP_MODELING_READINESS.md) — Complete 29-crop readiness registry (14 Model Ready, 9 Analytics Ready, 6 Insufficient).
- [MULTI_CROP_MODEL_ARCHITECTURE_DECISION.md](MULTI_CROP_MODEL_ARCHITECTURE_DECISION.md) — Empirical justification for separate crop-specific regressors vs single pooled model.
- [MULTI_CROP_FEATURE_COMPATIBILITY.md](MULTI_CROP_FEATURE_COMPATIBILITY.md) — Feature observation timing, pre-season validity, and anti-leakage audit.
- [MULTI_CROP_BASELINE_METHODOLOGY.md](MULTI_CROP_BASELINE_METHODOLOGY.md) — 4 deterministic baseline model benchmarks across 98 evaluation instances.
- [DAY18_SCIENTIFIC_VALIDATION.md](DAY18_SCIENTIFIC_VALIDATION.md) — Day 18 scientific validation report.
- [DAY18_FINAL_SCORECARD.md](DAY18_FINAL_SCORECARD.md) — Final modeling readiness scorecard.

---

## 4. Multi-Crop Forecasting: Random Forest vs Gradient Boosting (Day 19)
- [DAY19_MODELING_METHODOLOGY.md](DAY19_MODELING_METHODOLOGY.md) — Multi-crop pre-season feature engineering, temporal CV, and zero-leakage training.
- [DAY19_MODEL_RESULTS.md](DAY19_MODEL_RESULTS.md) — Empirical leaderboard, Random Forest vs Gradient Boosting vs Baselines comparison.
- [DAY19_ERROR_ANALYSIS.md](DAY19_ERROR_ANALYSIS.md) — Error quantiles ($P_{25}, P_{50}, P_{75}, P_{90}$) and empirical tree dispersion.
- [DAY19_MODEL_REGISTRY.md](DAY19_MODEL_REGISTRY.md) — Multi-crop model registry schema, artifact storage layout, and SHA-256 hashes.
- [DAY19_SCIENTIFIC_VALIDATION.md](DAY19_SCIENTIFIC_VALIDATION.md) — 10-point scientific validation checklist and protocol compliance.
- [DAY19_FINAL_SCORECARD.md](DAY19_FINAL_SCORECARD.md) — Day 19 final scorecard (14 crops modeled, 4 accepted, 10 baseline-preferred).

---

## 5. Temporal Robustness & Walk-Forward Validation (Day 20)
- [DAY20_TEMPORAL_ROBUSTNESS.md](DAY20_TEMPORAL_ROBUSTNESS.md) — Walk-forward expanding-window cross-validation methodology (2014–2017 origins).
- [DAY20_MODEL_STABILITY.md](DAY20_MODEL_STABILITY.md) — Multi-origin model stability, fold win rates, and Day 19 lineage transition audit.
- [DAY20_FEATURE_TIMING_AUDIT.md](DAY20_FEATURE_TIMING_AUDIT.md) — Feature observation timing, zero lookahead enforcement, and spatial cluster audit.
- [DAY20_ERROR_STABILITY.md](DAY20_ERROR_STABILITY.md) — Residual distributions across expanding folds and 2016 climate shock analysis.
- [DAY20_SCIENTIFIC_VALIDATION.md](DAY20_SCIENTIFIC_VALIDATION.md) — Hypothesis testing ($H_0$), empirical evidence, and scientific integrity proof.
- [DAY20_FINAL_SCORECARD.md](DAY20_FINAL_SCORECARD.md) — Day 20 final multi-crop scorecard (2 Robust Accepted, 10 Split-Sensitive, 2 Baseline Preferred).

---

## 6. Crop-Specific Model Selection & Error Diagnosis (Day 21)
- [DAY21_MODEL_DIAGNOSIS.md](DAY21_MODEL_DIAGNOSIS.md) — Multi-origin walk-forward error diagnosis, regime breakdown, and decision hierarchy.
- [DAY21_ERROR_ANALYSIS.md](DAY21_ERROR_ANALYSIS.md) — Five-axis error decomposition (quantiles, yield regimes, temporal regimes, 2016 shock, districts).
- [DAY21_FEATURE_DIAGNOSTICS.md](DAY21_FEATURE_DIAGNOSTICS.md) — Fold-by-fold feature stability scores ($S_{\text{feat}}$), rank variance, and timing audit.
- [DAY21_MODEL_SELECTION.md](DAY21_MODEL_SELECTION.md) — Deterministic model selection rules and Day 19 → Day 20 → Day 21 lineage audit.
- [DAY21_FORECASTING_STRATEGY.md](DAY21_FORECASTING_STRATEGY.md) — Crop-specific operational policies, fallback architectures, and evidence strength scores.
- [DAY21_SCIENTIFIC_VALIDATION.md](DAY21_SCIENTIFIC_VALIDATION.md) — Scientific integrity verification and protocol compliance checklist.
- [DAY21_FINAL_SCORECARD.md](DAY21_FINAL_SCORECARD.md) — Day 21 final executive scorecard and artifact catalog.

---

## 7. Exogenous Data Integration & Pre-Season Feature Expansion (Day 22)
- [DAY22_EXOGENOUS_DATA.md](DAY22_EXOGENOUS_DATA.md) — Authoritative external data integration, provider licenses, and spatial/temporal resolutions.
- [DAY22_TEMPORAL_ALIGNMENT.md](DAY22_TEMPORAL_ALIGNMENT.md) — Pre-season forecast origin ($t \le \text{May 31}$), zero lookahead mandate, and timing audit.
- [DAY22_GEOGRAPHIC_ALIGNMENT.md](DAY22_GEOGRAPHIC_ALIGNMENT.md) — 311 standardized district mapping and 100% spatial match rate certification.
- [DAY22_FEATURE_ENGINEERING.md](DAY22_FEATURE_ENGINEERING.md) — Mathematical formulas and agronomic definitions for 10 pre-season features.
- [DAY22_LEAKAGE_AUDIT.md](DAY22_LEAKAGE_AUDIT.md) — 6-point static and empirical anti-leakage audit gate verification.
- [DAY22_ABLATION_RESULTS.md](DAY22_ABLATION_RESULTS.md) — 5-tier ablation benchmark (EXP-22A through EXP-22E) evaluating marginal gains.
- [DAY22_MODEL_COMPARISON.md](DAY22_MODEL_COMPARISON.md) — Model A vs Model B vs Model C multi-origin comparison and the "negative result" principle.
- [DAY22_SCIENTIFIC_VALIDATION.md](DAY22_SCIENTIFIC_VALIDATION.md) — Statistical proof of mutual information decay and non-causal guardrails.
- [DAY22_FINAL_SCORECARD.md](DAY22_FINAL_SCORECARD.md) — Final executive scorecard and production deployment recommendations.

---

## 7. Machine Learning Methodology (Rice Scoped Baseline)
- [DATA_METHODOLOGY.md](DATA_METHODOLOGY.md) — Panel preprocessing protocols, outlier quarantining, and anti-leakage splitting.
- [FINAL_ML_METHODOLOGY.md](FINAL_ML_METHODOLOGY.md) — Pre-season and post-harvest Random Forest specifications and validation.
- [MODEL_RESULTS.md](MODEL_RESULTS.md) — Model comparison benchmarks, error distributions, calibration, and drift.
- [MODEL_VALIDATION.md](MODEL_VALIDATION.md) — Out-of-time evaluation, decile calibration curves, and PSI monitoring.

---

## 6. Spatial, Temporal & Decision Intelligence
- [GEOSPATIAL_METHODOLOGY.md](GEOSPATIAL_METHODOLOGY.md) — Spatial clustering, Moran's I spatial autocorrelation, and regional similarity.
- [MONITORING_EARLY_WARNING.md](MONITORING_EARLY_WARNING.md) — Multi-window temporal monitoring, CUSUM drift detection, and backtesting.
- [EXPLAINABILITY_METHODOLOGY.md](EXPLAINABILITY_METHODOLOGY.md) — Tree SHAP feature attributions and local sensitivity analysis.
- [SCENARIO_METHODOLOGY.md](SCENARIO_METHODOLOGY.md) — Hypothetical simulation, SLSQP Pareto optimization, and sensitivity sweeps.
- [DECISION_INTELLIGENCE_METHODOLOGY.md](DECISION_INTELLIGENCE_METHODOLOGY.md) — Multi-module evidence fusion, robustness, provenance, and audit certificates.
- [DECISION_EVIDENCE_REPORTS.md](DECISION_EVIDENCE_REPORTS.md) — Standardized Markdown and HTML decision report formats.

---

## 7. Scientific Governance & Portfolio Readiness
- [SCIENTIFIC_LIMITATIONS.md](SCIENTIFIC_LIMITATIONS.md) — 14 explicit domain limitations and non-causal guardrails.
- [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) — Concise platform research results and measured metrics.
- [EXPERIMENTAL_JOURNEY.md](EXPERIMENTAL_JOURNEY.md) — Chronological evolution from baseline to Day 16 production.
- [FINAL_RESEARCH_SUMMARY.md](FINAL_RESEARCH_SUMMARY.md) — Academic paper-style research summary.
- [PORTFOLIO_DESCRIPTION.md](PORTFOLIO_DESCRIPTION.md) — Portfolio overviews and resume-ready highlights.
- [INTERVIEW_PREPARATION.md](INTERVIEW_PREPARATION.md) — 22 technical interview questions and deep-dive answers.
- [DEMO_GUIDE.md](DEMO_GUIDE.md) — 5–7 minute live demonstration script and 15-slide technical storyboard.
- [FUTURE_WORK.md](FUTURE_WORK.md) — Future research roadmap and planned extensions.

---

## 8. Multi-Crop Modeling, Temporal Validation & Model Certification (Days 18–23)
- [DAY23_FINAL_VALIDATION.md](DAY23_FINAL_VALIDATION.md) — Master Day 23 validation coordinator and temporal range audit.
- [DAY23_RESIDUAL_DIAGNOSTICS.md](DAY23_RESIDUAL_DIAGNOSTICS.md) — Residual quantiles, yield-regime decompositions, and prediction bias.
- [DAY23_STRATEGY_EVALUATION.md](DAY23_STRATEGY_EVALUATION.md) — Operational policy evaluation vs ML and statistical baselines.
- [DAY23_REPRODUCIBILITY.md](DAY23_REPRODUCIBILITY.md) — Dual-run bitwise reproducibility audit and cryptographic certificate.
- [DAY23_MODEL_CERTIFICATION.md](DAY23_MODEL_CERTIFICATION.md) — Multi-crop operational taxonomy and certified deployment tiers.
- [DAY23_SCIENTIFIC_VALIDATION.md](DAY23_SCIENTIFIC_VALIDATION.md) — Scientific validation principles, directives, and lineage history.
- [DAY23_FINAL_SCORECARD.md](DAY23_FINAL_SCORECARD.md) — Authoritative platform scorecard and deployment recommendations.

---

## 9. Production Forecast Serving & Governance (Day 24)
- [DAY24_FORECAST_SERVING.md](DAY24_FORECAST_SERVING.md) — Production forecast serving architecture and certified strategy router.
- [DAY24_MODEL_GOVERNANCE.md](DAY24_MODEL_GOVERNANCE.md) — Model governance principles, pre-inference gates, and operational routing rules.
- [DAY24_PREDICTION_PROVENANCE.md](DAY24_PREDICTION_PROVENANCE.md) — Prediction provenance schema, cryptographic SHA-256 lineage, and audit logging.
- [DAY24_SAFETY_GUARDS.md](DAY24_SAFETY_GUARDS.md) — Pre-inference safety validation, rejection code taxonomy, and variance bounds.
- [DAY24_API_CONTRACT.md](DAY24_API_CONTRACT.md) — Production REST API contract, request/response schemas, and endpoint specifications.
- [DAY24_SCIENTIFIC_VALIDATION.md](DAY24_SCIENTIFIC_VALIDATION.md) — Scientific validation, dual-pass invariance benchmarks, and rejection auditing.
- [DAY24_FINAL_SCORECARD.md](DAY24_FINAL_SCORECARD.md) — Day 24 master serving scorecard and operational certification verdict.

---

## 10. Final Research Packaging & Scientific Audit (Day 25)
- [DATASET_CARD.md](DATASET_CARD.md) — Standardized dataset card for `AGRI_PANEL_1.0` (71,601 records, source tracking, boundaries).
- [MODEL_CARDS.md](MODEL_CARDS.md) — Comprehensive model cards for Oilseeds, Sugarcane, and 12 statistical baseline strategies.
- [ARCHITECTURE.md](ARCHITECTURE.md) — Full system architecture specification and Mermaid execution blueprint.
- [FINAL_SCIENTIFIC_AUDIT.md](FINAL_SCIENTIFIC_AUDIT.md) — Canonical validation protocols, metric definitions, and terminology standard.
- [FINAL_REPRODUCIBILITY_REPORT.md](FINAL_REPRODUCIBILITY_REPORT.md) — Final dual-run bitwise invariance benchmark ($\Delta = 0.000000$).
- [DEMO_SCRIPT.md](DEMO_SCRIPT.md) — 5–7 minute step-by-step interview demonstration script.
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) — Executive portfolio summary and verified scale highlights.
- [DAY25_REPOSITORY_INVENTORY.md](DAY25_REPOSITORY_INVENTORY.md) — Comprehensive repository asset inventory and categorization.
- [REPOSITORY_CLEANUP_AUDIT.md](REPOSITORY_CLEANUP_AUDIT.md) — File triage and cleanup verification audit.
- [DAY25_FINAL_STATUS.md](DAY25_FINAL_STATUS.md) — Day 25 compliance matrix and production release scorecard.

---

## 11. Production Deployment & Cloud Readiness (Day 26)
- [DAY26_DEPLOYMENT.md](DAY26_DEPLOYMENT.md) — Production deployment manual, Docker Compose operations, and configuration.
- [DAY26_DEPLOYMENT_ARCHITECTURE.md](DAY26_DEPLOYMENT_ARCHITECTURE.md) — Production deployment architecture blueprint and Mermaid topologies.
- [DAY26_DEPLOYMENT_AUDIT.md](DAY26_DEPLOYMENT_AUDIT.md) — Pre-deployment architecture, path, and security audit report.
- [DAY26_FINAL_STATUS.md](DAY26_FINAL_STATUS.md) — Day 26 containerization and operation compliance scorecard.

---

## 12. Performance Engineering & Load Testing (Day 27)
- [DAY27_PERFORMANCE_BASELINE.md](DAY27_PERFORMANCE_BASELINE.md) — Latency and throughput benchmarks across analytical and inference endpoints.
- [DAY27_LOAD_TEST_REPORT.md](DAY27_LOAD_TEST_REPORT.md) — Concurrent multi-client load testing and throughput validation.
- [DAY27_DETERMINISM_REPORT.md](DAY27_DETERMINISM_REPORT.md) — Concurrent bitwise deterministic inference verification.
- [DAY27_RESOURCE_PROFILE.md](DAY27_RESOURCE_PROFILE.md) — CPU, memory RSS, and process profiling under sustained workload.
- [DAY27_PERFORMANCE_AUDIT.md](DAY27_PERFORMANCE_AUDIT.md) — Critical bottlenecks, hot-path analysis, and caching efficacy.
- [DAY27_FINAL_STATUS.md](DAY27_FINAL_STATUS.md) — Day 27 performance scorecard and latency SLO sign-off.

---

## 13. Production Observability & Operational Intelligence (Day 28)
- [DAY28_OBSERVABILITY_AUDIT.md](DAY28_OBSERVABILITY_AUDIT.md) — Pre-observability audit and telemetry assessment.
- [DAY28_MONITORING_ARCHITECTURE.md](DAY28_MONITORING_ARCHITECTURE.md) — Production observability architecture, ring buffers, and storage topology.
- [DAY28_RUNTIME_METRICS.md](DAY28_RUNTIME_METRICS.md) — Telemetry principles, latency percentiles ($P_{50}, P_{90}, P_{95}, P_{99}$), and zero-fabrication standards.
- [DAY28_ALERTING_POLICY.md](DAY28_ALERTING_POLICY.md) — Configured operational alert rules, thresholds, and evaluation lifecycle.
- [DAY28_FAILURE_DIAGNOSTICS.md](DAY28_FAILURE_DIAGNOSTICS.md) — Structured error categorization taxonomy and operational remediation guidance.
- [DAY28_FORECAST_TRACE.md](DAY28_FORECAST_TRACE.md) — Step-by-step 6-stage prediction trace pipeline, microsecond latency capture, and audit fallback.
- [DAY28_FINAL_STATUS.md](DAY28_FINAL_STATUS.md) — Day 28 final operational intelligence compliance scorecard.

---

## 14. Prediction Explorer & Forecast Explainability (Day 29)
- [DAY29_PREDICTION_EXPLORER.md](DAY29_PREDICTION_EXPLORER.md) — Traceable agricultural yield prediction explorer architecture, user flow, and semantic labels.
- [DAY29_FORECAST_EXPLAINABILITY.md](DAY29_FORECAST_EXPLAINABILITY.md) — Strategy resolution, model evidence, feature importance attribution, and empirical uncertainty limits.
- [DAY29_SCIENTIFIC_VALIDATION.md](DAY29_SCIENTIFIC_VALIDATION.md) — Dual-run bitwise invariance, temporal correctness enforcement, and cryptographic provenance integrity.
- [DAY29_FINAL_STATUS.md](DAY29_FINAL_STATUS.md) — Day 29 final compliance matrix and release scorecard.

---

## 15. Forecast Monitoring, Drift Detection & Outcome Intelligence (Day 30)
- [DAY30_FORECAST_MONITORING.md](DAY30_FORECAST_MONITORING.md) — Operational monitoring architecture, request telemetry tracking, prediction distribution moments, and semantic classifications.
- [DAY30_OUTCOME_EVALUATION.md](DAY30_OUTCOME_EVALUATION.md) — Leak-free post-outcome evaluation methodology, temporal isolation boundaries, and stratified error decomposition.
- [DAY30_DRIFT_AND_BIAS.md](DAY30_DRIFT_AND_BIAS.md) — Population Stability Index (PSI) drift calculation, covariate stability thresholds, and directional bias diagnostics.
- [DAY30_SCIENTIFIC_VALIDATION.md](DAY30_SCIENTIFIC_VALIDATION.md) — Scientific integrity audit, determinism invariance, and 4-golden-case validation scorecard.
- [DAY30_FINAL_STATUS.md](DAY30_FINAL_STATUS.md) — Day 30 final compliance scorecard, operational statistics, and sign-off report.

---

## 16. Decision Intelligence & Evidence-Based Forecast Briefs (Day 31)
- [DAY31_DECISION_INTELLIGENCE.md](DAY31_DECISION_INTELLIGENCE.md) — Decision intelligence architecture, evidence harvesting engine, and lifecycle integration.
- [DAY31_EVIDENCE_FRAMEWORK.md](DAY31_EVIDENCE_FRAMEWORK.md) — Semantic evidence typologies, completeness scoring, and structured evidence schema.
- [DAY31_DECISION_BRIEF.md](DAY31_DECISION_BRIEF.md) — 9-dimension decision brief specification, export formats, and section layout.
- [DAY31_SCIENTIFIC_VALIDATION.md](DAY31_SCIENTIFIC_VALIDATION.md) — 11-rule automated validation suite, non-causal language guard, and test suite verification.
- [DAY31_FINAL_STATUS.md](DAY31_FINAL_STATUS.md) — Day 31 final compliance matrix, deliverables checklist, and release sign-off.

---

## 17. Decision Workspace & Scenario Comparison (Day 32)
- [DAY32_DECISION_WORKSPACE.md](DAY32_DECISION_WORKSPACE.md) — Decision workspace architecture, evidence-first synthesis layer, card specifications, and non-autonomous governance.
- [DAY32_SCENARIO_INTEGRATION.md](DAY32_SCENARIO_INTEGRATION.md) — Governed scenario engine integration, simulation vs forecast distinction, and side-by-side comparison matrix.
- [DAY32_EVIDENCE_TRACEABILITY.md](DAY32_EVIDENCE_TRACEABILITY.md) — Strict semantic taxonomy (`OBSERVED`, `PREDICTED`, `SCENARIO`, etc.), SHA-256 provenance fingerprinting, and temporal boundary enforcement.
- [DAY32_SCIENTIFIC_VALIDATION.md](DAY32_SCIENTIFIC_VALIDATION.md) — Scientific freeze verification, 4-golden-case validation, edge-case audit, and analytical determinism report.
- [DAY32_FINAL_STATUS.md](DAY32_FINAL_STATUS.md) — Day 32 final compliance scorecard, test verification summary, and real discovered limitations.

---

## 18. Production Acceptance, Security & Failure Resilience (Day 33)
- [DAY33_PRODUCTION_ACCEPTANCE.md](DAY33_PRODUCTION_ACCEPTANCE.md) — Executive acceptance summary, 100% test pass confirmation, signoff criteria, and environment validation.
- [DAY33_SECURITY_AUDIT.md](DAY33_SECURITY_AUDIT.md) — Security test results, path traversal results, injection testing, input bounds, security headers, exception containment, and OWASP alignment.
- [DAY33_FAILURE_RESILIENCE.md](DAY33_FAILURE_RESILIENCE.md) — Failure isolation testing, graceful degradation, missing service fallback, audit error containment, and boundary behaviors.
- [DAY33_END_TO_END_WORKFLOWS.md](DAY33_END_TO_END_WORKFLOWS.md) — Golden journeys for all 4 commodities (Oilseeds, Sugarcane, Rice, Wheat) + complete negative/edge cases matrix.
- [DAY33_API_CONTRACT_VALIDATION.md](DAY33_API_CONTRACT_VALIDATION.md) — Full schema contract validation across all 12 key endpoints, status codes, payload shapes, and backward compatibility.
- [DAY33_SCIENTIFIC_REGRESSION.md](DAY33_SCIENTIFIC_REGRESSION.md) — Verification of zero scientific drift, frozen models, preserved walk-forward validation metrics, and reproducible baseline comparisons.
- [DAY33_FINAL_STATUS.md](DAY33_FINAL_STATUS.md) — Day 33 final acceptance report and executive release signoff.










