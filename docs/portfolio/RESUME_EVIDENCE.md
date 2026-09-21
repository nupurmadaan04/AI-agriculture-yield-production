# Verified Resume Evidence Extraction Guide

This guide provides verified, mathematically defensible statements for resume preparation. To ensure technical integrity during background checks and technical interviews, every claim is mapped to its underlying physical repository artifact, accompanied by approved safe phrasing and phrases to strictly avoid.

---

## 1. Data Engineering & Pipeline Infrastructure

```
CATEGORY: DATA ENGINEERING & HARMONIZATION
VERIFIED STATEMENT:
Harmonized longitudinal agricultural panel data containing 71,601 records across 29 crops, 20 Indian states, and 311 districts (1966–2017 historical context; 2010–2017 active panel) with zero duplicate keys and verified algebraic consistency.

EVIDENCE SOURCE:
- Datasets/processed/agricultural_panel.csv (SHA-256: 13f882d7d4617e77b6ded31c7febb55e599f3f6a13e981f23b94c4cecd47f13b)
- Datasets/metadata/dataset_manifest.json
- tests/test_day36_reproducibility.py::test_canonical_agricultural_panel_physical_counts

SAFE WORDING:
"Engineered an automated data harmonization pipeline processing 71,600+ longitudinal district-level agricultural panel records across 29 commodities and 311 districts, enforcing zero data leakage and strict unit consistency."

DO NOT SAY:
- "Built India's largest agricultural dataset." (Unsupported national claim)
- "Scraped real-time agricultural data from across India." (Data is historical panel data, not live web-scraped feeds)
- "Trained models on 52 continuous years across all crops." (Panel is 2010–2017 for multi-crop modeling)
```

```
CATEGORY: LEAKAGE ISOLATION
VERIFIED STATEMENT:
Enforced strict pre-season temporal boundary isolation by masking contemporaneous harvest production from yield regression features, preventing 100% target leakage.

EVIDENCE SOURCE:
- src/feature_pipeline.py
- docs/research_paper/05_methodology.md

SAFE WORDING:
"Designed a zero-leakage pre-season feature extraction pipeline utilizing autoregressive lags and rolling moving averages, eliminating mathematical target leakage from contemporaneous harvest variables."

DO NOT SAY:
- "Achieved 99.9% prediction accuracy on harvest day." (That reflects algebraic reconstruction, not genuine pre-season forecasting)
```

---

## 2. Modeling & Governed Strategy Architecture

```
CATEGORY: CROP-SPECIFIC STRATEGY SELECTION
VERIFIED STATEMENT:
Evaluated machine learning algorithms against statistical persistence baselines across 14 commodities under 4-fold walk-forward cross-validation, establishing an evidence-based governance framework that certified ML for 2 crops and mandated baseline persistence for 12 crops.

EVIDENCE SOURCE:
- Datasets/metadata/final_model_certification.csv
- Models/multicrop/forecast_strategy_registry.json
- tests/test_day36_reproducibility.py::test_oilseeds_governed_ml_evidence_chain

SAFE WORDING:
"Architected an evidence-governed model selection registry that benchmarks tree-based ML ensembles against historical district baselines across 14 crops, certifying unconstrained ML only where walk-forward validation proves statistically superior."

DO NOT SAY:
- "Trained state-of-the-art deep learning models for all Indian crops." (Deep learning was not used; models are RF, GBDT, and Baselines)
- "Deployed machine learning for all agricultural commodities." (ML failed in 12 crops and was responsibly rejected)
```

```
CATEGORY: OILSEEDS PRODUCTION MODELING
VERIFIED STATEMENT:
Certified Random Forest Regressor for Oilseeds achieving a 75.0% fold win-rate and +12.79% mean MAE reduction over historical district mean persistence across four expanding walk-forward folds.

EVIDENCE SOURCE:
- Datasets/metadata/multicrop_model_selection.csv
- Datasets/metadata/final_model_certification.csv

SAFE WORDING:
"Developed a Random Forest yield forecaster for Oilseeds delivering a +12.79% mean MAE improvement over statistical persistence with a 75% win rate across sequential historical validation origins."

DO NOT SAY:
- "Improved oilseed yields by 12.8%." (The model improves forecast accuracy, it does not physically grow more crops)
```

```
CATEGORY: SUGARCANE REGIME INSTABILITY & VARIANCE CLIPPING
VERIFIED STATEMENT:
Diagnosed raw GBDT failure during the 2015 drought shock (-10.19% loss in Fold 2, -1.60% net loss) and implemented an automated 3-sigma district variance clipping fallback that restored an aggregate +1.19% MAE gain over baseline persistence.

EVIDENCE SOURCE:
- Datasets/metadata/multicrop_model_selection.csv
- Datasets/metadata/final_model_certification.csv
- tests/test_day36_reproducibility.py::test_sugarcane_dual_metrics_resolution

SAFE WORDING:
"Mitigated out-of-distribution climate shock errors in Gradient Boosted models by implementing runtime 3-sigma variance clipping, converting a -1.60% raw ML degradation into a +1.19% net operational gain."

DO NOT SAY:
- "Sugarcane model achieved +5.62% accuracy across all years." (Historical typo; authoritative governed result is +1.19%)
```

---

## 3. Negative Scientific Findings & Ablation

```
CATEGORY: EXOGENOUS WEATHER FEATURE ABLATION
VERIFIED STATEMENT:
Conducted a 5-tier walk-forward ablation study across 14 commodities evaluating pre-season rainfall anomalies and temperature extremes, proving that pre-season district weather aggregations produced no meaningful gain over historical autoregressive lags.

EVIDENCE SOURCE:
- Datasets/metadata/exogenous_model_selection.csv
- docs/research_paper/tables/TABLE_08_EXOGENOUS_ABLATION.md
- tests/test_day36_reproducibility.py::test_day22_exogenous_negative_result_invariants

SAFE WORDING:
"Led an empirical 5-tier feature ablation demonstrating that pre-season district weather aggregations offered no statistically meaningful improvement over autoregressive historical lags across 14 commodities."

DO NOT SAY:
- "Proved that weather has no impact on crop growth." (Weather impacts biology, but pre-season district aggregations yielded no gain at pre-planting lead times)
```

---

## 4. Production Engineering, Serving & Reliability

```
CATEGORY: PRODUCTION SERVING & LATENCY
VERIFIED STATEMENT:
Engineered a FastAPI forecast serving runtime achieving sub-50ms P95 latency with Pydantic v2 boundary validation and pre-inference certification checks.

EVIDENCE SOURCE:
- backend/main.py
- src/prediction_service.py
- reports/sequential_benchmark_results.json

SAFE WORDING:
"Engineered high-throughput FastAPI inference microservices delivering sub-50ms P95 latency with strict schema validation and automated safety guards."

DO NOT SAY:
- "Built a distributed Kubernetes cluster handling millions of live daily users." (Deployment is local Docker Compose with Nginx reverse proxy)
```

```
CATEGORY: CRYPTOGRAPHIC PROVENANCE
VERIFIED STATEMENT:
Implemented digital provenance tracking generating bitwise reproducible SHA-256 execution signatures for every prediction, recording model version, input parameters, and dataset hashes in an append-only audit log.

EVIDENCE SOURCE:
- src/provenance_service.py
- Datasets/metadata/prediction_audit_log.csv
- tests/test_provenance_chain.py

SAFE WORDING:
"Integrated cryptographic SHA-256 digital provenance logging into inference serving, guaranteeing immutable audit trails and parameter reproducibility for every generated forecast."
```

```
CATEGORY: OBSERVABILITY & DRIFT MONITORING
VERIFIED STATEMENT:
Built continuous monitoring pipelines tracking feature distribution shifts via Population Stability Index (PSI) and evaluating post-harvest directional signed bias (predicted - observed) against ground-truth census data.

EVIDENCE SOURCE:
- src/monitoring_service.py
- src/outcome_evaluation.py
- Datasets/metadata/operational_telemetry.jsonl

SAFE WORDING:
"Deployed MLOps monitoring pipelines calculating Population Stability Index (PSI) for covariate drift detection and decomposing post-harvest residuals into signed bias and empirical error quantiles."
```

```
CATEGORY: CONTAINERIZATION & ACCESSIBILITY
VERIFIED STATEMENT:
Containerized the full-stack architecture using Docker Compose and Nginx reverse proxy with non-root security (appuser UID 10001), fail-closed readiness probes (HTTP 503), and a WCAG 2.1 AA compliant WebShell.

EVIDENCE SOURCE:
- Dockerfile, docker-compose.yml, nginx/nginx.conf
- tests/test_deployment_verification.py
- tests/test_day35_ui_contracts.py

SAFE WORDING:
"Hardened production multi-container deployments with Nginx reverse proxy, non-root execution, fail-closed health probes, and WCAG 2.1 AA accessible frontend interfaces."
```
