# Day 25: Comprehensive Repository Inventory & Asset Classification

## 1. Executive Summary

This inventory audits every directory, module, dataset, model artifact, API schema, frontend component, test suite, and document across the repository. Assets are categorized into:
- **`ACTIVE`**: Actively executing in production forecasting, API serving, tests, or UI rendering.
- **`REFERENCE`**: Authoritative domain documentation, dataset cards, model cards, or scientific audit records.
- **`GENERATED_ARTIFACT`**: Empirical outputs, certified strategy registries, and audit logs.
- **`LEGACY`**: Pre-existing foundational baseline scripts retained for chronological traceability.
- **`DUPLICATE` / `UNUSED` / `SAFE_TO_REMOVE`**: Verified orphaned or obsolete temporary files.

---

## 2. Inventory Classification Matrix

### A. Core Engine & Source Modules (`src/`)

| File / Component | Category | Functional Purpose | Dependency Tracing |
| :--- | :--- | :--- | :--- |
| `src/strategy_registry.py` | `ACTIVE` | Compiles multi-crop strategy registry from Day 23 certification | Imported by `src.prediction_service`, `src.forecast_validation`, `backend.services.modeling_service` |
| `src/certification_guard.py` | `ACTIVE` | Pre-inference governance guard & input validator | Imported by `src.prediction_service`, `tests.test_certification_guard` |
| `src/forecast_router.py` | `ACTIVE` | Strategy router executing ML and statistical baselines | Imported by `src.prediction_service`, `tests.test_forecast_router` |
| `src/prediction_provenance.py` | `ACTIVE` | Constructs SHA-256 cryptographic provenance records | Imported by `src.prediction_service`, `tests.test_prediction_provenance` |
| `src/prediction_audit.py` | `ACTIVE` | Thread-safe append logger for prediction audit trail | Imported by `src.prediction_service`, `backend.services.modeling_service` |
| `src/prediction_service.py` | `ACTIVE` | Master forecast request orchestrator | Imported by `backend.services.modeling_service`, `src.forecast_validation` |
| `src/forecast_validation.py` | `ACTIVE` | Deterministic verification and benchmark runner | Executed directly via `python -m src.forecast_validation` |
| `src/multicrop_pipeline.py` | `ACTIVE` | Zero-leakage pre-season lag feature generator | Used by multicrop pipelines and model loaders |
| `src/multicrop_model_selection.py`| `ACTIVE`| Evaluates models across temporal walk-forward folds | Produces `multicrop_model_selection.csv` |
| `src/multicrop_error_diagnosis.py`| `ACTIVE`| Computes yield regime and district error decomposition | Produces `multicrop_district_errors.csv` |
| `src/exogenous_ablation.py` | `ACTIVE` | Exogenous weather/market feature ablation benchmark | Produces `exogenous_ablation_results.csv` |
| `src/final_model_certification.py`| `ACTIVE`| Evaluates final certification criteria across 14 crops | Produces `final_model_certification.csv` |
| `src/reproducibility_audit.py` | `ACTIVE` | Dual-run bitwise reproducibility verifier | Produces `reproducibility_certificate.json` |
| `src/decision_intelligence.py` | `ACTIVE` | Multi-module fusion engine for executive decision briefs | Backend Decision Intelligence service |
| `src/explainability_engine.py` | `ACTIVE` | Tree SHAP feature attribution & local sensitivity | Backend Explainability service |
| `src/scenario_engine.py` | `ACTIVE` | SLSQP Pareto scenario optimization engine | Backend Scenario Intelligence service |
| `src/early_warning_engine.py` | `ACTIVE` | Multi-window temporal monitoring & CUSUM drift | Backend Early Warning service |

### B. Backend API Layer (`backend/`)

| File / Component | Category | Functional Purpose |
| :--- | :--- | :--- |
| `backend/main.py` | `ACTIVE` | Master FastAPI router (50+ endpoints including 7 `/api/forecast/*` routes) |
| `backend/schemas/modeling.py` | `ACTIVE` | Pydantic data contracts for modeling readiness, robustness, certification, and forecast serving |
| `backend/services/modeling_service.py`| `ACTIVE` | Domain queries for multi-crop readiness, certification, and forecast prediction |
| `backend/services/agriculture_service.py`| `ACTIVE` | Historical agricultural panel queries |
| `backend/services/risk_service.py` | `ACTIVE` | Composite risk scoring engine |
| `backend/services/explainability_service.py`| `ACTIVE`| Feature attribution bridge |

### C. Frontend Application (`frontend/`)

| File / Component | Category | Functional Purpose |
| :--- | :--- | :--- |
| `frontend/src/pages/ForecastIntelligence.tsx` | `ACTIVE` | 3-tab production forecast wizard, strategy registry & audit trail UI |
| `frontend/src/pages/ModelingReadiness.tsx` | `ACTIVE` | 7-tab multi-crop readiness, robustness, exogenous & certification UI |
| `frontend/src/pages/DecisionIntelligence.tsx` | `ACTIVE` | Executive decision briefs & provenance DAGs |
| `frontend/src/pages/AgriculturalMonitoring.tsx`| `ACTIVE` | Real-time temporal anomaly & drift monitoring |
| `frontend/src/services/api.ts` | `ACTIVE` | TanStack Query API client and hooks |
| `frontend/src/types/modeling.ts` | `ACTIVE` | TypeScript interfaces for all serving & certification contracts |
| `frontend/src/routes/AppRoutes.tsx` | `ACTIVE` | React Router configuration |

### D. Datasets & Processed Panels (`Datasets/`)

| File / Component | Category | Record Count / Description |
| :--- | :--- | :--- |
| `Datasets/processed/agricultural_panel.csv` | `ACTIVE` | Authoritative `AGRI_PANEL_1.0` (71,601 records, 29 crops, 1966–2017) |
| `Datasets/metadata/forecast_strategy_registry.csv`| `GENERATED_ARTIFACT` | Authoritative strategy table across 14 certified commodities |
| `Datasets/metadata/forecast_coverage.csv` | `GENERATED_ARTIFACT` | 9,019 crop-state-district coverage mappings |
| `Datasets/metadata/forecast_validation_results.csv`| `GENERATED_ARTIFACT` | Dual-run deterministic validation results ($\Delta = 0$) |
| `Datasets/metadata/prediction_audit_log.csv`| `GENERATED_ARTIFACT` | Append-oriented audit log of all inference requests & rejections |
| `Datasets/metadata/final_model_certification.csv`| `GENERATED_ARTIFACT` | Final Day 23 certification matrix |
| `Datasets/rice_data.csv` | `LEGACY` | Historical baseline rice dataset (ICRISAT 1966–2017) |

### E. Model Artifacts (`Models/`)

| File / Component | Category | Description |
| :--- | :--- | :--- |
| `Models/multicrop/oilseeds/model_pipeline.pkl` | `ACTIVE` | Certified `RandomForestRegressor` pipeline for Oilseeds |
| `Models/multicrop/sugarcane/model_pipeline.pkl` | `ACTIVE` | Certified `GradientBoostingRegressor` pipeline for Sugarcane |
| `Models/multicrop/forecast_strategy_registry.json`| `GENERATED_ARTIFACT`| Authoritative JSON forecast registry |
| `Models/multicrop/model_registry.json` | `GENERATED_ARTIFACT`| Multi-crop model registry with evaluation metrics |
| `Models/multicrop/reproducibility_certificate.json`| `GENERATED_ARTIFACT`| Cryptographic SHA-256 reproducibility certificate |
| `Models/rice_random_forest.joblib` | `LEGACY` | Rice baseline model artifact ($R^2=0.7866$) |

### F. Obsolete / Redundant Files Identified for Cleanup

| File / Component | Category | Cleanup Recommendation |
| :--- | :--- | :--- |
| `CLEANUP_INVENTORY.md` (root) | `LEGACY` | Superseded by `docs/DAY25_REPOSITORY_INVENTORY.md` |
| `CLEANUP_REVIEW.md` (root) | `LEGACY` | Superseded by `docs/REPOSITORY_CLEANUP_AUDIT.md` |
| `FINAL_REPOSITORY_CLEANUP.md` (root) | `LEGACY` | Superseded by Day 25 audit docs |
| `FINAL_REPOSITORY_STRUCTURE.md` (root) | `LEGACY` | Superseded by Day 25 architecture docs |
| `RELEASE_CHECKLIST.md` (root) | `LEGACY` | Superseded by Day 25 release status scorecard |
| `DAY17_MULTI_CROP_IMPLEMENTATION_REPORT.md` (root) | `LEGACY` | Relocate to `docs/` for historical tracking |
