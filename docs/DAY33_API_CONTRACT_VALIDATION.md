# Day 33: API Contract & Schema Conformance Report

## 1. Overview & Protocol

The API Contract Validation suite (`tests/test_api_contracts.py`) exercises the public REST interfaces of the Agricultural Forecasting & Decision Intelligence Platform. Each test verifies:
- HTTP status codes
- Mandatory top-level and nested response keys
- Strict data type conformance (e.g., float yields, ISO-8601 timestamps, list shapes)
- Semantic field invariants (e.g., non-empty SHA-256 strings, valid certification status enums)

All 12 contract tests passed with zero schema violations.

---

## 2. API Contract Verification Matrix

| Endpoint | HTTP Method | Expected Status | Primary Keys Validated | Result |
| :--- | :---: | :---: | :--- | :---: |
| `/health` | `GET` | 200 | `status`, `version`, `timestamp` | **PASSED** |
| `/ready` | `GET` | 200 | `status`, `services` | **PASSED** |
| `/api/forecast/predict` | `POST` | 200 | `request_id`, `crop`, `state`, `district`, `forecast_year`, `forecast_yield_kg_ha`, `strategy`, `model_name`, `model_version`, `certification_status`, `provenance_hash`, `timestamp` | **PASSED** |
| `/api/forecast/strategies` | `GET` | 200 | Dictionary of supported crops mapping to strategy metadata objects | **PASSED** |
| `/api/forecast/coverage` | `GET` | 200 | `supported_crops`, `commodity_details`, `temporal_range`, `geographic_coverage` | **PASSED** |
| `/api/forecast/provenance/{req_id}` | `GET` | 200 | `request_id`, `provenance_hash`, `model_artifact_hash`, `data_source`, `features_used`, `certification_status` | **PASSED** |
| `/api/workspace/analyze` | `POST` | 200 | `workspace_id`, `crop`, `state`, `district`, `forecast_year`, `baseline_forecast`, `historical_context`, `validation`, `uncertainty`, `monitoring`, `attribution`, `scenarios`, `comparison_matrix`, `limitations`, `generated_at` | **PASSED** |
| `/api/workspace/templates` | `GET` | 200 | `archetypes`, `supported_features`, `parameter_bounds`, `disclaimer` | **PASSED** |
| `/api/decision/analyze` | `POST` | 200 | `decision_id`, `context`, `brief`, `is_scientifically_validated` | **PASSED** |
| `/api/decision/brief` | `GET` | 200 | `decision_id`, `executive_summary`, `historical_context`, `validation_evidence`, `uncertainty_evidence`, `monitoring_evidence`, `limitations`, `audit_record` | **PASSED** |
| `/api/observability/health` | `GET` | 200 | `status`, `subsystems`, `overall_health` | **PASSED** |
| `/api/observability/summary` | `GET` | 200 | `operations`, `runtime`, `monitoring`, `telemetry_generated_at` | **PASSED** |

---

## 3. Schema Structure Highlights

### 3.1 Workspace Master Schema (`DecisionWorkspaceResponse`)
```json
{
  "workspace_id": "WS-XXXXXXXXXX",
  "crop": "Oilseeds",
  "state": "Madhya Pradesh",
  "district": "Indore",
  "forecast_year": 2017,
  "generated_at": "2026-09-21T08:51:34.082105+00:00",
  "baseline_forecast": {
    "forecast_yield_kg_ha": 767.7,
    "strategy": "Historical ML (RandomForestRegressor)",
    "certification_status": "PRODUCTION_READY"
  },
  "historical_context": {
    "sample_count": 48,
    "mean_yield_kg_ha": 746.4,
    "trend_slope_kg_ha_yr": 12.4
  },
  "validation": {
    "strategy_tier": "PRODUCTION_READY",
    "validation_mae": 549.67,
    "baseline_mae": 616.60,
    "gain_vs_baseline_pct": 10.85
  },
  "uncertainty": {
    "is_available": true,
    "empirical_p10_kg_ha": 571.2,
    "empirical_p90_kg_ha": 932.1
  },
  "monitoring": {
    "status": "HEALTHY",
    "prediction_drift_psi": 0.0,
    "post_outcome_evaluation_status": "EVALUATION_AVAILABLE"
  },
  "scenarios": [...],
  "comparison_matrix": {
    "scenario_headers": [...],
    "rows": [...],
    "disclaimer": "..."
  },
  "limitations": [...]
}
```

### 3.2 Backward Compatibility Assessment
All modifications made during Day 33 (e.g. security headers, column case normalization) are strictly additive or internal hardening. Zero existing JSON keys were removed or renamed in public schemas.
