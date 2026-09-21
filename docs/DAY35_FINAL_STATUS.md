# DAY 35 — FINAL STATUS

PASS WITH LIMITATIONS

## Summary Scorecard

Homepage:
PASS

Navigation:
PASS

Forecast workflow:
PASS

Prediction Explorer:
PASS

Monitoring:
PASS

Decision Intelligence:
PASS

Decision Workspace:
PASS

Provenance:
PASS

Accessibility:
PASS

Responsive:
PASS

Demo workflow:
PASS

Scientific terminology:
PASS

Stale claim audit:
PASS

Frontend build:
PASS

Backend regression:
PASS

Scientific regression:
PASS

Actual test count:
89 passed across Day 33, Day 34, and Day 35 production acceptance suites (574 total tests collected across the platform repository)

## Known Limitations

1. **Local Containerization Scope**: The platform is containerized and validated for production-style deployment with Nginx reverse proxy; cloud hosting (AWS/GCP/Azure) is an architecture preparation and not independently verified on a remote cloud provider.
2. **File-Based Audit Persistence**: Prediction audit logs (`Datasets/metadata/prediction_audit_log.csv`) and operational telemetry operate as append-only host volume files rather than a replicated, managed cloud RDBMS.
3. **Historical Data Horizon**: Canonical dataset represents historical district-level observations spanning 1966–2017 (ICRISAT). Historical outcome evaluation for future/unharvested years (e.g. 2026) is explicitly flagged as `EVALUATION_UNAVAILABLE`.
4. **Attribution Scope**: Feature attribution represents mathematical sensitivity within the trained model space (Marginal Reference Perturbation / Tree SHAP) and must not be interpreted as causal or biological certainty. Deterministic baselines (Rice, Wheat) do not have ML feature attribution.

---

## Detailed Test Breakdown

### Day 35 UI Contracts & Accessibility Suite (`tests/test_day35_ui_contracts.py`)
- `test_strategy_transparency_oilseeds_governed_ml`: PASSED
- `test_strategy_transparency_sugarcane_conditional_ml`: PASSED
- `test_strategy_transparency_rice_baseline`: PASSED
- `test_strategy_transparency_wheat_baseline`: PASSED
- `test_decision_workspace_uncertainty_contract_ml`: PASSED
- `test_decision_workspace_uncertainty_contract_baseline`: PASSED
- `test_decision_workspace_semantic_entity_typing`: PASSED
- `test_webshell_skip_to_content_link`: PASSED
- `test_breadcrumbs_aria_landmarks`: PASSED
- `test_navbar_accessible_controls`: PASSED
- `test_zero_user_facing_internal_day_labels`: PASSED
- `test_zero_ungrounded_marketing_claims`: PASSED
- **Total Day 35 UI Contracts**: 12 / 12 PASSED (100%) in 11.24s

### Day 34 Deployment Verification Suite (`tests/test_deployment_verification.py`)
- 13 / 13 PASSED in 5.93s

### Combined Acceptance & Production Regression Suite (Days 33–35)
- `tests/test_end_to_end_acceptance.py`: 13 / 13 PASSED
- `tests/test_security_acceptance.py`: 20 / 20 PASSED
- `tests/test_failure_resilience.py`: 6 / 6 PASSED
- `tests/test_api_contracts.py`: 12 / 12 PASSED
- `tests/test_provenance_chain.py`: 5 / 5 PASSED
- `tests/test_concurrency_safety.py`: 8 / 8 PASSED
- `tests/test_deployment_verification.py`: 13 / 13 PASSED
- `tests/test_day35_ui_contracts.py`: 12 / 12 PASSED
- **Total Combined Production Acceptance Suite**: 89 / 89 PASSED (100%)

### Frontend Production Build
- Command: `npm run build`
- Result: Clean exit (code 0) in 7.89s
- Bundles: `dist/index.html` (1.31 kB), `dist/assets/index-DS-ncBtH.css` (70.78 kB), `dist/assets/index-6e0IyN8X.js` (1,450.09 kB)
- TypeScript checks: 0 errors
