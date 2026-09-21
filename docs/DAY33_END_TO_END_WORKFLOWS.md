# Day 33: End-to-End User Workflows & Validation Matrix

## 1. Scope & Execution Strategy

Every production workflow was tested across:
1. **Governed Forecast Generation** (`POST /api/forecast/predict`)
2. **Cryptographic Provenance Retrieval** (`GET /api/forecast/provenance/{request_id}`)
3. **Audit Log Inspection** (`GET /api/forecast/audit?limit=20`)
4. **Decision Workspace Analysis** (`POST /api/workspace/analyze`)
5. **Decision Intelligence Synthesis** (`POST /api/decision/analyze` & `GET /api/decision/brief`)

Testing was conducted via automated suite `tests/test_end_to_end_acceptance.py` (13 tests, 100% pass).

---

## 2. Golden User Journeys (All 4 Commodities)

### 2.1 Commodity 1: Oilseeds (Governed ML Production)

- **Input Parameters**: `crop="Oilseeds"`, `state="Madhya Pradesh"`, `district="Indore"`, `year=2017`
- **Strategy Resolved**: Historical ML (`RandomForestRegressor`)
- **Certification Tier**: `PRODUCTION_READY`
- **Fallback Used**: `False`
- **Yield Forecast**: 767.70 kg/ha
- **Walk-Forward Validation**: MAE = 549.67 kg/ha (vs Baseline: 616.60 kg/ha, +10.85% gain)
- **Provenance Hash**: `SHA256:8051a5a5...`
- **Workspace Verification**:
  - Historical reference context: Empirical sample count = 48, mean = 746.4 kg/ha.
  - Uncertainty: Empirical P10 = 571.2 kg/ha, P90 = 932.1 kg/ha.
  - Scenarios: 4 archetypes generated (`status_quo`, `conservative_improvement`, `moderate_improvement`, `stress_scenario`).
  - Comparison matrix: Contains baseline vs scenario projections with valid percentage changes.
  - Limitations: Complete non-causal disclaimer and methodology notice returned.

### 2.2 Commodity 2: Sugarcane (Conditional ML Production)

- **Input Parameters**: `crop="Sugarcane"`, `state="Uttar Pradesh"`, `district="Meerut"`, `year=2017`
- **Strategy Resolved**: Historical ML (`GradientBoostingRegressor`)
- **Certification Tier**: `CONDITIONAL_PRODUCTION`
- **Fallback Used**: `False`
- **Yield Forecast**: 61,048.20 kg/ha
- **Walk-Forward Validation**: MAE = 6,561.40 kg/ha (Baseline: 6,432.10 kg/ha, -2.01% gain)
- **Provenance Hash**: `SHA256:5896ff91...`
- **Workspace Verification**:
  - Generates conditional production governance notices.
  - Successfully explores acreage and input scenarios without crashing.

### 2.3 Commodity 3: Rice (Baseline Production)

- **Input Parameters**: `crop="Rice"`, `state="Punjab"`, `district="Ludhiana"`, `year=2017`
- **Strategy Resolved**: Historical District Mean / Persistence
- **Certification Tier**: `BASELINE_PRODUCTION`
- **Fallback Used**: `False`
- **Yield Forecast**: 4,480.32 kg/ha
- **Walk-Forward Validation**: MAE = 310.28 kg/ha, RMSE = 418.88 kg/ha, R² = 0.7866
- **Provenance Hash**: Generated and cryptographically logged
- **Workspace Verification**:
  - Non-ML baseline correctly acknowledged in validation section.
  - Uncertainty marked explicitly as unavailable (`is_available: false`) with reason.
  - Historical trend slope: +88.93 kg/ha/yr.

### 2.4 Commodity 4: Wheat (Baseline Production)

- **Input Parameters**: `crop="Wheat"`, `state="Haryana"`, `district="Karnal"`, `year=2017`
- **Strategy Resolved**: Historical District Mean / Persistence
- **Certification Tier**: `BASELINE_PRODUCTION`
- **Fallback Used**: `False`
- **Yield Forecast**: 4,643.15 kg/ha
- **Walk-Forward Validation**: MAE = 344.91 kg/ha, RMSE = 482.14 kg/ha, R² = 0.7612
- **Provenance Hash**: Generated and logged
- **Workspace Verification**:
  - Consistent baseline persistence metrics across all workspace sub-elements.

---

## 3. Negative & Boundary Workflows

| Scenario | Input Tested | Expected Response | Verified Code & Message |
| :--- | :--- | :--- | :--- |
| **Unsupported Commodity** | `crop="Barley"` | 400 Bad Request | `400: UNSUPPORTED_CROP: Commodity 'Barley' is not supported.` |
| **Unsupported District** | `crop="Oilseeds", district="Atlantis"` | 400 Bad Request | `400: District 'Atlantis' not found in certified registry.` |
| **Missing Crop Field** | `{"state": "Punjab", "district": "Ludhiana"}` | 422 Unprocessable | Standard Pydantic field validation error |
| **Missing District Field** | `{"crop": "Rice", "state": "Punjab"}` | 400 Bad Request | `400: Field 'district' is required for district-level forecasts.` |
| **Invalid Year Type** | `{"year": "nineteen-ninety"}` | 422 Unprocessable | Pydantic integer type parsing validation |
| **Malformed JSON Payload** | `"{invalid json..."` | 400 / 422 Bad Request | Fast JSON parser error containment |
| **Non-Existent Route** | `GET /api/forecast/nonexistent` | 404 Not Found | Fast 404 response without stack trace |
| **Unsupported Method** | `PUT /api/forecast/predict` | 405 Method Not Allowed | Fast 405 with allowed headers |
| **Unharvested Future Year** | `year=2026` | 200 OK with `EVALUATION_UNAVAILABLE` | Post-outcome evaluation flagged as unavailable; zero fake data |
