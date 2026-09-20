# Day 28: Operational Failure Diagnostics & Error Categorization

## 1. Structured Error Categorization

To enable operators to diagnose failures without opening code, the platform structures operational errors into standard categories:

| Error Category | HTTP Code | Root Cause / Trigger | Remediation Guidance |
| :--- | :--- | :--- | :--- |
| **`INPUT_VALIDATION_ERROR`** | 422 | Missing required parameters, invalid year/numeric ranges, or malformed types. | Verify client payload conforms to FastAPI/Pydantic schemas. |
| **`CERTIFICATION_REJECTION`** | 422 | Crop is categorized as `UNSUPPORTED` or requested outside governed operational limits. | Check crop registration in Strategy Registry (`/api/observability/registry`). |
| **`UNSUPPORTED_CROP`** | 422 | Crop name not among the 29 verified crops. | Check valid crops list in `/api/filters`. |
| **`MODEL_INTEGRITY_FAILURE`** | 500 | Model artifact missing on disk or SHA-256 hash mismatch. | Inspect `Models/` directory and restore verified artifact. |
| **`DATASET_UNAVAILABLE`** | 500 | `Datasets/processed/agricultural_panel.csv` inaccessible or altered. | Check file permissions and canonical dataset path. |
| **`STRATEGY_LOOKUP_FAILURE`** | 500 | Strategy registry configuration file missing or corrupted. | Restore `Models/multicrop/forecast_strategy_registry.json`. |
| **`INFERENCE_ERROR`** | 500 | Unhandled exception during scikit-learn pipeline inference. | Check feature input shapes, NaNs, and pipeline compatibility. |
| **`INTERNAL_SERVER_ERROR`** | 500 | Unexpected Python runtime exception. | Inspect `/api/observability/errors` for full structured event details. |

---

## 2. Error Inspection & Log Access

Operators can inspect active and recent error logs at:
- **UI Route**: `/observability` (under the "Operational Events" section)
- **API Endpoint**: `GET /api/observability/errors`

The endpoint returns:
- `total_events_logged`: Overall event count.
- `total_errors_count`: Error occurrences in the active buffer.
- `errors_by_category`: Breakdown by error type.
- `recent_events`: Structured log entries with timestamp, severity, endpoint, status code, request ID, and message.
