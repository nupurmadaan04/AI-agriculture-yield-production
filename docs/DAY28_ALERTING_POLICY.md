# Day 28: Operational Alerting Policy & Thresholds

## 1. Alerting Philosophy

Operational alerting in Day 28 is strictly configured around system reliability, data availability, and cryptographic integrity.

**Important Distinctions**:
- Operational alerts monitor runtime errors, latency spikes, and system resource exhaustion.
- Operational alerts **do not** judge scientific model validity (which is governed by expanding walk-forward validation and certification guards).
- Configured thresholds are documented and exposed; unconfigured alerts are never silently fabricated.

---

## 2. Configured Operational Alert Rules

| Alert Rule | Metric / Condition | Default Threshold | Severity | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`ALT-ERROR-RATE`** | `error_rate_pct > threshold` | 5.0% | `CRITICAL` | Triggered when 5xx HTTP response rate over the active window exceeds 5%. |
| **`ALT-LATENCY-P95`** | `p95_latency_ms > threshold` | 500.0 ms | `WARNING` | Triggered when 95th percentile request latency exceeds 500ms. |
| **`ALT-MODEL-INTEGRITY`** | `failed_models_count > 0` | 0 failures | `CRITICAL` | Triggered if any registered model artifact fails SHA-256 cryptographic verification. |
| **`ALT-DATASET-INTEGRITY`** | `file_missing OR row_count != 71601` | 71,601 records | `CRITICAL` | Triggered if the canonical dataset file is missing or row count differs from 71,601. |
| **`ALT-SYSTEM-MEMORY`** | `system_memory_percent > threshold` | 85.0% | `WARNING` | Triggered when host RAM utilization exceeds 85%. |

---

## 3. Alert Lifecycle & Evaluation

Alerts are evaluated dynamically on each query to `/api/observability/alerts` or `/api/observability/summary`.
- **Active Alerts**: Displayed with amber or red semantic styling in the Observability Center.
- **Resolved Alerts**: Kept in the resolved list with normal baseline status and timestamp.
- **Zero-Alert State**: When all checks pass, the UI displays `"No active operational alerts."`
