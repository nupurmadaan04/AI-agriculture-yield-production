# Day 28: Runtime Telemetry & Performance Metrics

## 1. Principles of Telemetry Measurement

The platform adheres to strict measurement standards:
1. **Zero Fabrication**: All counts, durations, and rates are directly observed from actual requests and processes.
2. **Cold-Start Integrity**: When no requests have been received, the UI and API explicitly communicate `"No runtime observations yet."` rather than presenting fake zeros or hardcoded placeholders.
3. **Statistical Integrity**: Latency percentiles (P50, P90, P95, P99) are computed dynamically from sorted arrays of real request durations in milliseconds.

---

## 2. Runtime Metrics Specifications

| Metric | Computation | Source |
| :--- | :--- | :--- |
| **Total Requests** | $N = \text{len}(\text{buffer})$ | Request ring buffer |
| **Success Rate** | $(\text{count}_{\text{status } < 400} / N) \times 100$ | Observed HTTP status codes |
| **Error Rate** | $(\text{count}_{\text{status } \ge 500} / N) \times 100$ | Observed HTTP status codes |
| **Rejection Rate** | $(\text{count}_{\text{status } = 422} / N) \times 100$ | Governed domain validation & certification rejections |
| **Throughput (RPS)** | $N / \text{uptime\_seconds}$ | Time since process initialization |
| **P50 Latency** | 50th percentile of duration list | `time.perf_counter()` per request |
| **P90 Latency** | 90th percentile of duration list | `time.perf_counter()` per request |
| **P95 Latency** | 95th percentile of duration list | `time.perf_counter()` per request |
| **P99 Latency** | 99th percentile of duration list | `time.perf_counter()` per request |
| **Process CPU %** | Process CPU utilization | `psutil.Process().cpu_percent()` |
| **Host Memory %** | Virtual memory utilization | `psutil.virtual_memory().percent` |
| **Process Memory RSS** | Resident set size (MB) | `psutil.Process().memory_info().rss` |

---

## 3. Forecast Operations Metrics

Forecast operations track application runtime activity, distinct from agricultural population statistics:
- **Total Forecast Invocations**: Total governed predictions requested.
- **Strategy Distribution**: Real-time breakdown of invocations across `PRODUCTION_READY` (Oilseeds RF), `CONDITIONAL_PRODUCTION` (Sugarcane GB), and `BASELINE_PRODUCTION` (Historical District Mean / Persistence).
- **Crop Distribution**: Runtime frequency of specific crop requests.
- **Fallback Invocations**: Invocations routed to fallback mechanisms (e.g., district mean) when feature inputs or geographic coverage are sparse.
