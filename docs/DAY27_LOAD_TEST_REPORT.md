# Day 27: Concurrency Load Test & Scaling Report

## 1. Concurrency Testing Scope & Architecture

- **Endpoint Tested**: `POST /api/forecast/predict` (Production ML Path: Oilseeds, Punjab, Ludhiana).
- **Concurrency Tiers Evaluated**: 1, 5, 10, 25, 50 concurrent worker threads.
- **Sample Count**: 100 requests per tier (500 total requests evaluated).
- **Concurrency Tool**: Python `ThreadPoolExecutor` load generator with monotonic per-request timing.

---

## 2. Concurrency Load Test Results

| Concurrency | Requests | Success | Errors | Error Rate | Throughput (RPS) | Min (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) | Determinism ($\Delta=0$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 100 | 100 | 0 | 0.0% | **6.97** | 72.24 | 120.73 | 228.46 | 254.32 | 256.57 | **VERIFIED** |
| **5** | 100 | 100 | 0 | 0.0% | **6.78** | 145.29 | 688.68 | 1099.40 | 1117.13 | 1120.07 | **VERIFIED** |
| **10** | 100 | 100 | 0 | 0.0% | **6.09** | 453.81 | 1644.27 | 2037.98 | 2328.26 | 2557.88 | **VERIFIED** |
| **25** | 100 | 100 | 0 | 0.0% | **8.43** | 206.17 | 2709.87 | 3303.03 | 3337.41 | 3338.87 | **VERIFIED** |
| **50** | 100 | 100 | 0 | 0.0% | **6.95** | 325.86 | 7159.38 | 8190.62 | 8225.67 | 8225.76 | **VERIFIED** |

---

## 3. Scaling & Queue Dynamics Analysis

### Throughput Stability
- Across all concurrency levels (1 through 50), the single-process ASGI backend sustained a consistent throughput of **6.09 to 8.43 RPS**.
- Zero HTTP errors (0/500 requests failed), with a 100.0% completion rate across all concurrency tiers.

### Queueing Behavior in Single-Worker Setup
- In a single Uvicorn worker process running synchronous model inference inside the event loop, requests queue linearly behind ongoing CPU execution.
- Consequently, latency scales proportionally with concurrency depth ($T_{\text{wait}} \approx C \times T_{\text{infer}}$).
- For multi-core production scale deployments, running multiple Uvicorn worker processes (e.g. `uvicorn --workers 4` or Gunicorn/Uvicorn process clusters) distributes incoming concurrent connections across distinct CPU cores, avoiding event loop serialization.

---

## 4. Performance Classification

- **Throughput Profile**: **STEADY THROUGHPUT** (~7-8 RPS on single CPU worker).
- **Reliability Profile**: **100% AVAILABILITY** (0 dropped requests, 0 unhandled exceptions).
- **Concurrency Regime**: **QUEUE-LIMITED LATENCY** (Predictable linear queueing under high concurrency on single worker).
- **Governance Behavior**: **100% AUDIT INTEGRITY** (Every concurrent request generated a distinct cryptographic provenance record and audit entry).
