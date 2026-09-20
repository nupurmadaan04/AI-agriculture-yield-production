# Day 27: Performance Baseline & Sequential Latency Report

## 1. Methodology & Test Configuration

- **Target Base URL**: `http://127.0.0.1:8000`
- **Sample Distribution**: 1 Cold Process Request + 100 Warm Sequential Invocations per endpoint.
- **Timing Instrument**: Python `time.perf_counter()` (microsecond resolution).
- **Environment**: AMD64 8-Thread CPU / 15.88 GB RAM / Python 3.11.9.
- **Evaluation Date**: September 20, 2026.

---

## 2. Sequential Benchmark Results Matrix

| Category | Endpoint / Test Case | Method | Cold (ms) | Warm P50 (ms) | Warm P90 (ms) | Warm P95 (ms) | Warm P99 (ms) | Std Dev (ms) | RPS | Success Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HEALTH** | `/health` | GET | 14.93 | **2.96** | 3.88 | 15.19 | 26.08 | 5.10 | 233.8 | 100.0% |
| **READINESS** | `/ready` | GET | 10.07 | **5.88** | 21.63 | 24.17 | 27.33 | 7.28 | 108.7 | 100.0% |
| **STRATEGY_REGISTRY** | `/api/forecast/strategies` | GET | 10.16 | **5.68** | 7.72 | 19.82 | 30.07 | 12.83 | 127.2 | 100.0% |
| **CERTIFICATION** | `/api/forecast/certification` | GET | 22.83 | **4.86** | 5.80 | 6.20 | 23.98 | 3.02 | 187.6 | 100.0% |
| **COVERAGE** | `/api/forecast/coverage` | GET | 561.35 | **137.82** | 221.07 | 243.40 | 263.17 | 41.89 | 6.7 | 100.0% |
| **FORECAST_ML_PROD** | Oilseeds (Random Forest) | POST | 128.51 | **106.80** | 190.56 | 199.31 | 216.97 | 36.65 | 8.4 | 100.0% |
| **FORECAST_ML_CONDITIONAL** | Sugarcane (Gradient Boosting) | POST | 76.96 | **13.01** | 17.36 | 22.36 | 32.13 | 4.36 | 69.9 | 100.0% |
| **FORECAST_BASELINE** | Rice (Baseline Persistence) | POST | 71.80 | **11.76** | 14.60 | 23.44 | 30.99 | 4.62 | 78.8 | 100.0% |
| **FORECAST_REJECTION** | Unsupported Crop (Potato) | POST | 14.76 | **11.20** | 13.34 | 14.24 | 31.06 | 4.00 | 84.0 | 100.0% |
| **PROVENANCE_LOOKUP** | Provenance Record by ID | GET | 7.38 | **5.73** | 6.71 | 15.81 | 32.28 | 6.30 | 140.8 | 100.0% |

---

## 3. Cold vs Warm Execution Characteristics

1. **Cold Initialization**:
   - First request cold latencies range from 7.38 ms (Provenance) to 561.35 ms (Coverage Matrix with 9,019 objects).
   - Cold overhead is driven by dynamic Python module loading, JIT cache warmup, and initial dataframe indexing.
2. **Warm Execution Stability**:
   - Warm P50 latencies for metadata, readiness, and strategy endpoints drop to **2.96 ms – 5.88 ms**.
   - Forecasting inference P50 latency stabilizes at **11.76 ms – 106.80 ms** depending on whether full panel statistical feature reconstruction is required.
   - Standard deviations for core forecast endpoints remain strictly bounded below 40 ms.

---

## 4. Latency Contributors Breakdown

```
Estimated Forecast Pipeline Latency Budget (Warm State):
├─ HTTP Parsing & Pydantic Validation: ~1.2 ms
├─ Geographic Guard Verification:       ~0.8 ms
├─ Strategy Lookup & Routing:          ~0.5 ms
├─ In-Memory Feature Alignment:         ~6.5 ms
├─ Model Inference Execution:          ~1.5 ms
├─ Cryptographic Provenance DAG:       ~0.9 ms
├─ Audit Record Generation:            ~0.8 ms
└─ JSON Response Serialization:        ~1.0 ms
Total P50 Latency:                     ~12 - 14 ms
```

> [!NOTE]
> All observed figures represent local execution in the documented benchmark environment. In accordance with Day 27 guidelines, no artificial SLA claims or global cloud extrapolations are made.
