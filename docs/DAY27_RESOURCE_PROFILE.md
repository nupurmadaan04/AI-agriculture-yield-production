# Day 27: System Resource Profile & Memory Analysis

## 1. Overview & Measurement Methodology

System resource behavior was profiled during sequential benchmarking and concurrent load testing across all 5 concurrency levels (100 requests per tier).

- **Monitoring Scope**: CPU Utilization (%), Resident Set Size (RSS Memory MB), Memory Growth / Delta ($\Delta$ MB), Process Stability.
- **Instrument**: Python `psutil` sampling process memory before and after test execution runs.

---

## 2. Resource Utilization Summary

| Test Phase | Concurrency | Requests | CPU Util (%) | Memory RSS (MB) | Memory Delta ($\Delta$ MB) | Process Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Idle Baseline** | 0 | 0 | 1.2% | 21.74 MB | 0.00 MB | Healthy |
| **Tier 1 Load** | 1 | 100 | 61.9% | 23.76 MB | +2.02 MB | Stable |
| **Tier 2 Load** | 5 | 100 | 47.0% | 23.86 MB | +0.11 MB | Stable |
| **Tier 3 Load** | 10 | 100 | 52.0% | 24.12 MB | +0.27 MB | Stable |
| **Tier 4 Load** | 25 | 100 | 41.0% | 24.34 MB | +0.21 MB | Stable |
| **Tier 5 Load** | 50 | 100 | 46.9% | 24.36 MB | +0.02 MB | Stable |

---

## 3. Key Observations & Memory Stability

1. **Absence of Memory Leaks**:
   - Initial memory footprint grew by ~2.02 MB during the initial cache population phase (storing unique query stats in `_stats_cache` and caching the strategy response).
   - Across subsequent heavy concurrency batches (Tiers 2 through 5, comprising 400 additional requests), total resident memory increased by only **0.60 MB total** (from 23.76 MB to 24.36 MB), asymptoting toward complete stability ($\Delta = +0.02\text{ MB}$ at Concurrency 50).
2. **CPU Utilization**:
   - CPU utilization stayed smoothly between **41.0% and 61.9%** under continuous request generation.
   - No runaway threads, deadlocks, or event-loop starvation occurred.
3. **Container & Process Restarts**:
   - Total process crashes or unexpected restarts: **0**.
   - Process count remained strictly constant throughout the test cycle.

---

## 4. Resource Assessment

- **Classification**: **RESOURCE STABLE**
- **Memory Growth Behavior**: **BOUNDED & ASYMPTOTIC**
- **Process Stability**: **100% UNINTERRUPTED**
