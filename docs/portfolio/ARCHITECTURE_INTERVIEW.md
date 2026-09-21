# System Architecture Deep-Dive: System Design Interview Guide

This guide prepares candidates to defend the end-to-end software and data architecture of the platform during technical system design and MLOps engineering interviews.

---

## 1. High-Level Network & Container Topology

```
+---------------------------------------------------------------------------------------------------------+
|                                    PRODUCTION NETWORK TOPOLOGY                                          |
+---------------------------------------------------------------------------------------------------------+

                 [Public / Client Network]
                            |
                     (HTTP Port 80)
                            v
       +-----------------------------------------+
       |           Nginx Reverse Proxy           |
       |  (Alpine Linux Container | Rate Limiter) |
       |  - Injects OWASP Security Headers       |
       |  - Serves Static React Assets           |
       |  - Proxies /api/v1/* to Backend         |
       +--------------------+--------------------+
                            |
                 [Internal Docker Bridge]
                 (Network: ai-agri-network)
                            |
                 (Internal Port 8000 only)
                            v
       +-----------------------------------------+
       |         FastAPI Backend Service         |
       |     (Python 3.11-slim | Non-Root User)   |
       |  - Pydantic v2 Schema Validation        |
       |  - Strategy Router & Certification Guard|
       |  - Forecast Serving Engine (<50ms P95)  |
       |  - Cryptographic Provenance Logger      |
       |  - Population Stability Index Engine    |
       +--------------------+--------------------+
                            |
               [Persistent Storage Volumes]
               (Host Mount: Read-Only Models & Data; Read-Write Metadata)
               ├── Models/ (Serialized Estimators & Registries)
               ├── Datasets/processed/ (Canonical Panel CSV)
               └── Datasets/metadata/ (Append-Only Audit Logs & Telemetry)
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Step-by-Step Forecast Request Lifecycle

When a client submits an inference request: `POST /api/v1/forecast/predict`:

```
1. INGRESS & TLS TERMINATION (Nginx)
   • Nginx receives the request on port 80.
   • Injects security headers: X-Content-Type-Options: nosniff, X-Frame-Options: DENY.
   • Forwards /api/v1/* traffic to backend:8000 over the internal Docker network.

2. INPUT SANITIZATION & SCHEMA VALIDATION (FastAPI & Pydantic)
   • Request payload is parsed against Pydantic v2 schemas (`ForecastRequest`).
   • Strict type boundaries: `crop` (string enum), `state` (string), `district` (string), `year` (int: 2010..2030).
   • Injection defense: Strings containing path traversal characters (`..`, `/`, `\`) or SQL/script injections are rejected with HTTP 422.

3. STRATEGY REGISTRY LOOKUP (Certification Guard)
   • `ForecastRouter` queries `Models/multicrop/forecast_strategy_registry.json`.
   • Checks if `(crop, district)` combination is certified.
   • If crop is not model-ready, rejects immediately with HTTP 400 (`UNSUPPORTED_CROP`).

4. INFERENCE EXECUTION & SAFETY BOUNDS (Prediction Service)
   • Branch A (Oilseeds: PRODUCTION_READY):
     - Extracts pre-season lag features from memory-mapped feature store.
     - Executes `RandomForestRegressor.predict(X)`.
     - Queries 150 individual tree estimators to compute empirical P10–P90 uncertainty spread.
   • Branch B (Sugarcane: CONDITIONAL_PRODUCTION):
     - Executes `GradientBoostingRegressor.predict(X)`.
     - Checks 3-sigma variance bound: If $|\hat{y} - \mu_{\text{dist}}| > 3\sigma_{\text{dist}}$, automatically falls back to $\mu_{\text{dist}}$ and flags response (`strategy_used: "FALLBACK_3SIGMA"`).
   • Branch C (12 Staples: BASELINE_PRODUCTION):
     - Retrieves Historical District Mean ($\mu_{\text{dist}}$). Uncertainty is set to district historical standard deviation.

5. CRYPTOGRAPHIC PROVENANCE HASHING (Provenance Service)
   • Computes SHA-256 fingerprint:
     $$\text{Hash} = \text{SHA256}(\text{Request UUID} \parallel \text{Crop} \parallel \text{District} \parallel \text{Year} \parallel \text{Model Version} \parallel \text{Dataset Hash})$$
   • Synchronously appends execution record to `Datasets/metadata/prediction_audit_log.csv`.

6. RESPONSE PAYLOAD EMISSION
   • Emits structured JSON response containing: `prediction_id`, `crop`, `district`, `year`, `predicted_yield_kg_ha`, `uncertainty_p10`, `uncertainty_p90`, `strategy_used`, `provenance_hash`, and execution latency (<50ms).
```

---

## 3. Reliability, Resilience & Failure Modes

### 3.1 Fail-Closed Readiness Probing
- **Endpoint**: `GET /api/v1/health/ready`
- **Logic**: During container initialization, the probe verifies physical existence and SHA-256 checksums of:
  1. `Datasets/processed/agricultural_panel.csv`
  2. `Models/multicrop/forecast_strategy_registry.json`
  3. Certified model pickle files (`forecasting_pipeline.pkl`)
- **Resilience Behavior**: If any core model or dataset is missing or corrupt, the probe fails closed with **`HTTP 503 Service Unavailable`**, preventing traffic routing to an ungrounded inference engine.

### 3.2 Liveness Probing
- **Endpoint**: `GET /api/v1/health/live`
- **Logic**: Lightweight process health check returning `HTTP 200 {"status": "alive"}` for container orchestrator restart monitoring.

### 3.3 Missing Data Fallback Hierarchy
If a forecast is requested for a valid district where recent lag-1 yield is missing:
$$\text{Primary ML} \longrightarrow \text{Historical District Mean} \longrightarrow \text{State Agro-Climatic Mean}$$
The fallback is transparently recorded in the response and audit log.

---

## 4. Security Hardening & Isolation

1. **Non-Root Containerization**: Dockerfile defines an unprivileged user:
   ```dockerfile
   RUN addgroup --system --gid 10001 appuser && adduser --system --uid 10001 --ingroup appuser appuser
   USER appuser
   ```
2. **Network Isolation**: Backend port 8000 is not mapped to the host (`ports: ["8000:8000"]` is prohibited); it is declared solely via `expose: ["8000"]` for internal Nginx communication.
3. **Read-Only Code Mounts**: Application code files in `/app/` are mounted read-only, preventing in-memory code tampering.
4. **Secret Scanning Invariants**: Zero hardcoded secrets, database passwords, or private API keys exist in the repository.
