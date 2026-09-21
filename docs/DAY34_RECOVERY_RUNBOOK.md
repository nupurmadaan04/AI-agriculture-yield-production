# DAY 34 — Production Incident Recovery & Disaster Runbook

## Overview

This practical runbook specifies the 12-step incident recovery workflow for the Agricultural Forecasting & Decision Intelligence platform. Use this runbook whenever the service exhibits degraded health, unresponsiveness, container failure, or potential state divergence.

---

## 12-Step Operational Recovery Workflow

### Step 1: Detect Failure
Monitor alerting channels, user reports, or automated uptime probes.
```bash
# Check if reverse proxy responds on port 80
curl -I -s http://localhost/ | head -n 5
```
*Criteria*: HTTP 502 Bad Gateway, 503 Service Unavailable, connection refused, or connection timeout indicates failure condition.

---

### Step 2: Check Health (Liveness)
Probe the API process liveness endpoint to check whether ASGI workers are responding.
```bash
# Via Nginx reverse proxy
curl -s http://localhost/api/health | python -m json.tool
```
*Expected*: HTTP 200 with `{"status": "ok", "service": "agricultural-intelligence-api", ...}`.  
*Action on Failure*: Process has terminated, frozen in a loop, or is blocked on I/O. Proceed immediately to Step 4.

---

### Step 3: Check Readiness (Fail-Closed Subsystems)
Probe the deep readiness endpoint to check required data assets, model registries, and engines.
```bash
# Via Nginx reverse proxy
curl -s http://localhost/api/ready | python -m json.tool
```
*Expected*: HTTP 200 with `{"status": "ready", "components": {"dataset": "ready", "forecast_strategy_registry": "ready", ...}}`.  
*Action on Failure*: HTTP 503 with `"status": "not_ready"`. Identify which specific subsystem flagged `not_ready` (e.g., `dataset`, `forecast_strategy_registry`, `forecast_router`).

---

### Step 4: Inspect Container Status
Determine state of Docker containers and host processes.
```bash
# Inspect container health and exit codes
docker-compose ps

# Verify process status on host if running directly
Get-Process -Name "*python*", "*uvicorn*", "*node*", "*nginx*"
```
*Expected*: All services (`frontend`, `backend`) in `Up` (healthy) state.  
*Action on Failure*: Note exit code (e.g., 137 OOM killed, 1 unhandled exception).

---

### Step 5: Inspect Logs
Examine tail logs for structured errors, stack traces, and volume mount issues.
```bash
# Backend ASGI error trace (tail last 100 lines)
docker-compose logs --tail=100 backend

# Frontend / Nginx access and error logs
docker-compose logs --tail=50 frontend

# Inspect operational telemetry log
tail -n 25 Datasets/metadata/operational_telemetry.jsonl
```
*Look for*: `FileNotFoundError` (missing volumes), `ValidationError`, `PermissionError`, or missing dependencies.

---

### Step 6: Restart Affected Service
Execute isolated or complete service recovery based on failure scope.

#### Option A: Backend Service Restart (Single Container)
```bash
docker-compose restart backend
sleep 8
```

#### Option B: Full Stack Restart
```bash
docker-compose down
docker-compose up -d
sleep 12
```

#### Option C: Clean Rebuild & Restart (Code/Dependency Change)
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
sleep 15
```

---

### Step 7: Verify Readiness
Confirm that all subsystems are initialized and report ready post-restart.
```bash
curl -s http://localhost/api/ready | python -m json.tool
```
*Verification*: Must return HTTP 200 and `"status": "ready"`. If HTTP 503 persists, re-inspect logs (Step 5) for corrupted volumes or permissions.

---

### Step 8: Run Golden Forecast
Execute governed analytical requests across the 4 core commodities to confirm deterministic model serving.
```bash
# Automated golden verification suite
pytest tests/test_deployment_verification.py -v -k "test_deterministic_restart"

# Manual golden check (Oilseeds)
curl -s -X POST http://localhost/api/forecast/predict \
  -H "Content-Type: application/json" \
  -d '{"crop":"Oilseeds","state":"Madhya Pradesh","district":"Indore","forecast_year":2017}' \
  | python -m json.tool
```
*Expected*: Valid prediction returned with `strategy="RandomForestRegressor"`, `certification_status="CERTIFIED_PRODUCTION"`.

---

### Step 9: Verify Provenance
Verify cryptographic SHA-256 digital provenance chain for the forecast.
```bash
# Fetch provenance record using request_id from Step 8
curl -s http://localhost/api/forecast/provenance/<REQUEST_ID> | python -m json.tool
```
*Expected*: Valid `provenance_hash` prefixed with `SHA256:`, matching historical reference hash for identical inputs.

---

### Step 10: Verify Audit
Ensure append-only audit logging succeeded without data loss.
```bash
# Check last line of prediction audit log
tail -n 3 Datasets/metadata/prediction_audit_log.csv
```
*Expected*: Record contains timestamp, `request_id`, crop, state, district, strategy, prediction, and provenance hash.

---

### Step 11: Verify Frontend
Confirm user-facing web shell, routing, and SPA navigation load properly through Nginx.
```bash
# Check SPA HTML index
curl -s -I http://localhost/ | grep "HTTP/1.1 200 OK"

# Check SPA deep route fallback
curl -s -I http://localhost/prediction-explorer | grep "HTTP/1.1 200 OK"
curl -s -I http://localhost/decision-workspace | grep "HTTP/1.1 200 OK"
```
*Browser Verification*: Open `http://localhost/` in browser, navigate through Prediction Explorer, Decision Intelligence, and Decision Workspace.

---

### Step 12: Declare Recovery
Once Steps 1–11 pass with 100% success:
1. Log operational incident recovery timestamp and duration to `Datasets/metadata/operational_telemetry.jsonl`.
2. Update system status badge to `OPERATIONAL`.
3. Notify stakeholders and release incident summary.

---

## Escalation Matrix & Root-Cause Actions

| Failure Mode | Diagnosis | Remediation |
|---|---|---|
| Model registry missing | Readiness reports `forecast_strategy_registry: not_ready` | Restore `Models/multicrop/forecast_strategy_registry.json` from git/backup |
| Dataset corrupted | Readiness reports `dataset: not_ready` | Re-copy canonical `Datasets/final_agricultural_data_cleaned.csv` |
| Port conflict on 80 | Nginx fails to bind host port 80 | Terminate conflicting process (`netstat -ano \| findstr :80`) |
| Backend 500 error | Internal unhandled exception | Check sanitized log; inspect `backend/main.py` exception middleware |
