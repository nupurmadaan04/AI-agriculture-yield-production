# Production Deployment & Disaster Readiness Specification

## 1. Multi-Container Orchestration

Orchestrated via `docker-compose.yml`:
- **Frontend Service**: Multi-stage build on Node.js 20-alpine; compiled bundle served via Nginx.
- **Backend Service**: Python 3.11-slim container running Uvicorn ASGI server with 4 worker processes.
- **Reverse Proxy**: Nginx Alpine container routing port 80 to internal frontend and backend endpoints.

---

## 2. Health & Readiness Probes

- **Liveness Probe** (`/live`): Responds `HTTP 200 {"status": "alive"}` for container health monitoring.
- **Readiness Probe** (`/ready`): Fails closed (`HTTP 503`) if any serialized model, dataset panel, or strategy registry is missing or corrupt.

---

## 3. Disaster Recovery & Backup Runbooks

Documented in `docs/DAY34_RUNBOOK.md`:
- Database persistence classification: `Datasets/processed/agricultural_panel.csv` and `Models/` classified as Gold Assets.
- Zero-Downtime rollbacks: Strategy registry updates can revert dynamically by updating `active_strategy` pointers without recompiling Docker images.
