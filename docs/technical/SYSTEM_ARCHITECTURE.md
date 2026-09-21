# System Architecture Specification

## 1. Network Topology & Container Infrastructure

The production system runs as a multi-container Docker Compose deployment fronted by an Nginx reverse proxy:

```
[Incoming User Request: Port 80]
              |
              v
       +---------------+
       | Nginx Reverse |
       |     Proxy     |
       +---------------+
        /             \
       / (Static UI)   \ (API /api/v1/*)
      v                 v
+------------+   +-------------------------------+
|  Frontend  |   |        Backend Service        |
|  Container |   |  (FastAPI on internal :8000)  |
+------------+   +-------------------------------+
                                |
                   [Local Host-Volume Storage]
                   |-- Models/ (Serialized Estimators & Registries)
                   |-- Datasets/processed/ (Canonical Long Panel)
                   +-- Datasets/metadata/ (Audit Logs & Telemetry)
```

---

## 2. Component Interaction & Security Isolation

1. **Port Internalization**: The FastAPI backend port (`8000`) is not published to the host machine. It is exposed exclusively to the internal Docker network `ai-agri-network`, eliminating host port bypass attacks.
2. **Reverse Proxy Routing**: Nginx serves compiled static frontend assets and reverse-proxies `/api/v1/*` requests to `http://backend:8000`, injecting OWASP security headers (`X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`).
3. **Fail-Closed Readiness Probing**: The `/ready` health probe executes startup validation on model pipelines, strategy registries, and canonical panel files. If any required asset is absent, it responds with `HTTP 503 Service Unavailable`, preventing ungrounded inference serving.
