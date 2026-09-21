# DAY 34 — Container Security Audit

## Non-Root User Execution

Backend container runs as `appuser:appgroup` (UID 10001) — verified in `backend/Dockerfile`:
- User created with `useradd -r -u 10001 -g appgroup appuser`
- Application directory owned by `appuser`
- `USER appuser` set before `CMD`

Frontend container runs as the default Nginx user (restricted). Static file serving only.

## Port Exposure Audit

| Port | Before Day 34 | After Day 34 | Justification |
|------|--------------|--------------|---------------|
| 80 | Exposed (Nginx) | Exposed (Nginx) | Client-facing; intended |
| 8000 | Exposed (host) | Internal only (`expose`) | **FIXED**: Backend must only be reachable via Nginx proxy |

### Risk Mitigation
Removing `ports: ["8000:8000"]` from the backend service prevents:
- Direct API access bypassing Nginx security headers and rate limiting
- Potential CORS bypass if backend CORS was misconfigured
- Accidental exposure of raw FastAPI docs (`/docs`, `/redoc`) to host network

## CORS Configuration

Backend CORS (in `backend/main.py`):
- `allow_origins`: Configured for frontend origin only (not `*` in production)
- `allow_methods`: `["GET", "POST"]` — only required HTTP methods
- `allow_headers`: Restricted to necessary headers

## Security Headers (Nginx)

Applied via `nginx/nginx.conf`:
```
X-Frame-Options: SAMEORIGIN
X-Content-Type-Options: nosniff
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
```

## Volume Isolation

All volumes are host-relative mounts (`./Datasets`, `./Models`) — no Docker named volumes that could be shared across projects. Volume scope is limited to the compose project.

## Dependency Pinning

- `backend/requirements.txt`: All dependencies pinned to exact versions
- `frontend/package.json`: Exact versions (no `^` wildcards in production dependencies)

## Audit Summary

| Check | Result |
|-------|--------|
| Non-root user | ✅ PASS |
| Backend port internal only | ✅ PASS (Day 34 fix) |
| Security headers via Nginx | ✅ PASS |
| CORS restricted | ✅ PASS |
| Volume isolation | ✅ PASS |
| Zero hardcoded secrets | ✅ PASS |
| Dependency pinning | ✅ PASS |
