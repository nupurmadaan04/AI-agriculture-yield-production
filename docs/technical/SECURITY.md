# Security Architecture & Hardening Standards

## 1. Container Hardening & Execution Safety

1. **Non-Root Execution**: Backend and frontend containers run under unprivileged user accounts (`appuser`, UID 10001), preventing container escape attacks.
2. **Read-Only Code Mounts**: Application code files in `/app/` are mounted read-only, preventing in-memory tampering or unauthorized runtime modifications.
3. **Network Isolation**: Backend port 8000 is internal to Docker network; only port 80 (Nginx) is exposed to host.

---

## 2. API Security & Input Sanitization

1. **OWASP HTTP Headers**: Injected via Nginx:
   - `X-Frame-Options: DENY`
   - `X-Content-Type-Options: nosniff`
   - `Referrer-Policy: strict-origin-when-cross-origin`
2. **Pydantic Validation**: Strict regex validation on crop names, state names, and district identifiers prevents SQL, shell, or JSON injection vectors.
3. **Secret Scanning**: Zero hardcoded API keys, tokens, or credentials exist in code or repository configuration.
