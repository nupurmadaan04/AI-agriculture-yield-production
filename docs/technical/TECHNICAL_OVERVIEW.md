# Technical Overview: Agricultural Forecasting & Decision Platform

## 1. System Purpose & Core Mission

The Agricultural Forecasting & Decision Intelligence Platform provides evidence-governed, district-level crop yield forecasts and multi-objective decision briefs across India. It addresses the fundamental challenges of spatial agro-ecological heterogeneity, climate shock regime shifts, and data leakage by enforcing rigorous expanding walk-forward validation and fail-safe strategy governance.

---

## 2. Core Operational Modules

```
+---------------------------------------------------------------------------------------------------------+
|                                    PLATFORM COMPONENT HIERARCHY                                         |
+--------------------------+------------------------------------------------------------------------------+
| Module                   | Operational Responsibilities                                                 |
+--------------------------+------------------------------------------------------------------------------+
| **Data Engine**          | ICRISAT/DES normalization, zero-leakage lag feature pipeline, panel store.    |
| **Strategy Router**      | Runtime governance enforcing certified strategy assignments across crops.   |
| **Forecast Engine**      | High-performance sub-50ms inference runtime with 3-sigma variance bounds.    |
| **XAI Engine**           | Marginal Reference Perturbation Attribution and Tree SHAP sensitivity logic. |
| **Observability Engine** | Population Stability Index drift tracker, post-harvest error decomposition.  |
| **Provenance Engine**    | Cryptographic SHA-256 digital execution signature generator.                 |
| **Decision Workspace**   | Non-prescriptive multi-evidence synthesis and bounded scenario simulation.   |
| **Web UI Shell**         | WCAG 2.1 AA accessible React/Vite interface served via Nginx reverse proxy.  |
+--------------------------+------------------------------------------------------------------------------+
```

---

## 3. Technology Stack

- **Backend Runtime**: Python 3.11.9, FastAPI, Uvicorn, Scikit-learn 1.6.1, Pandas 2.2.3, NumPy 2.2.3, SciPy.
- **Frontend Runtime**: Node.js v24.14.0, React 18, TypeScript, Tailwind CSS, Lucide Icons, Vite.
- **Containerization**: Docker Engine, Docker Compose, Nginx (Alpine Linux, reverse proxy on port 80).
- **Security Posture**: Non-root execution (`appuser`), internal Docker bridge networking, OWASP security headers.
