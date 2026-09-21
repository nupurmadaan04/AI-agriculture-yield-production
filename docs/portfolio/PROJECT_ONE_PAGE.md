# Agricultural Intelligence Platform: Executive Brief

**A 60-Second Overview for Recruiters & Hiring Managers**

---

### The Problem
District-level agricultural forecasting in developing agricultural economies is vital for national food security, grain procurement, and drought relief. However, conventional machine learning models frequently fail in production due to target leakage (using harvest production to predict yield), random cross-validation failure under spatial-temporal autocorrelation, and severe out-of-distribution tail errors during climate shocks.

### The Solution
An **evidence-governed forecasting and decision intelligence platform** that combines strict pre-season leakage isolation, expanding walk-forward temporal validation, empirical model governance, explainability, real-time drift monitoring, and sub-50ms production serving.

```
RAW SOURCES (ICRISAT / DES / IMD)
      ↓
CANONICAL PANEL (71,601 records | 29 crops | 20 states | 311 districts | 2010–2017)
      ↓
LEAKAGE-SAFE PRE-SEASON FEATURES (Lag-1, 3-Yr Rolling Mean, Cultivated Area Share)
      ↓
EXPANDING WALK-FORWARD TOURNAMENT (4 Origins: 2014, 2015, 2016, 2017)
      ↓
STRATEGY REGISTRY GATE (PRODUCTION_READY | CONDITIONAL_PRODUCTION | BASELINE_PRODUCTION)
      ↓
FASTAPI FORECAST RUNTIME (Sub-50ms | 3-Sigma Fallback | SHA-256 Provenance | PSI Drift)
      ↓
DECISION WORKSPACE & WEBSHELL (Multi-Evidence Briefs | Bounded What-If Scenarios)
```

---

### Core Technical Accomplishments

1. **Longitudinal Panel Harmonization**: Unified **71,601 records** across 29 crops, 20 states, and 311 districts (1966–2017 historical context; 2010–2017 active panel) with zero duplicate keys and 100% algebraic consistency.
2. **Empirical Model Governance (14 Commodities)**:
   - **Oilseeds** (`PRODUCTION_READY` ML): Certified Random Forest Regressor achieving **75.0% fold win-rate** and **+12.79% mean MAE gain** over baseline persistence.
   - **Sugarcane** (`CONDITIONAL_PRODUCTION` ML): Raw GBDT lost -1.60% during drought regimes; governed GBDT with 3-$\sigma$ district variance clipping restored an aggregate **+1.19% MAE gain** over baseline.
   - **12 Baseline Crops** (`BASELINE_PRODUCTION`): Historical District Mean persistence outperformed machine learning across temporal shocks in primary staples (Rice, Wheat, Chickpea, Maize, etc.).
3. **Negative Weather Ablation Finding**: Evaluated a 5-tier exogenous ablation across all 14 crops; demonstrated that pre-season district weather aggregations produced **no meaningful improvement** over historical autoregressive lags.
4. **Production Engineering & Reliability**:
   - Sub-50ms $P95$ FastAPI inference runtime with Pydantic v2 schemas and 3-$\sigma$ safety bounds.
   - Immutable digital provenance via SHA-256 execution signatures appended to an audit log.
   - Continuous monitoring via Population Stability Index (PSI) drift tracking and post-harvest signed bias.
   - Multi-container Docker deployment with Nginx reverse proxy, non-root execution (`appuser`), fail-closed `/ready` probe (HTTP 503), and WCAG 2.1 AA accessible WebShell.
   - Comprehensive test suite: **581 collected tests** across unit, contract, security, and reproducibility layers.

---

### Technology Stack
- **Languages & Frameworks**: Python 3.11.9, FastAPI, Scikit-learn 1.6.1, Pandas, NumPy, SciPy, TypeScript, React 18, Vite, Tailwind CSS.
- **Infrastructure & Tools**: Docker Engine, Docker Compose, Nginx (Alpine), Pytest, Git.

---

### Primary Links
- **Full Research Paper**: [docs/research_paper/paper.md](file:///docs/research_paper/paper.md)
- **Technical Case Study**: [docs/portfolio/PROJECT_CASE_STUDY.md](file:///docs/portfolio/PROJECT_CASE_STUDY.md)
- **Master Model Card**: [docs/MODEL_CARD.md](file:///docs/MODEL_CARD.md)
- **Dataset Card**: [docs/DATASET_CARD.md](file:///docs/DATASET_CARD.md)
- **Reproducibility Guide**: [docs/reproducibility/REPRODUCIBILITY_GUIDE.md](file:///docs/reproducibility/REPRODUCIBILITY_GUIDE.md)
