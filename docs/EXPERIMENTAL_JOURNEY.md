# The Experimental Journey: Evolution of the Platform

## 1. Phase-by-Phase Progression

```
Baseline Modeling (Days 1–3)
       │
       ▼
Data Quality & Preprocessing (Days 4–7)
       │
       ▼
Spatial Intelligence (Day 8)
       │
       ▼
Model Validation & Reliability (Day 9)
       │
       ▼
Scenario Simulation & Optimization (Day 10)
       │
       ▼
Temporal Monitoring & Early Warning (Day 12)
       │
       ▼
Explainability & Attribution (Day 13)
       │
       ▼
Decision Intelligence & Evidence Synthesis (Day 14)
       │
       ▼
Production Hardening & Integration (Day 15)
       │
       ▼
Final Research Packaging & Demo Readiness (Day 16)
```

---

## 2. Rationale Behind Each Architectural Evolution

### Days 1–3: Baseline Modeling & Initial Feasibility
- **Goal**: Assess whether district panel covariates (soil, rainfall, inputs) have predictive signal for rice yield.
- **Outcome**: Established simple Linear and Random Forest baselines ($R^2 \approx 0.54 - 0.71$).

### Days 4–7: Preprocessing & Outlier Quarantining
- **Goal**: Resolve data quality issues, handle territory changes, and filter reporting anomalies.
- **Outcome**: Created anti-leakage temporal feature pipelines, missing-value imputation, and clean 2,469 panel records.

### Day 8: Geospatial Risk & Spatial Intelligence
- **Goal**: Account for spatial autocorrelation where neighboring districts exhibit correlated yield performance.
- **Outcome**: Integrated K-Means agro-climatic clustering ($k=4$), Local Moran's I spatial autocorrelation, and within-state Z-score normalization.

### Day 9: Rigorous Model Validation & Reliability Governance
- **Goal**: Prevent optimistic bias caused by random train/test splits.
- **Outcome**: Enforced chronological out-of-time validation ($2013-2017$), establishing verified benchmark metrics ($R^2 = 0.7866$, $\text{MAE} = 353.01\text{ kg/ha}$) and 10-decile prediction spread calibration.

### Day 10: Scenario Simulation & Decision Optimization
- **Goal**: Allow agronomists to evaluate hypothetical input adjustments (e.g. fertilizer reallocation, precipitation shifts).
- **Outcome**: Built constrained SLSQP optimization with physical non-negativity bounds and scenario audit tracking.

### Day 12: Temporal Monitoring & Early Warning
- **Goal**: Continuously monitor multi-year yield trajectories and detect abrupt declines before harvest.
- **Outcome**: Implemented 3-year/5-year moving averages, two-sided CUSUM drift detection, hazard index calculation, and step-forward backtesting across 311 districts.

### Day 13: Explainable AI & Feature Attribution
- **Goal**: Provide transparent rationale for model predictions without black-box opacity.
- **Outcome**: Integrated Tree SHAP local feature attributions and local sensitivity sweeps with non-causal guardrails.

### Day 14: Decision Intelligence & Evidence Reports
- **Goal**: Unify disparate analytical outputs into an actionable, auditable decision brief.
- **Outcome**: Developed multi-module evidence ranking, robustness classification, Statement $\rightarrow$ Model $\rightarrow$ Data provenance DAG, and deterministic SHA-256 decision audit records (`DEC-xxxx`).

### Days 15–16: Production Hardening, Release & Live Demo Readiness
- **Goal**: Consolidate code, centralize paths/configs, add observability probes, containerize, and package for live demonstrations and portfolio showcase.
- **Outcome**: 283 passed automated tests, single-command Docker deployment, React ErrorBoundary, complete technical documentation.
