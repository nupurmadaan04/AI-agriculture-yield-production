# Day 31: Decision Intelligence & Evidence-Based Forecast Briefs

## 1. Overview & Architectural Principle

The **Decision Intelligence & Evidence-Based Forecast Briefs** module establishes the decision synthesis layer of the Agricultural Intelligence & Forecasting Platform. It operates strictly on top of the frozen scientific, forecasting, explainability, provenance, and monitoring layers.

The fundamental governance invariant is:
**The decision layer must NEVER independently invent, fabricate, or extrapolate recommendations, causal directives, or speculative yields.**
Instead, the system gathers authentic analytical artifacts across the entire operational lifecycle into a transparent, verifiable, and auditable **Decision Brief**.

```
  DATA (AGRI_PANEL_1.0)
       |
       v
  FORECAST (Certified Strategies & Fallbacks)
       |
       v
  EXPLAINABILITY (Tree SHAP & Feature Attribution)
       |
       v
  PROVENANCE (SHA-256 Cryptographic Audit Lineage)
       |
       v
  MONITORING (Operational Health & PSI Drift)
       |
       v
  OBSERVED OUTCOMES (Leak-Free Evaluation Records)
       |
       v
  EVIDENCE SYNTHESIS (11-Rule Scientific Harmonization)
       |
       v
  DECISION BRIEF (9-Dimension Structured Evidence Brief)
```

---

## 2. Decision Intelligence Architecture

The Day 31 architecture is organized into four clean layers:

### A. Evidence Harvesting Engine (`src/decision_intelligence.py`)
- **Strict Temporal Boundary Isolation**: All historical statistical moments (mean, median, standard deviation, trend slopes, sample depth) are calculated strictly on panel records where $\text{Year} < \text{forecast\_year}$. Zero lookahead leakage is enforced.
- **Governed Forecast & Provenance Linkage**: Calls `PredictionService` to obtain pre-season forecasts alongside the model's cryptographic SHA-256 fingerprint, strategy tier, and request ID.
- **Model Validation Proof**: Extracts certified walk-forward out-of-time evaluation metrics from `StrategyRegistry` (MAE, baseline MAE, fold win rate %, gain vs baseline %, legacy benchmarks).
- **Tree-Ensemble Dispersion Uncertainty**: Evaluates empirical P10–P90 spread across estimator trees when ML models are certified, accompanied by clear disclaimers that this is an empirical spread, not a formal distribution-free confidence interval.
- **Operational Monitoring & Bias**: Ingests Day 30 Population Stability Index (PSI) drift values, runtime error logs, and post-outcome directional bias ($\text{forecast} - \text{observed}$) when evaluation is published. If forecasting future/unharvested seasons (e.g. 2026), post-outcome evaluation is transparently marked `EVALUATION_UNAVAILABLE`.
- **Explainability Deconstruction**: Decomposes model adjustments using Tree SHAP feature attributions, explaining feature dependencies without asserting biological causality.

### B. Decision Brief Generator (`src/decision_brief.py`)
Synthesizes the harvested evidence matrix into a 9-dimension executive structure:
1. **Executive Summary**: Context, forecast, strategy, certainty tier, and key signals.
2. **Context & Scope**: Administrative entity, commodity, forecast horizon, and land scale.
3. **Governed Forecast Summary**: Forecasted yield, units, model name, and version.
4. **Historical Baseline Comparison**: Empirical moments, longitudinal slope, and recent observations.
5. **Model & Validation Evidence**: Strategy certification tier, expanding walk-forward metrics, baseline gains, and legacy benchmark notes.
6. **Uncertainty & Model Dispersion**: Empirical P10–P90 ranges with non-probabilistic disclaimers.
7. **Explainability & Attribution**: Tree SHAP attributions labeled mathematically.
8. **Operational Monitoring & Health**: Real-time PSI drift, telemetry error rates, and signed bias.
9. **Decision Considerations**: Trade-offs, assumptions, and explicit operational limitations.

### C. Scientific Validation Guard (`src/decision_validation.py`)
Enforces 11 automated scientific rules before any brief is emitted:
- Rule 1: Dataset version matches `AGRI_PANEL_1.0`.
- Rule 2: Model version corresponds to registered crop artifact.
- Rule 3: Strategy tier matches `StrategyRegistry`.
- Rule 4: Temporal ordering preserves $t < T_{\text{forecast}}$.
- Rule 5: Historical observations are authentic panel records.
- Rule 6: Validation metrics match walk-forward benchmarks.
- Rule 7: Empirical dispersion contains non-confidence disclaimer.
- Rule 8: Explainability attributions are labeled mathematical dependencies.
- Rule 9: Monitoring metrics match live telemetry state.
- Rule 10: Decision options are strictly tagged `SIMULATED` and `DERIVED`.
- Rule 11: Absolute zero causal language ("will increase", "causes", "guaranteed", "farmers must").

### D. REST & Interactive Frontend Layer
- **REST Endpoints**:
  - `POST /api/decision/analyze` & `GET /api/decision/analyze`
  - `POST /api/decision/brief` & `GET /api/decision/brief`
  - `GET /api/decision/evidence/{crop}`
  - `POST /api/decision/options` & `POST /api/decision/robustness`
  - `GET /api/decision/history` & `GET /api/decision/methodology`
  - `GET /api/decision/{decision_id}/audit` & `GET /api/decision/{decision_id}/provenance`
- **Frontend Dashboard (`/decision-intelligence`)**:
  - 14-commodity dynamic selector with district filtering.
  - Interactive Evidence Matrix with category filtering and semantic badge color coding.
  - Scenario simulation options with trade-offs and sensitivity robustness.
  - Audit certificate verification with SHA-256 lineage DAG.
  - One-click Markdown and JSON export capabilities.

---

## 3. Four Golden Use Cases

| Commodity | Strategy Tier | Validation Evidence | Legacy Benchmark | Non-Causal Boundary |
|:---|:---|:---|:---|:---|
| **Oilseeds** | `PRODUCTION_READY` (ML) | 4-fold WF Gain: +7.2%, Win Rate: 75% | N/A | Feature attribution reflects mathematical tree dependencies. |
| **Sugarcane** | `CONDITIONAL_PRODUCTION` (ML) | 4-fold WF Gain: +2.1%, Win Rate: 50% | N/A | Sensitivity to split bounds noted; conditional oversight required. |
| **Rice** | `BASELINE_PRODUCTION` | Historical District Mean / Persistence | Validated Benchmark: $R^2 = 0.7866$, $\text{MAE} = 353.01\text{ kg/ha}$ | Baseline preferred under multi-origin walk-forward protocol. |
| **Wheat** | `BASELINE_PRODUCTION` | Historical District Mean / Persistence | N/A | Deterministic district mean preferred over overfitting ML models. |

---

## 4. Operational Invariance
- Zero retraining of machine learning weights.
- Bitwise deterministic outputs across repeated calls.
- Total traceability from every prose statement back to an authentic evidence ID (`EV-HIST-xxxx`, `EV-FORE-xxxx`, `EV-VALI-xxxx`, etc.).
