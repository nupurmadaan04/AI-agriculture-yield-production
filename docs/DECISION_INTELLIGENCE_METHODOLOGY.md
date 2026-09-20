# Agricultural Decision Intelligence & Evidence Synthesis Methodology

## 1. Purpose & Core Philosophy
The **Agricultural Decision Intelligence Layer** (Day 14) serves as the top-level orchestration and decision-support synthesis layer of the platform. It unifies the analytical findings of Days 1 through 13 into structured, auditable **Decision Briefs** and **Evidence Reports**.

The platform adheres to fundamental scientific integrity principles:
1. **Zero Fabrication**: No synthetic probabilities, ungrounded risk percentages, or unmodeled agricultural variables (e.g. unmeasured chemical fertilizers).
2. **Explicit Evidence Typing**: Every data point is tagged as `OBSERVED`, `PREDICTED`, `SIMULATED`, `DERIVED`, `MODEL_ATTRIBUTION`, or `VALIDATION`.
3. **Strict Non-Causal Grounding**: Models describe statistical associations and simulation response curves within historical distributions, never physical or biological causal truth.
4. **Deterministic Reproducibility**: SHA-256 Decision Certificates (`DEC-xxxx`) verify that identical inputs, model registry versions, and evidence sets produce identical output hashes.

---

## 2. Decision Intelligence Architecture & Flow
```
                 ┌─────────────────┐
                 │  ICRISAT DATA   │
                 └────────┬────────┘
                          ↓
                ┌───────────────────┐
                │   ML FORECAST     │
                └────────┬──────────┘
                         ↓
       ┌─────────────────────────────────────┐
       │                                     │
       ↓                                     ↓
GEOSPATIAL RISK                    TEMPORAL MONITORING
       │                                     │
       ↓                                     ↓
SPATIAL SIGNALS                    EARLY WARNING SIGNALS
       │                                     │
       └────────────────┬────────────────────┘
                        ↓
              ┌──────────────────┐
              │ MODEL RELIABILITY│
              └────────┬─────────┘
                       ↓
              ┌──────────────────┐
              │ XAI EXPLANATION  │
              └────────┬─────────┘
                       ↓
             ┌────────────────────┐
             │ SCENARIO OPTIONS   │
             └─────────┬──────────┘
                       ↓
             ┌────────────────────┐
             │ OPTIMIZATION       │
             └─────────┬──────────┘
                       ↓
              ┌──────────────────┐
              │ DECISION BRIEF   │
              └────────┬─────────┘
                       ↓
              ┌──────────────────┐
              │ AUDIT CERTIFICATE│
              └──────────────────┘
```

---

## 3. Evidence Classification Taxonomy
| Taxonomy Type | Definition | Source Module Example |
| :--- | :--- | :--- |
| **`OBSERVED`** | Empirical historical data points directly recorded in verified dataset | `data_loader` (ICRISAT panel yields) |
| **`PREDICTED`** | Machine learning estimates produced by registered validation models | `forecast_service` (Pre-season forecast) |
| **`SIMULATED`** | Hypothetical scenario and optimization projections under modified parameters | `scenario_service`, `optimization_service` |
| **`DERIVED`** | Deterministic mathematical and statistical transformations | `trend_service`, `geospatial_service` (z-scores, CUSUM) |
| **`MODEL_ATTRIBUTION`** | Feature contribution explanations quantifying model behavior | `explainability_service` (Marginal attribution) |
| **`VALIDATION`** | Out-of-time chronological validation metrics | `validation_service` ($R^2=0.7866$, $\text{MAE}=353.01$) |

---

## 4. Multi-Signal Fusion & Decision Priority
Rather than generating speculative probabilities (e.g. "87% chance of crop failure"), the platform computes transparent **Signal Strength** (`HIGH`, `MODERATE`, `LOW`), persistence, and severity metrics grounded in supporting evidence IDs.

Analytical priorities are evaluated based on:
1. Early warning risk severity and persistence
2. Multi-year trajectory slope ($\text{kg/ha/yr}$)
3. Spatial peer departures
4. Ensemble prediction spread ($\pm\text{kg/ha}$)
5. Out-of-time model reliability ($R^2$, $\text{MAE}$)

---

## 5. Scenario Options & Robustness Analysis
Decision options are mapped directly from Day 10 scenario archetypes (Status Quo, Moderate Improvement, Target Reallocation, Pareto Optimal) and evaluated across continuous sensitivity sweeps (Day 10 $\pm20\%$ parameter perturbations):
- **`ROBUST`**: Projected outcome remains favorable across the full perturbation range.
- **`MODERATELY ROBUST`**: Outcome remains positive with moderate elasticity.
- **`SENSITIVE`**: Outcome varies significantly under feature shifts.
- **`UNSUPPORTED`**: Variables lie outside the trained feature space.

---

## 6. Provenance DAG & SHA-256 Decision Certificates
Every decision brief includes a Directed Acyclic Graph (DAG) linking:
$$\text{Decision Statement} \longrightarrow \text{Evidence IDs} \longrightarrow \text{Source Modules} \longrightarrow \text{Model Version} \longrightarrow \text{Dataset Version}$$

Cryptographic SHA-256 certificates (`DEC-{hash}`) are computed from canonicalized JSON payloads ensuring full computational reproducibility.
