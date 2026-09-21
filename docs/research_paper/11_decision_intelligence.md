# 10. Decision Intelligence & Multi-Scenario Synthesis

### 10.1 From Isolated Point Forecasts to Multi-Evidence Decision Briefs

Operational decision-makers—such as district agricultural officers, relief commissioners, and commodity planners—cannot act effectively on isolated point estimates ($\hat{y} = 2,450\text{ kg/ha}$). Practical agricultural decisions require contextual evidence that illuminates underlying historical stability, regional baselines, empirical dispersion, feature drivers, and model reliability.

The Decision Workspace (`src/decision_workspace.py`) addresses this by synthesizing seven distinct evidentiary streams into an inspectable **Decision Brief**:

1. **Certified Forecast**: Expected yield $\hat{y}$ generated strictly via the certified strategy router.
2. **Historical District Mean Reference**: Long-term historical district average ($\mu_{\text{dist}}$) serving as the anchor baseline.
3. **Autoregressive Context**: Prior year yield ($y_{t-1}$) and three-year rolling average ($\bar{y}_{t-1:t-3}$) indicating recent local trajectory.
4. **Empirical Uncertainty Spread**: P10–P90 tree ensemble dispersion indicating estimator variance.
5. **Model Attribution Profile**: Primary local drivers derived via Marginal Reference Perturbation or Tree SHAP.
6. **Operational Telemetry Context**: Covariate drift status (PSI) and active strategy fallback flags.
7. **Cryptographic Lineage**: SHA-256 digital fingerprint guaranteeing full auditability.

Crucially, the platform operates strictly as a **decision support and evidence synthesis tool**. It does not autonomously execute policy or command resource reallocations; rather, it augments human agronomic expertise by rendering all analytical evidence transparent and inspectable.

---

### 10.2 Scenario Simulation: Exploring What-If Manifolds

To support proactive planning, the Decision Workspace enables agricultural planners to conduct hypothetical "what-if" scenario simulations:

```
+----------------------------------------------------------------------------------------------------+
|                                    SCENARIO SIMULATION BOUNDARIES                                  |
+----------------------------------------------------------------------------------------------------+
| CORE PRINCIPLE: SCENARIO != FORECAST                                                               |
| - A Forecast represents an empirical pre-season prediction based on observed historical features.  |
| - A Scenario represents a simulated mathematical perturbation across hypothetical inputs.         |
| - Scenario outputs represent response manifolds, NOT observed outcomes or causal certainties.      |
+----------------------------------------------------------------------------------------------------+
```

#### Simulation Mechanics & Bounded Manifolds
- **Input Parameters**: Planners can adjust hypothetical inputs, such as assumed pre-season rainfall anomalies ($\pm 30\%$) or shifted acreage allocations ($\pm 20\%$).
- **Manifold Constraints**: To prevent extreme out-of-distribution hallucinations, scenario perturbations are bounded to the 5th–95th empirical percentiles of historical district observations. If a user inputs a scenario exceeding historical bounds, the interface generates an explicit out-of-manifold warning.
- **Multi-Objective Trade-Offs**: Where acreage reallocations across competing crops are evaluated, the engine optimizes regional gross production subject to water availability and land constraints using Sequential Least Squares Programming (SLSQP).
- **Non-Causal Disclaimer**: Every scenario brief explicitly displays the entity tag **`[SCENARIO]`** alongside a mandatory disclaimer:
  > *"Scenario outputs reflect model sensitivity under hypothetical input assumptions. They do not constitute agronomic guarantees, biological causal effects, or deterministic policy outcomes."*
