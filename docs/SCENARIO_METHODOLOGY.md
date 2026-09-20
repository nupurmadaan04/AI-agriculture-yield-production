# Scenario Simulation & Decision Optimization Methodology

## 1. Scenario Simulation Principles
Hypothetical scenario simulations evaluate model responses under modified input assumptions:
- **Baseline Preservation:** Every simulated intervention is evaluated against an unchanged baseline:

$$\Delta_{\text{yield}} = \hat{y}(\mathbf{x}_{\text{scenario}}) - \hat{y}(\mathbf{x}_{\text{baseline}})$$

$$\Delta_{\text{yield}}\% = \frac{\Delta_{\text{yield}}}{\hat{y}(\mathbf{x}_{\text{baseline}})} \times 100$$

- **Non-Causal Representation:** Outputs denote *model-estimated scenario responses*, never guaranteed agronomic outcomes.

---

## 2. Decision Optimization Engine
The optimization engine searches for resource-constrained input modifications that maximize expected yield or minimize climate risk:

$$\max_{\boldsymbol{\delta}} \quad \mathbb{E}[\hat{y}(\mathbf{x}_0 + \boldsymbol{\delta})] - \lambda \cdot \|\boldsymbol{\delta}\|^2$$

$$\text{subject to} \quad \mathbf{l} \le \boldsymbol{\delta} \le \mathbf{u}, \quad \text{Budget}(\boldsymbol{\delta}) \le B$$

---

## 3. Sensitivity & Tradeoff Analysis
- **One-at-a-Time (OAT) Sensitivity:** Sweeps individual variables through $[-30\%, +30\%]$ in discrete steps.
- **Tradeoff Ranking:** Multi-attribute utility ranking weighting yield gain, input cost, and risk exposure.
- **Scenario Audit Trail:** Every simulation logs `scenario_id`, parameters, and delta response for auditability.
