# Explainable AI (XAI) Implementation Specification

## 1. Marginal Reference Perturbation Attribution

The core model-agnostic explainability engine is defined in `src/explainability_engine.py`:
- **Attribution Equation**:
  $$\phi_j(\mathbf{x}) = f(x_1, \dots, x_j, \dots, x_M) - f(x_1, \dots, x_j^{(0)}, \dots, x_M)$$
- **Reference Profile**: $\mathbf{x}^{(0)}$ is computed as the district's median historical feature vector across the training partition.
- **Controlled Perturbation**: Features are individually substituted by their reference counterpart to measure the isolated marginal response of the estimator.

---

## 2. Tree SHAP Decompositions

For tree ensembles in the Decision Workspace (`src/decision_workspace.py`), local additive feature attributions are computed via Tree SHAP:
$$\hat{y}(\mathbf{x}) = \phi_0 + \sum_{j=1}^M \phi_j(\mathbf{x})$$
where $\phi_0$ is the base expected model prediction.

---

## 3. Scientific Invariants & Guardrails

1. **Baseline Invariant**: Baselines transparently return a notice stating that feature attributions do not apply.
2. **Non-Causal Guardrail**: All UI views and API payloads attach the tag `MODEL_ATTRIBUTION` and disclaim biological causality.
