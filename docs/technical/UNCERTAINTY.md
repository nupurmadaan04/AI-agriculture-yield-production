# Empirical Uncertainty Framework

## 1. Ensemble Dispersion Intervals

Uncertainty is generated in `src/decision_workspace.py` by querying individual estimators within the trained tree ensembles:
- **Interval Bounds**:
  $$\hat{y}_{P10} = \text{Quantile}_{0.10} \left( \{ f_b(\mathbf{x}) \}_{b=1}^B \right)$$
  $$\hat{y}_{P90} = \text{Quantile}_{0.90} \left( \{ f_b(\mathbf{x}) \}_{b=1}^B \right)$$
  where $B=150$ individual tree estimators.
- **Dispersion Spread**: $\Delta = \hat{y}_{P90} - \hat{y}_{P10}$.

---

## 2. Terminology Standards & Calibration Limits

1. **Mandatory Nomenclature**: Referred to exclusively as **"Empirical P10–P90 Ensemble Interval"**.
2. **Prohibited Terminology**: Must never be described as a frequentist confidence interval or Bayesian credible interval.
3. **Coverage Benchmark**: Achieved 81.3% empirical coverage in the 2016–2017 Rice holdout test split.
