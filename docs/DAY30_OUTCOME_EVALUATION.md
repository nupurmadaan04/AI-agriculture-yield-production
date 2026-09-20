# Day 30: Post-Outcome Evaluation & Temporal Isolation

## 1. The Core Scientific Boundary

A machine learning pre-season yield forecast is generated prior to sowing or early in the crop lifecycle. Post-outcome evaluation must strictly preserve the arrow of time:

$$\text{Forecast Origin} < \text{Forecast Horizon} \le \text{Outcome Publication}$$

### Non-Negotiable Scientific Principles
1. **Zero Retroactive Modification**: The original pre-season forecast must remain immutable once generated and hashed via SHA-256 provenance.
2. **Outcome Isolation**: Harvest observations may only be utilized for evaluation after the season concludes. They must NEVER be fed back into the model to alter the original prediction before evaluating it.
3. **Unharvested Future Horizons**: For requests targeting future or ongoing harvest periods (e.g., 2026/2027 where official harvest statistics do not yet exist), the system explicitly returns `EVALUATION_UNAVAILABLE` rather than fabricating synthetic outcomes.

---

## 2. Evaluation Mathematics & Error Metrics

Let $\hat{y}_i$ denote the pre-season predicted yield (kg/ha) for district $i$, and $y_i$ denote the actual observed yield (kg/ha) recorded in the canonical panel.

### Signed Forecast Error (Directional Bias)
$$e_i = \hat{y}_i - y_i$$
- $e_i > 0$: Over-prediction (model forecast exceeded actual harvest).
- $e_i < 0$: Under-prediction (model forecast was lower than actual harvest).

### Absolute Error
$$|e_i| = |\hat{y}_i - y_i|$$

### Aggregate Evaluation Metrics
- **Mean Absolute Error (MAE)**:
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |\hat{y}_i - y_i|$$
- **Root Mean Squared Error (RMSE)**:
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2}$$
- **Median Absolute Error**:
  $$\text{MedAE} = \text{median}(|\hat{y}_1 - y_1|, \dots, |\hat{y}_N - y_N|)$$
- **Mean Signed Bias**:
  $$\text{Bias} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$
- **Mean Absolute Percentage Error (MAPE)**:
  $$\text{MAPE} = \frac{100\%}{N} \sum_{i=1, y_i > 0}^N \left| \frac{\hat{y}_i - y_i}{y_i} \right|$$
  *(Protected against divide-by-zero on uncultivated/zero-yield districts).*

---

## 3. Stratified Error Decomposition

Evaluation errors are stratified across three distinct operational dimensions:

1. **Temporal Walk-Forward Origins (2014–2017)**:
   Verifies year-over-year error stability across expanding training horizons.
2. **Geographic District Slices**:
   Identifies spatial dispersion in forecast accuracy across agro-climatic zones with mandatory sample size constraints ($N \ge 3$).
3. **Yield Regimes**:
   Evaluates performance across yield quantiles:
   - Low Yield Regime: $y \le Q_{25}$
   - Normal Yield Regime: $Q_{25} < y < Q_{75}$
   - High Yield Regime: $y \ge Q_{75}$
