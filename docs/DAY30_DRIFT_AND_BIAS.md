# Day 30: Covariate Drift Detection & Systematic Bias Diagnostics

## 1. Population Stability Index (PSI) Methodology

Population Stability Index (PSI) quantifies the degree of shift in continuous feature distributions between a historical baseline reference period and an out-of-time evaluation period.

$$\text{PSI} = \sum_{k=1}^K (P_k - Q_k) \times \ln\left(\frac{P_k}{Q_k}\right)$$

where:
- $K = 10$ quantile bins constructed from the reference distribution.
- $P_k$: Proportion of observations in bin $k$ from the evaluation window ($2016–2017$).
- $Q_k$: Proportion of observations in bin $k$ from the baseline reference window ($2010–2015$).
- $\epsilon = 10^{-4}$ smoothing factor for empty bin regularization.

### Standard Industry Threshold Rules
- $\text{PSI} < 0.10$: **STABLE / NO DRIFT**. Distribution exhibits negligible shift; models remain statistically calibrated.
- $0.10 \le \text{PSI} < 0.25$: **MODERATE DRIFT / WATCH**. Mild shift detected; feature distributions warrant monitoring.
- $\text{PSI} \ge 0.25$: **SIGNIFICANT DRIFT / ACTION REQUIRED**. Substantial distributional shift; triggers operational alerts for agronomic review.

---

## 2. Monitored Feature Covariates

The drift monitoring engine continuously inspects key agricultural features:
1. `RICE AREA (1000 ha)`
2. `TOTAL_CROPPED_AREA`
3. `RICE_AREA_SHARE`
4. `RICE_YIELD_LAG1` (1-year lag yield)
5. `RICE_YIELD_ROLL3` (3-year rolling mean yield)
6. `WHEAT AREA (1000 ha)`
7. `COTTON AREA (1000 ha)`
8. `SUGARCANE AREA (1000 ha)`
9. `RICE YIELD (Kg per ha)`

---

## 3. Systematic Directional Bias Classification

Systematic bias evaluates whether models consistently over-estimate or under-estimate yield across multi-fold walk-forward validation sets.

### Normalized Mean Error (NME %)
$$\text{NME\%} = \left( \frac{\frac{1}{N}\sum (\hat{y}_i - y_i)}{\bar{y}_{\text{actual}}} \right) \times 100\%$$

### Decision Boundaries
- $\text{NME\%} > +3.0\%$: `OVER_PREDICTION_BIAS` (Model systematically forecasts yields above observed ground truth).
- $\text{NME\%} < -3.0\%$: `UNDER_PREDICTION_BIAS` (Model systematically forecasts yields below observed ground truth).
- $-3.0\% \le \text{NME\%} \le +3.0\%$: `NO_CLEAR_BIAS` (Residuals operate within certified tolerances).

### Empirical Multi-Crop Bias Summary
- **Oilseeds**: $+54.52\%$ NME (Managed by primary ML Random Forest with strict fallback guards).
- **Sugarcane**: $+2.92\%$ NME (`NO_CLEAR_BIAS`, operating within $3\sigma$ clipping bounds).
- **Rice**: $-3.13\%$ NME (Governed under `BASELINE_PRODUCTION` Historical District Mean).
- **Wheat**: $-1.69\%$ NME (`NO_CLEAR_BIAS`, governed under `BASELINE_PRODUCTION`).
