# Day 21: Multi-Crop Model Diagnosis & Robustness Evaluation

## Executive Summary
Day 21 establishes an evidence-based **Model Selection & Error Diagnosis Framework** that re-evaluates all 14 `MODEL_READY` agricultural commodities using multi-origin walk-forward validation evidence (test origins 2014, 2015, 2016, 2017).

Rather than selecting models merely because of aggregate mean performance across a single temporal split, Day 21 establishes strict decision rules based on:
1. **Multi-Origin Win Rate** ($\ge 75\%$ for unconstrained adoption)
2. **Worst-Fold Degradation** ($\le 5\%$ allowable loss in adverse seasons)
3. **Yield Regime Decomposition** (Low $\le Q_{25}$, Normal $Q_{25}-Q_{75}$, High $\ge Q_{75}$)
4. **District-Level Error Homogeneity** ($N \ge 3$ observation threshold)
5. **Feature Timing & Anti-Leakage Audit**

---

## 1. Decision Hierarchy & Taxonomy

```
Day 20 Walk-Forward Results (4 Origins × 14 Commodities)
                    ↓
Temporal Robustness & Win Rate Assessment
                    ↓
Error Stability & Tail Dispersion Analysis
                    ↓
Baseline Comparison (Historical District Mean vs ML)
                    ↓
Failure & Regime Diagnosis (Low vs High Yield, 2016 High-Error Regime)
                    ↓
Feature Timing & Leakage Verification
                    ↓
Deterministic Model Selection Rule
                    ↓
Crop-Specific Operational Forecasting Policy
```

### Classification Statuses
- **`ROBUST_ML`**: Clear empirical superiority across $\ge 75\%$ of walk-forward origins, positive average and median gain, and negligible worst-fold degradation ($\le 5\%$).
- **`ML_WITH_CONDITIONS`**: Demonstrates positive average gain and $\ge 50\%$ win rate, but suffers from isolated worst-fold volatility or regime-specific sensitivity. Deployed with mandatory baseline fallback rules.
- **`BASELINE_PREFERRED`**: Statistical baselines (Historical District Mean) consistently match or outperform ML across temporal splits. District mean serves as the primary operational forecaster.
- **`RESEARCH_CANDIDATE`**: ML captures localized predictive signal in selected districts or regimes, but overall win rate is low ($\le 50\%$) or mean gain is negative. Retained in shadow mode pending exogenous weather/satellite features.
- **`INSUFFICIENT_EVIDENCE`**: Sample size or observation density is inadequate to establish statistical defensibility.

---

## 2. Multi-Crop Diagnosis Overview

| Commodity | Selected Model | Win Rate | Mean Gain (%) | Worst Fold (%) | MAE CV | Day 21 Decision | Primary Policy |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Oilseeds** | RandomForestRegressor | **75.0%** | **+12.79%** | **-2.34%** | 0.3803 | `ROBUST_ML` | Primary ML forecaster; District Mean fallback |
| **Chickpea** | GradientBoostingRegressor | **75.0%** | **+2.31%** | **-22.30%** | 0.1746 | `ML_WITH_CONDITIONS` | Primary in normal regimes; District Mean in extremes |
| **Kharif Sorghum** | RandomForestRegressor | **50.0%** | **+1.30%** | **-2.28%** | 0.1088 | `ML_WITH_CONDITIONS` | Primary in normal regimes; District Mean in extremes |
| **Minor Pulses** | GradientBoostingRegressor | **50.0%** | **-5.75%** | **-21.48%** | 0.1619 | `RESEARCH_CANDIDATE` | Historical District Mean primary; ML shadow mode |
| **Maize** | RandomForestRegressor | **25.0%** | **-2.63%** | **-6.22%** | 0.1169 | `RESEARCH_CANDIDATE` | Historical District Mean primary; ML shadow mode |
| **Wheat** | GradientBoostingRegressor | **25.0%** | **-14.73%** | **-24.63%** | 0.2078 | `RESEARCH_CANDIDATE` | Historical District Mean primary; ML shadow mode |
| **Sugarcane** | GradientBoostingRegressor | **25.0%** | **-0.66%** | **-8.87%** | 0.1462 | `RESEARCH_CANDIDATE` | Historical District Mean primary; ML shadow mode |
| **Rice** | RandomForestRegressor | **25.0%** | **-8.39%** | **-12.72%** | 0.1223 | `BASELINE_PREFERRED` | Historical District Mean / Naive Persistence |
| **Sesamum** | RandomForestRegressor | **25.0%** | **-1.19%** | **-3.79%** | 0.0948 | `BASELINE_PREFERRED` | Historical District Mean / Naive Persistence |
| **Pigeonpea** | RandomForestRegressor | **0.0%** | **-5.68%** | **-8.98%** | 0.1244 | `BASELINE_PREFERRED` | Historical District Mean / Naive Persistence |
| **Rapeseed & Mustard** | RandomForestRegressor | **25.0%** | **-5.62%** | **-10.74%** | 0.1246 | `BASELINE_PREFERRED` | Historical District Mean / Naive Persistence |
| **Groundnut** | RandomForestRegressor | **25.0%** | **-7.87%** | **-12.24%** | 0.0674 | `BASELINE_PREFERRED` | Historical District Mean / Naive Persistence |
| **Sorghum** | RandomForestRegressor | **25.0%** | **-3.50%** | **-8.15%** | 0.1163 | `BASELINE_PREFERRED` | Historical District Mean / Naive Persistence |
| **Pearl Millet** | RandomForestRegressor | **0.0%** | **-6.04%** | **-8.72%** | 0.0247 | `BASELINE_PREFERRED` | Historical District Mean / Naive Persistence |

---

## 3. Scientific Invariants
1. **Zero Data Leakage**: Temporal walk-forward folds strictly prevent future information leakage.
2. **Preserved Lineage**: Historical single-split Day 19 and Day 20 metrics remain intact in registry records.
3. **No Phantom Metrics**: All numbers are derived directly from verified test partitions.
