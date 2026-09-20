# Day 21: Operational Forecasting Strategies & Fallback Policies

## 1. Operational Policy Framework

For agricultural intelligence platforms, prediction reliability is paramount. A blind machine learning model that makes unconstrained errors in adverse weather conditions damages trust and policy decisions.

Day 21 defines strict **operational policies** and **graceful fallback mechanisms** for each commodity:

```
                          ┌─────────────────────────────┐
                          │ Pre-Season Forecast Request │
                          └──────────────┬──────────────┘
                                         │
                         Is Crop Status ROBUST_ML?
                                ├── YES ──> [Deploy ML Model (RandomForest / GradientBoosting)]
                                │           Fallback: Historical District Mean (if features missing)
                                │
                      Is Crop Status ML_WITH_CONDITIONS?
                                ├── YES ──> Check Operational Conditions:
                                │           - Are trailing yield anomalies within [-20%, +20%]?
                                │           - Is district observation count N >= 3?
                                │             ├── YES ──> [Deploy ML Model]
                                │             └── NO  ──> [Fallback to Historical District Mean]
                                │
                   Is Crop BASELINE_PREFERRED / RESEARCH?
                                └── YES ──> [Deploy Historical District Mean as Primary]
                                            (ML Model runs in shadow mode for research evaluation)
```

---

## 2. Crop-Specific Policy Details

### 1. Oilseeds (`ROBUST_ML`)
- **Primary Model**: `RandomForestRegressor`
- **Fallback Model**: `Historical District Mean`
- **Operating Conditions**: Deploy as primary production forecaster across all districts.
- **Evidence Strength**: 85 / 100 (Strong empirical support).

### 2. Chickpea (`ML_WITH_CONDITIONS`)
- **Primary Model**: `GradientBoostingRegressor`
- **Fallback Model**: `Historical District Mean`
- **Operating Conditions**: Active in normal regimes; automatically fall back to district mean if district experienced severe preceding drought ($>25\%$ yield deficit in $t-1$).
- **Evidence Strength**: 70 / 100 (Moderate empirical support).

### 3. Kharif Sorghum (`ML_WITH_CONDITIONS`)
- **Primary Model**: `RandomForestRegressor`
- **Fallback Model**: `Historical District Mean`
- **Operating Conditions**: Active in normal regimes; fall back to district mean in extreme climate anomalies.
- **Evidence Strength**: 65 / 100 (Moderate empirical support).

### 4. Research Candidates (`Minor Pulses`, `Maize`, `Wheat`, `Sugarcane`)
- **Primary Model**: `Historical District Mean`
- **Fallback Model**: `Naive Persistence`
- **Operating Conditions**: Statistical baseline serves production users. ML model runs in shadow/research mode.
- **Evidence Required for Upgrade**: Inclusion of pre-season weather covariates (precipitation, SPEI) and demonstration of $\ge 75\%$ win rate across 4 walk-forward origins.
- **Evidence Strength**: 40 / 100.

### 5. Baseline Preferred (`Rice`, `Sesamum`, `Pigeonpea`, `Rapeseed & Mustard`, `Groundnut`, `Sorghum`, `Pearl Millet`)
- **Primary Model**: `Historical District Mean`
- **Fallback Model**: `Naive Persistence`
- **Operating Conditions**: Statistical baseline serves production users with high stability and zero overfitting.
- **Evidence Strength**: 30 / 100.
