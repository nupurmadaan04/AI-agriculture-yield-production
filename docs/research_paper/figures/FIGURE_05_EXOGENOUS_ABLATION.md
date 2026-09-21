# Figure 5: Five-Tier Exogenous Weather Feature Ablation Comparison

```
+---------------------------------------------------------------------------------------------------------+
|                               FIGURE 5: FIVE-TIER EXOGENOUS FEATURE ABLATION                            |
+---------------------------------------------------------------------------------------------------------+

  Evaluating relative MAE performance across 14 crops under 4 expanding walk-forward folds:

  Tier EXP-22A : Historical Only (Lag1, Roll3, Area Share)       [BASELINE: 0.00% Ref Error]
  Tier EXP-22B : Historical + Pre-Season Rainfall Anomalies     [+1.85% Error Increase / Degradation]
  Tier EXP-22C : Historical + Pre-Season Temperature Extremes    [+2.14% Error Increase / Degradation]
  Tier EXP-22D : Historical + Combined Pre-Season Weather        [+3.42% Error Increase / Degradation]
  Tier EXP-22E : Historical + All Exogenous Covariates           [+4.18% Error Increase / Degradation]

  ERROR DEGRADATION VISUALIZATION (Average across 14 Commodities):
  
  EXP-22A (Hist Only) :  |======== (Optimal Error Baseline)
  EXP-22B (+Rainfall)  :  |========== (+1.85% Higher MAE)
  EXP-22C (+Temp)      :  |=========== (+2.14% Higher MAE)
  EXP-22D (+Weather)   :  |============= (+3.42% Higher MAE)
  EXP-22E (+All Exo)   :  |=============== (+4.18% Higher MAE)

  SUMMARY OF ABLATION OUTCOME:
  - 14 out of 14 Crops (100%) certified EXP-22A (Historical Only) as the optimal feature tier.
  - Zero crops demonstrated statistically significant gains from pre-season district weather aggregations.
```

**Interpretation**: Shows the empirical progression of mean forecasting error across the five ablation tiers, visually confirming the negative finding that pre-season weather features degraded out-of-sample accuracy across all evaluated commodities.
