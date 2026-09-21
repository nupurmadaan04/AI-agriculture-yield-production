# Figure 4: Governed Forecast Strategy Decision Tree & Matrix

```
+---------------------------------------------------------------------------------------------------------+
|                               FIGURE 4: GOVERNED STRATEGY DECISION TREE                                 |
+---------------------------------------------------------------------------------------------------------+

                         [COMMODITY CANDIDATE]
                                  |
                                  v
                    [Sufficiency Gate: Day 18]
                    Is N >= 100 & Temporal >= 5 yrs?
                                  |
                +-----------------+-----------------+
                | YES                               | NO
                v                                   v
    [Temporal Tournament: Day 20-21]    [ANALYTICS_READY / INSUFFICIENT]
    Evaluate 4 Walk-Forward Folds       (15 secondary crops)
                |
                +------------------------------------------------------+
                |                                                      |
                v                                                      v
  [Fold Win Rate >= 75%?]                               [Fold Win Rate < 50% OR]
  [Mean Gain > 0 & Worst Loss < 5%?]                   [Baseline MAE <= ML MAE?]
                |                                                      |
        +-------+-------+                                              v
        | YES           | NO                                  [BASELINE_PRODUCTION]
        v               v                                     Primary: Hist. District Mean
  [PRODUCTION_READY] [Win Rate >= 50% &               Fallback: Crop Long-Term Mean
  Primary: RF ML       Regime Volatility?]                    (12 Crops: Rice, Wheat,
  Fallback: Mean                |                             Chickpea, Kharif Sorghum,
  (Oilseeds: +12.79%)   +-------+-------+                     Maize, Minor Pulses, etc.)
                        | YES           | NO
                        v               v
            [CONDITIONAL_PRODUCTION] [BASELINE_PRODUCTION]
            Primary: GBDT + 3-sigma Clip
            Fallback: Hist. District Mean
            (Sugarcane: +1.19%)
```

**Interpretation**: Illustrates the empirical decision logic executed by the certification guard, ensuring that machine learning is deployed only where temporal robustness is statistically demonstrated, while defaulting to transparent statistical district baselines otherwise.
