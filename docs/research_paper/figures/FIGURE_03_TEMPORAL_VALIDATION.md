# Figure 3: Expanding Walk-Forward Temporal Validation Design

```
+---------------------------------------------------------------------------------------------------------+
|                         FIGURE 3: EXPANDING WALK-FORWARD TEMPORAL VALIDATION DESIGN                     |
+---------------------------------------------------------------------------------------------------------+

  All feature engineering and model training strictly respect historical time boundaries (No Future Lookahead).

  FOLD 1 (Origin T = 2014):
  Train: [2010 | 2011 | 2012 | 2013]  =============>  Test: [2014]
  - Historical Sample: 4 years of pre-origin panel observations.
  - Climate Context: Normal baseline monsoon.

  FOLD 2 (Origin T = 2015):
  Train: [2010 | 2011 | 2012 | 2013 | 2014]  =======>  Test: [2015]
  - Historical Sample: 5 years of pre-origin panel observations.
  - Climate Context: Major Pan-India Drought Shock (Rainfall deficit ~14%).
  - Diagnostic Role: Unmasked catastrophic over-prediction in unconstrained GBDT trees.

  FOLD 3 (Origin T = 2016):
  Train: [2010 | 2011 | 2012 | 2013 | 2014 | 2015]  ==>  Test: [2016]
  - Historical Sample: 6 years of pre-origin panel observations.
  - Climate Context: Post-drought recovery monsoon.

  FOLD 4 (Origin T = 2017):
  Train: [2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016]  =>  Test: [2017]
  - Historical Sample: 7 years of pre-origin panel observations.
  - Climate Context: Favorable monsoon across central and southern zones.

  AGGREGATE EVALUATION CRITERIA (Per Commodity):
  1. Fold Win Rate (%)           : Percentage of folds where Model MAE < Baseline MAE (Threshold >= 75%).
  2. Mean MAE Improvement (%)    : Average percentage reduction in MAE across all 4 folds.
  3. Worst Fold Degradation (%)  : Maximum loss incurred in any single test fold (Safety Gate < 5%).
  4. Coefficient of Variation    : Standard deviation of fold MAE divided by mean fold MAE.
```

**Interpretation**: Illustrates the expanding training slices and isolated single-year test holdouts across the four evaluation origins, emphasizing how Fold 2 (2015) stress-tested model resilience under extreme drought conditions.
