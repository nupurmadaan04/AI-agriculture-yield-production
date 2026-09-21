# Figure 6: Empirical P10–P90 Uncertainty Interval Construction

```
+---------------------------------------------------------------------------------------------------------+
|                              FIGURE 6: EMPIRICAL P10–P90 UNCERTAINTY SPREAD                             |
+---------------------------------------------------------------------------------------------------------+

  ENFORCED DEFINITION:
  The P10–P90 interval represents the 10th and 90th percentiles of individual tree predictions across
  the 150 decision trees in the ensemble estimator. It reflects model parameter dispersion within the
  feature manifold; it is NOT a frequentist confidence interval.

  INDIVIDUAL ESTIMATOR PREDICTIONS DISTRIBUTION (District i):
  
  Lower Tail (10th Percentile)                  Median (P50)             Upper Tail (90th Percentile)
         [P10]                                      [P50]                           [P90]
           |                                          |                               |
  ---------+--------------+--------------+------------+------------+--------------+---+---------
           |              |              |            |            |              |   |
         Tree 12       Tree 45        Tree 88      Tree 112     Tree 141       Tree 3  Tree 99
         (1,820 kg/ha)                             (2,150 kg/ha)                   (2,540 kg/ha)
  
  |<--------------------- EMPIRICAL P10–P90 SPREAD: 720 kg/ha ----------------------->|

  EMPIRICAL COVERAGE BENCHMARK:
  - Historical Single-Crop Rice Holdout (2016–2017): Achieved 81.3% empirical coverage (nominal 80%).
  - Multi-Crop Walk-Forward Folds: Coverage varies between ~50% and 78% across drought and normal years.
```

**Interpretation**: Illustrates how the empirical uncertainty interval is extracted directly from the dispersion across the ensemble's 150 individual tree estimators, accompanied by empirical coverage benchmarks.
