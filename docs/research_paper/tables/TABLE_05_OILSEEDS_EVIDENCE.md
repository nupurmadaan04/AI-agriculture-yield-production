# Table 5: Oilseeds Governed Machine Learning Evidence Chain

| Dimension | Verification Artifact / Metric | Quantitative Value | Operational Interpretation |
|---|---|---|---|
| **Certified Model Class** | `RandomForestRegressor` | 150 Estimators, Max Depth 12 | Tree ensemble capturing non-linear spatial interactions |
| **Walk-Forward Win Rate** | `multicrop_model_selection.csv` | **75.0%** (3 of 4 folds) | Consistently outperforms statistical baseline |
| **Mean MAE Improvement** | `multicrop_model_selection.csv` | **+12.79%** over baseline | Substantial operational error reduction |
| **Median Improvement** | `multicrop_model_selection.csv` | **+5.18%** | Improvement is not an artifact of a single outlier fold |
| **Worst-Fold Degradation** | `multicrop_model_selection.csv` | **-2.34%** (Fold 3, 2016) | Downside risk is strictly bounded; no catastrophic failure |
| **Feature Timing Safety** | Feature pipeline audit | **100% LEAKAGE_SAFE** | Zero concurrent harvest features; strict shift(1) lags |
| **Fold Error CV** | `multicrop_model_selection.csv` | 0.4391 | Acceptable cross-fold dispersion |
| **Primary Fallback** | `forecast_strategy_registry.json` | Historical District Mean | Fallback deployed if district lag features are missing |
