# Table 4: Walk-Forward Temporal Tournament Results Across Four Origins

| Commodity | Evaluated Model Class | Fold 1 (2014) Gain (%) | Fold 2 (2015) Gain (%) | Fold 3 (2016) Gain (%) | Fold 4 (2017) Gain (%) | Fold Win Rate (%) | Mean MAE Gain (%) | Worst Fold Loss (%) |
|---|---|---|---|---|---|---|---|---|
| **Oilseeds** | `RandomForestRegressor` | +14.2% | +8.9% | -2.34% | +30.4% | **75.0%** | **+12.79%** | -2.34% |
| **Sugarcane (Raw)** | `GradientBoostingRegressor` | -9.92% | -10.19% | +9.75% | +3.96% | 50.0% | -1.60% | -10.19% |
| **Sugarcane (Clip)** | Governed GBDT + 3-$\sigma$ Clip | -9.92% | -5.78% | +9.75% | +8.44% | 50.0% | **+1.19%** | -9.92% |
| **Rice** | `RandomForestRegressor` | -4.2% | -18.6% | +2.1% | -12.9% | 25.0% | -8.39% | -18.6% |
| **Wheat** | `RandomForestRegressor` | -12.1% | -24.8% | +1.4% | -37.9% | 50.0% | -18.36% | -37.9% |
| **Chickpea** | `RandomForestRegressor` | -2.1% | -6.4% | +4.2% | -1.8% | 25.0% | -1.53% | -6.4% |
| **Maize** | `RandomForestRegressor` | -5.4% | -11.2% | +0.8% | -7.6% | 25.0% | -5.85% | -11.2% |
| **Rapeseed & Must.**| `RandomForestRegressor` | -3.8% | -10.74%| +2.4% | -10.3% | 25.0% | -5.62% | -10.74% |
