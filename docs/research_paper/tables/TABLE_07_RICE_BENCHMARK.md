# Table 7: Canonical Rice Single-Crop Benchmark Specification

| Metric | Benchmark Value | Unit | Evaluation Split | Physical Evidence Source |
|---|---|---|---|---|
| **Coefficient of Determination ($R^2$)** | **0.7866** | Dimensionless | 2016–2017 Out-of-time Holdout (618 records) | `Models/forecasting_model_metadata.json` |
| **Mean Absolute Error (MAE)** | **353.01** | kg/ha | 2016–2017 Out-of-time Holdout (618 records) | `Models/forecasting_model_metadata.json` |
| **Root Mean Squared Error (RMSE)** | **513.11** | kg/ha | 2016–2017 Out-of-time Holdout (618 records) | `Models/forecasting_model_metadata.json` |
| **Mean Absolute Percentage Error (MAPE)** | **18.04%** | Percent (%) | 2016–2017 Out-of-time Holdout (618 records) | `Models/forecasting_model_metadata.json` |
| **Nominal 80% Uncertainty Coverage** | **81.3%** | Percent (%) | 2016–2017 Out-of-time Holdout (618 records) | `docs/DAY18_PRE_MODELING_AUDIT.md` |
| **Model Pipeline Artifact** | `forecasting_pipeline.pkl` | 10,274,922 bytes | Serialized Random Forest Regressor | SHA-256: `5d64b8f896f5fb9cb...` |
| **Training Population** | 1,851 records | District-Years | 2010–2015 historical training window | `Datasets/rice_data_outlier_removed.csv` |
| **Test Population** | 618 records | District-Years | 2016–2017 out-of-time holdout window | `Datasets/rice_data_outlier_removed.csv` |
