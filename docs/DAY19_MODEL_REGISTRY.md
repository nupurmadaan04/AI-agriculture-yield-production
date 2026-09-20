# Day 19: Multi-Crop Model Registry Specification

## 1. Registry Architecture & Immutability

The multi-crop model registry (`Models/multicrop/model_registry.json`) records the complete provenance, SHA-256 cryptographic checksums, training windows, and evaluation metrics for every model-ready agricultural crop.

Existing Rice model registry entries remain completely immutable and strictly isolated in `Models/forecasting_model_metadata.json` and `Models/forecasting_pipeline.pkl`.

---

## 2. Model Registry Schema

```json
{
  "model_id": "multicrop_maize_forecaster",
  "crop": "Maize",
  "algorithm": "RandomForestRegressor",
  "version": "1.0.0",
  "dataset_version": "AGRI_PANEL_1.0",
  "feature_set_version": "PRE_SEASON_LAG_V1",
  "training_period": "2011–2015",
  "evaluation_period": "2016–2017",
  "train_records": 1394,
  "test_records": 559,
  "metrics": {
    "mae": 744.73,
    "rmse": 1056.28,
    "r2": 0.4958,
    "mape": 34.62,
    "smape": 31.05
  },
  "baseline_comparison": {
    "baseline_model": "Historical District Mean",
    "baseline_mae": 763.91,
    "mae_improvement_pct": 2.51
  },
  "status": "ACCEPTED",
  "artifact_path": "Models/multicrop/maize/model_pipeline.pkl",
  "sha256": "...",
  "created_at": "..."
}
```

---

## 3. Directory Layout

```
Models/
├── multicrop/
│   ├── model_registry.json
│   ├── maize/
│   │   ├── model_pipeline.pkl
│   │   └── model_metadata.json
│   ├── sesamum/
│   │   ├── model_pipeline.pkl
│   │   └── model_metadata.json
│   ├── pigeonpea/
│   │   ├── model_pipeline.pkl
│   │   └── model_metadata.json
│   ├── sugarcane/
│   │   ├── model_pipeline.pkl
│   │   └── model_metadata.json
│   └── ... (all 14 crop folders)
```
