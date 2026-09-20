# Day 21: Final Executive Scorecard & Artifact Registry

## Executive Decision Scorecard

```
======================================================================================================
                     AI AGRICULTURE INTELLIGENCE PLATFORM — DAY 21 SCORECARD
                    CROP-SPECIFIC MODEL SELECTION & ERROR DIAGNOSIS FRAMEWORK
======================================================================================================
  [1] ROBUST_ML (1 Commodity):
      • Oilseeds (RandomForestRegressor) — Win Rate: 75.0%, Mean Gain: +12.79%, Median Gain: +5.18%

  [2] ML_WITH_CONDITIONS (2 Commodities):
      • Chickpea (GradientBoostingRegressor) — Win Rate: 75.0%, Mean Gain: +2.31%, Median Gain: +6.11%
      • Kharif Sorghum (RandomForestRegressor) — Win Rate: 50.0%, Mean Gain: +1.30%, Median Gain: +0.79%

  [3] RESEARCH_CANDIDATE (4 Commodities — Shadow Mode / Exogenous Research):
      • Minor Pulses (GradientBoostingRegressor) — Win Rate: 50.0%, Mean Gain: -5.75%
      • Maize (RandomForestRegressor) — Win Rate: 25.0%, Mean Gain: -2.63%
      • Wheat (GradientBoostingRegressor) — Win Rate: 25.0%, Mean Gain: -14.73%
      • Sugarcane (GradientBoostingRegressor) — Win Rate: 25.0%, Mean Gain: -0.66%

  [4] BASELINE_PREFERRED (7 Commodities — Historical District Mean / Naive Persistence Primary):
      • Rice, Sesamum, Pigeonpea, Rapeseed and Mustard, Groundnut, Sorghum, Pearl Millet

  [5] INSUFFICIENT_EVIDENCE (0 Commodities)
======================================================================================================
  RICE STANDALONE REGRESSION (Day 9): PRESERVED & UNTOUCHED (R² = 0.7866, MAE = 353.01 kg/ha)
  SYSTEM TEST SUITE: 65 / 65 PASSING (100% PASS RATE)
  FRONTEND UI BUILD: PASS (Vite + TypeScript Clean Build)
======================================================================================================
```

---

## Artifact & Dataset Summary

### CSV Artifacts Created (`Datasets/metadata/`)
1. `multicrop_error_diagnosis.csv` (14 crop summary rows with error quantiles, dispersion bands, and stability CV)
2. `multicrop_error_regimes.csv` (42 rows decomposing Low, Normal, High yield regimes)
3. `multicrop_year_error_analysis.csv` (56 rows tracking multi-origin temporal error trajectories and 2016 shocks)
4. `multicrop_district_error_analysis.csv` (4,313 district rows satisfying $N \ge 3$)
5. `multicrop_feature_stability.csv` (84 feature rows with stability score $S_{\text{feat}}$)
6. `multicrop_feature_timing_audit.csv` (8 feature rows auditing pre-season timing safety)
7. `multicrop_model_selection.csv` (14 selection rows with 3-day lineage)
8. `multicrop_forecasting_strategy.csv` (14 operational policy rows with evidence strength scores)

### Python Modules Created (`src/`)
1. `src/multicrop_error_diagnosis.py`
2. `src/multicrop_feature_diagnostics.py`
3. `src/multicrop_model_selection.py`
4. `src/multicrop_forecasting_strategy.py`

### Backend REST API Endpoints (`backend/main.py`)
1. `GET /api/modeling/diagnosis`
2. `GET /api/modeling/diagnosis/{crop}`
3. `GET /api/modeling/diagnosis/{crop}/errors`
4. `GET /api/modeling/diagnosis/{crop}/districts`
5. `GET /api/modeling/diagnosis/{crop}/years`
6. `GET /api/modeling/diagnosis/{crop}/features`
7. `GET /api/modeling/selection`
8. `GET /api/modeling/selection/{crop}`
9. `GET /api/modeling/forecasting-strategy`
10. `GET /api/modeling/forecasting-strategy/{crop}`

### Frontend Dashboard (`frontend/src/pages/ModelingReadiness.tsx`)
- Interactive **Model Diagnosis & Strategy (Day 21)** tab with KPI cards, crop selector pills, operational strategy callouts, diagnostic scorecard, regime breakdown tables, year-by-year trajectories, district error breakdown, and feature timing audits.
