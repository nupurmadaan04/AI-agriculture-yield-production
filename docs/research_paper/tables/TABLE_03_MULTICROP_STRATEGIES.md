# Table 3: Governed Multi-Crop Production Strategies

| Commodity | Governed Status | Primary Deployed Strategy | Fallback Strategy | Primary Strategy MAE (kg/ha) | Baseline MAE (kg/ha) | Governed Gain vs. Baseline |
|---|---|---|---|---|---|---|
| **Oilseeds** | `PRODUCTION_READY` | Historical ML (`RandomForestRegressor`) | Historical District Mean (Sparse Fallback) | 549.67 | 616.60 | **+10.85%** |
| **Sugarcane** | `CONDITIONAL_PRODUCTION` | Historical ML (`GradientBoostingRegressor`) | Historical District Mean (3-$\sigma$ Fallback) | 1,467.97 | 1,485.70 | **+1.19%** |
| **Chickpea** | `BASELINE_PRODUCTION` | Historical District Mean Persistence | District 3-Year Rolling Mean | 260.35 | 263.70 | +1.27% (Baseline Preferred) |
| **Kharif Sorghum** | `BASELINE_PRODUCTION` | Historical District Mean Persistence | District 3-Year Rolling Mean | 294.65 | 296.69 | +0.69% (Baseline Preferred) |
| **Minor Pulses** | `BASELINE_PRODUCTION` | Historical District Mean Persistence | District 3-Year Rolling Mean | 345.16 | 345.16 | 0.00% (Baseline Preferred) |
| **Maize** | `BASELINE_PRODUCTION` | Historical District Mean Persistence | District 3-Year Rolling Mean | 638.38 | 638.38 | 0.00% (Baseline Preferred) |
| **Wheat** | `BASELINE_PRODUCTION` | Historical District Mean Persistence | District 3-Year Rolling Mean | 381.65 | 381.65 | 0.00% (Baseline Preferred) |
| **Rice** | `BASELINE_PRODUCTION` | Historical District Mean Persistence | District 3-Year Rolling Mean | 310.28 | 310.28 | 0.00% (Baseline Preferred) |
| **Sesamum** | `BASELINE_PRODUCTION` | Historical District Mean Persistence | District 3-Year Rolling Mean | 137.44 | 137.44 | 0.00% (Baseline Preferred) |
| **Pigeonpea** | `BASELINE_PRODUCTION` | Historical District Mean Persistence | District 3-Year Rolling Mean | 282.35 | 282.35 | 0.00% (Baseline Preferred) |
| **Rapeseed & Mustard** | `BASELINE_PRODUCTION` | Historical District Mean Persistence | District 3-Year Rolling Mean | 193.33 | 193.33 | 0.00% (Baseline Preferred) |
| **Groundnut** | `BASELINE_PRODUCTION` | Historical District Mean Persistence | District 3-Year Rolling Mean | 287.40 | 287.40 | 0.00% (Baseline Preferred) |
| **Sorghum** | `BASELINE_PRODUCTION` | Historical District Mean Persistence | District 3-Year Rolling Mean | 264.02 | 264.02 | 0.00% (Baseline Preferred) |
| **Pearl Millet** | `BASELINE_PRODUCTION` | Historical District Mean Persistence | District 3-Year Rolling Mean | 260.24 | 260.24 | 0.00% (Baseline Preferred) |
