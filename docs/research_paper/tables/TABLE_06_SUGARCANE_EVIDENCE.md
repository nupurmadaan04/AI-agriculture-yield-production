# Table 6: Sugarcane Dual-Metric Resolution & Governance Evidence

| Metric / Attribute | Protocol 1: Raw Unconstrained GBDT | Protocol 2: Governed GBDT + 3-$\sigma$ Clip | Evidence Source | Resolution & Scientific Meaning |
|---|---|---|---|---|
| **Certified Status** | `RESEARCH_CANDIDATE` | `CONDITIONAL_PRODUCTION` | `final_model_certification.csv` | Certified with mandatory variance fallback |
| **Strategy MAE** | 1,510.45 kg/ha | **1,467.97 kg/ha** | `final_model_certification.csv` | Governed clipping lowers overall error |
| **Baseline MAE** | 1,485.70 kg/ha | 1,485.70 kg/ha | `final_model_certification.csv` | Historical district mean persistence |
| **Aggregate Gain (%)**| **-1.60%** (Loss) | **+1.19%** (Gain) | `final_model_certification.csv` | **Authoritative resolution of dual figures** |
| **Fold 2 (2015 Drought)**| -10.19% (Severe tail error) | -5.78% (Bounded by clip) | `final_validation_results.csv` | 3-$\sigma$ fallback bounds extreme drought loss |
| **Fold 3 (2016)** | +9.75% | +9.75% | `final_validation_results.csv` | Model excels in post-drought recovery |
| **Fold 4 (2017)** | +3.96% | +8.44% | `final_validation_results.csv` | Fallback stabilizes anomalous district extremes |
| **Fold Win Rate** | 50.0% (2 of 4 folds) | 50.0% (2 of 4 folds) | `final_model_certification.csv` | Wins Folds 3 and 4; loses Folds 1 and 2 |
| **Operational Rule** | None (Unsafe) | Fallback to mean if $|\hat{y} - \mu| > 3\sigma$ | `forecast_strategy_registry.json` | Mandatory district variance clipping |
