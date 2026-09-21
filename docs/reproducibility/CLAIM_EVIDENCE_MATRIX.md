# Master Claim-to-Evidence Matrix

This matrix links all quantitative, algorithmic, and operational claims to their underlying physical artifacts, verification code paths, and explicit scientific limitations.

```
+-----------------------------------------------------------------------------------------------------------------------------------------------+
| CLAIM_ID    | DOMAIN           | SCIENTIFIC CLAIM                               | AUTHORITATIVE ARTIFACT    | STATUS     | VERIFICATION CODE PATH |
+-----------------------------------------------------------------------------------------------------------------------------------------------+
| CLM-DATA-01 | Canonical Panel  | Unified panel contains exactly 71,601 records  | agricultural_panel.csv    | REPRODUCED | test_day36_repro.py    |
| CLM-DATA-02 | Canonical Panel  | Panel covers exactly 29 verified crops         | agricultural_panel.csv    | REPRODUCED | test_day36_repro.py    |
| CLM-DATA-03 | Canonical Panel  | Panel spans 20 states and 311 districts        | agricultural_panel.csv    | REPRODUCED | test_day36_repro.py    |
| CLM-DATA-04 | Canonical Panel  | Active multi-crop window is 2010–2017          | agricultural_panel.csv    | REPRODUCED | test_day36_repro.py    |
| CLM-DATA-05 | Historical Rice  | Single-crop Rice panel contains 2,469 records  | rice_data_outlier_rem.csv | REPRODUCED | test_day36_repro.py    |
| CLM-MOD-01  | Rice Benchmark   | Rice legacy holdout R² = 0.7866                | forecasting_model_meta    | REPRODUCED | test_day36_repro.py    |
| CLM-MOD-02  | Rice Benchmark   | Rice legacy holdout MAE = 353.01 kg/ha         | forecasting_model_meta    | REPRODUCED | test_day36_repro.py    |
| CLM-MOD-03  | Rice Benchmark   | Rice legacy holdout RMSE = 513.11 kg/ha        | forecasting_model_meta    | REPRODUCED | test_day36_repro.py    |
| CLM-MOD-04  | Rice Benchmark   | Rice legacy holdout MAPE = 18.04%              | forecasting_model_meta    | REPRODUCED | test_day36_repro.py    |
| CLM-GOV-01  | Multi-Crop Gate  | 14 crops qualify as MODEL_READY (Day 18)       | crop_model_readiness.csv  | REPRODUCED | test_day36_repro.py    |
| CLM-GOV-02  | Governance Chain | Oilseeds certified PRODUCTION_READY ML         | final_model_cert.csv      | REPRODUCED | test_day36_repro.py    |
| CLM-GOV-03  | Oilseeds ML      | Oilseeds ML achieves 75.0% fold win rate       | multicrop_model_sel.csv   | REPRODUCED | test_day36_repro.py    |
| CLM-GOV-04  | Oilseeds ML      | Oilseeds ML achieves +12.79% mean gain         | multicrop_model_sel.csv   | REPRODUCED | test_day36_repro.py    |
| CLM-GOV-05  | Sugarcane Raw    | Sugarcane raw unclipped GBDT loses -1.60%      | multicrop_model_sel.csv   | REPRODUCED | test_day36_repro.py    |
| CLM-GOV-06  | Sugarcane Governed| Sugarcane governed GBDT gains +1.19% vs base   | final_model_cert.csv      | REPRODUCED | test_day36_repro.py    |
| CLM-GOV-07  | Sugarcane Governed| Sugarcane achieves 50.0% fold win rate         | final_model_cert.csv      | REPRODUCED | test_day36_repro.py    |
| CLM-GOV-08  | Baseline Crops   | 12 crops certified BASELINE_PRODUCTION         | final_model_cert.csv      | REPRODUCED | test_day36_repro.py    |
| CLM-EXO-01  | Weather Ablation | Pre-season weather showed no gain in 14 crops  | exogenous_model_sel.csv   | REPRODUCED | test_day36_repro.py    |
| CLM-UNC-01  | Uncertainty      | Empirical P10-P90 reflects tree dispersion     | decision_workspace.py     | REPRODUCED | test_day36_repro.py    |
| CLM-UNC-02  | Uncertainty      | Rice holdout empirical coverage was 81.3%      | DAY18_PRE_MODELING_AUDIT  | SUPPORTED  | test_day36_repro.py    |
| CLM-XAI-01  | Explainability   | Marginal Reference Perturbation Attribution    | explainability_engine.py  | REPRODUCED | test_day36_repro.py    |
| CLM-MON-01  | Drift Monitoring | PSI tracks feature distribution shifts          | monitoring_service.py     | REPRODUCED | test_day36_repro.py    |
| CLM-MON-02  | Outcome Eval     | Post-harvest signed bias tracks y_hat - y       | outcome_evaluation.py     | REPRODUCED | test_day36_repro.py    |
| CLM-SCN-01  | Scenario Engine  | Bounded simulations tagged as [SCENARIO]       | decision_workspace.py     | REPRODUCED | test_day36_repro.py    |
| CLM-PROV-01 | Provenance       | Cryptographic SHA-256 digital signatures       | prediction_service.py     | REPRODUCED | test_day36_repro.py    |
+-----------------------------------------------------------------------------------------------------------------------------------------------+
```
