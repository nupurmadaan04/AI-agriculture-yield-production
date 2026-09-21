# Table 11: Master Claim-to-Evidence Matrix

| Claim ID | Claim Summary | Authoritative Artifact | Code Path / Assertion | Reproduction Status | Key Limitation |
|---|---|---|---|---|---|
| **CLM-DATA-01** | Unified panel contains 71,601 records | `agricultural_panel.csv` | `len(df) == 71601` | **REPRODUCED** | 2010–2017 active window |
| **CLM-DATA-02** | Unified panel covers 29 verified crops | `agricultural_panel.csv` | `df['crop'].nunique() == 29` | **REPRODUCED** | Only 14 qualify as MODEL_READY |
| **CLM-DATA-03** | Geographic scope: 20 states, 311 districts | `agricultural_panel.csv` | `df['state'].nunique() == 20` | **REPRODUCED** | Harmonized to 1966 boundaries |
| **CLM-MOD-01** | Rice legacy holdout $R^2 = 0.7866$ | `forecasting_model_metadata.json` | `meta['evaluation']['r2']` | **REPRODUCED** | Single out-of-time test split |
| **CLM-MOD-02** | Rice legacy holdout MAE = 353.01 kg/ha | `forecasting_model_metadata.json` | `meta['evaluation']['mae']` | **REPRODUCED** | Evaluated on 618 holdout records |
| **CLM-GOV-02** | Oilseeds certified `PRODUCTION_READY` ML | `final_model_certification.csv` | `strategy == 'PRODUCTION_READY'` | **REPRODUCED** | Requires district lag availability |
| **CLM-GOV-03** | Oilseeds ML achieves 75% fold win rate | `multicrop_model_selection.csv` | `win_rate == 75.0` | **REPRODUCED** | Evaluated across 2014–2017 origins |
| **CLM-GOV-04** | Oilseeds ML achieves +12.79% mean gain | `multicrop_model_selection.csv` | `mean_mae_gain == 12.79` | **REPRODUCED** | Unbiased residuals across folds |
| **CLM-GOV-05** | Sugarcane raw unclipped GBDT loses -1.60% | `multicrop_model_selection.csv` | `mean_mae_gain == -1.60` | **REPRODUCED** | Unclipped GBDT failed 2015 drought |
| **CLM-GOV-06** | Sugarcane governed GBDT gains +1.19% | `final_model_certification.csv` | `gain_vs_baseline_pct == 1.19` | **REPRODUCED** | Requires 3-$\sigma$ district fallback |
| **CLM-EXO-01** | Pre-season weather showed no gain in 14 crops | `exogenous_model_selection.csv` | `status == 'NO_MEANINGFUL_GAIN'` | **REPRODUCED** | Pre-season district lead times only |
| **CLM-PROV-01** | Forecasts generate SHA-256 digital signatures | `prediction_audit_log.csv` | `provenance_hash.startswith('SHA256')` | **REPRODUCED** | Bitwise reproducible across runs |
