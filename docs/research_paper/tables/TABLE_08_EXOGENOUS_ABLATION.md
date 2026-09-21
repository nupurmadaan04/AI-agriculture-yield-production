# Table 8: Day 22 Exogenous Weather Feature Ablation Matrix

| Ablation Tier ID | Feature Group Composition | Mean MAE Across 14 Crops (kg/ha) | Relative Delta vs. Historical | Crops Where Tier Won | Authoritative Decision |
|---|---|---|---|---|---|
| **$EXP-22A$** | Historical Only (Lag-1, Roll-3, Area Share) | **445.12** | **0.00% (Baseline)** | **14 / 14 (100%)** | **PREFERRED & DEPLOYED** |
| **$EXP-22B$** | Historical + Pre-Season Rainfall Anomalies | 453.38 | +1.85% (Degradation) | 0 / 14 (0%) | REJECTED |
| **$EXP-22C$** | Historical + Pre-Season Temperature Extremes | 454.67 | +2.14% (Degradation) | 0 / 14 (0%) | REJECTED |
| **$EXP-22D$** | Historical + Combined Pre-Season Weather | 460.36 | +3.42% (Degradation) | 0 / 14 (0%) | REJECTED |
| **$EXP-22E$** | Historical + All Exogenous Covariates | 463.74 | +4.18% (Degradation) | 0 / 14 (0%) | REJECTED |
