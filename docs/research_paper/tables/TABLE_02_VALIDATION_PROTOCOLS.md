# Table 2: Scientific Validation Protocols Catalog

| Protocol ID | Protocol Name | Train Window | Test Window | Origins | Population | Primary Metric | Primary Outcome |
|---|---|---|---|---|---|---|---|
| **Protocol A** | Multi-Crop Screening | 2010–2017 | N/A | Cross-sectional | All 29 crops | Sufficiency ($N \ge 100$) | 14 crops classified as `MODEL_READY` |
| **Protocol B** | Initial Crop Holdout | 2010–2015 | 2016–2017 | 2016 | 14 crops | Out-of-time MAE, $R^2$ | Single-split baseline benchmarking |
| **Protocol C** | Multi-Origin Walk-Forward | Expanding ($<T$) | Isolated ($T$) | 2014, 2015, 2016, 2017 | 14 crops | Fold Win Rate, Mean MAE | Exposed drought regime instability |
| **Protocol D** | Model Selection Gate | Expanding ($<T$) | Isolated ($T$) | 2014–2017 | 14 crops | Win Rate $\ge 75\%$, CV | Only Oilseeds passed unconstrained |
| **Protocol E** | Exogenous Feature Ablation | Expanding ($<T$) | Isolated ($T$) | 2014–2017 | 14 crops | Delta MAE vs. $EXP-22A$ | Pre-season weather showed no gain |
| **Protocol F** | Governed Strategy Certification | Expanding ($<T$) | Isolated ($T$) | 2014–2017 | 14 crops | Governed Strategy MAE | Sugarcane 3-$\sigma$ clip restored +1.19% gain |
| **Protocol G** | Legacy Single-Crop Rice | 2010–2015 (1,851) | 2016–2017 (618) | 2016 | Rice districts | $R^2$, MAE, RMSE, MAPE | $R^2=0.7866$, MAE 353.01 kg/ha |
