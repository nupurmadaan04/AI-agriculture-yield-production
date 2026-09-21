# 2. Related Work

The development of operational crop yield intelligence intersects several major fields of statistical modeling, machine learning, and production system governance.

### 2.1 Machine Learning in Agricultural Yield Forecasting

Over the past two decades, empirical crop yield estimation has transitioned from traditional agro-meteorological crop growth simulation models (such as DSSAT and APSIM) toward statistical and machine learning approaches. In their comprehensive review, van Klompenburg et al. (2020) analyzed hundreds of machine learning yield prediction studies, observing that Random Forests (Breiman, 2001) and Gradient Boosted Decision Trees (Friedman, 2001) consistently achieve competitive predictive performance across diverse crops and geographic scales. Similarly, Lobell & Field (2007) demonstrated that panel regression and non-linear statistical models can effectively capture district- and state-level yield responses to climatic factors.

However, existing literature predominantly reports accuracy metrics derived from random $k$-fold cross-validation or single arbitrary train/test holdouts. As highlighted by Roberts et al. (2017), spatial and temporal autocorrelation in environmental data frequently invalidates standard cross-validation assumptions, leading to inflated performance estimates and acute vulnerability under out-of-sample temporal shifts.

### 2.2 Temporal Cross-Validation & Walk-Forward Testing

In time-series econometrics and operational forecasting, expanding window walk-forward validation (also termed rolling-origin evaluation or temporal cross-validation) is recognized as the gold standard for assessing genuine predictive capability (Hyndman & Athanasopoulos, 2018). In agricultural contexts, evaluating models across sequential historical origins ($T, T+1, T+2, \dots$) without future lookahead is vital because climatic shocks (e.g. El Niño/La Niña cycles) create non-stationary distribution shifts across consecutive agricultural years.

### 2.3 Exogenous Meteorological Features & The Lead-Time Dilemma

A substantial body of research incorporates satellite-derived vegetation indices (e.g. NDVI/EVI) and gridded weather data (temperature extremes, precipitation anomalies) into yield models. While in-season meteorological features observed during critical reproductive crop stages (flowering, grain-filling) possess strong biological correlation with final harvest yields, operational *pre-season* forecasting requires generating predictions prior to sowing. At this early decision horizon, in-season weather is fundamentally unknown, restricting exogenous features to pre-season lead times (pre-monsoon rainfall, soil moisture carryover). Evaluating whether pre-season weather aggregations provide incremental value beyond autoregressive yield lags under rigorous walk-forward protocols remains a critical, under-investigated empirical question.

### 2.4 Explainability, Model Governance, and Responsible ML

As machine learning systems transition into public-sector policy and agricultural decision support, algorithmic interpretability becomes essential. Methodologies such as Shapley additive explanations (Lundberg & Lee, 2017) and reference perturbation attribution provide post-hoc local attributions. Crucially, recent literature in responsible AI emphasizes that feature sensitivity must not be conflated with agronomic causality. Furthermore, operational frameworks require continuous monitoring for covariate distribution shifts using information-theoretic metrics such as the Population Stability Index (Yurdakul, 2010), accompanied by fail-safe fallback governance that prevents catastrophic model outputs.
