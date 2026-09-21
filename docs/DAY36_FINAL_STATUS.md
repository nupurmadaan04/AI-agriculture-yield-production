# DAY 36 — FINAL STATUS

## Summary Scorecard

Dataset:
REPRODUCED (71,601 canonical panel records across 29 crops, 20 states, and 311 districts independently verified)

Rice benchmark:
REPRODUCED (R² = 0.7866, MAE = 353.01 kg/ha, RMSE = 513.11 kg/ha, MAPE = 18.04% on 2016–2017 holdout verified)

Oilseeds:
REPRODUCED (PRODUCTION_READY certification verified with 75% fold win-rate and +12.79% mean gain over baseline)

Sugarcane:
REPRODUCED (Dual-figure discrepancy resolved: -1.60% raw unclipped GBDT vs +1.19% governed GBDT with 3-sigma fallback)

Baseline crop:
REPRODUCED (12 crops including Rice and Wheat verified to default to Historical District Mean persistence)

Uncertainty:
REPRODUCED WITH LIMITATIONS (Empirical P10–P90 ensemble spread verified as tree dispersion, not frequentist confidence interval)

XAI:
REPRODUCED (Marginal Reference Perturbation Attribution and Tree SHAP verified with explicit non-causal disclaimers)

Monitoring:
REPRODUCED (Population Stability Index drift tracking and post-harvest signed bias evaluation verified)

Scenario:
REPRODUCED (Hypothetical what-if simulations bounded to trained manifolds and strictly tagged as SCENARIO)

Provenance:
REPRODUCED (Cryptographic SHA-256 digital provenance signatures verified bitwise across restarts)

Overall:
REPRODUCIBILITY VERIFIED WITH LIMITATIONS

Backend tests:
96 passed across Days 33–36 production, deployment, contract, and reproducibility suites (581 total tests collected across repository)

Frontend build:
PASS (Built cleanly in 7.89s with 0 TypeScript errors)

Scientific regression:
PASS (100% scientific freeze maintained; zero weights, hyperparameters, or data values altered)

Known inconsistencies:
1. Sugarcane improvement figures (-1.60% vs +1.19%): Mathematically resolved as unclipped raw GBDT vs governed GBDT with 3-sigma variance fallback.
2. Dataset record counts (2,469 vs 71,601): Resolved as single-crop Rice historical subset vs 29-crop unified canonical panel.

Known limitations:
1. Historical Time Horizon: Canonical panel observations span 2010–2017; future/unharvested years cannot be evaluated against ground truth.
2. Spatial Aggregation: Predictions represent district-level geographic averages; field-level micro-variation is not captured.
3. Pre-Season Information Limit: Models utilize pre-season lags; unforeseen in-season extreme weather cannot be anticipated.
4. Non-Causal Nature: Neither feature attributions nor scenario perturbations constitute evidence of biological cause-and-effect.
5. Local Volume Persistence: Prediction audit logs and operational telemetry operate as host volume files rather than a distributed cloud RDBMS.
