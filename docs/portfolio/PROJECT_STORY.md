# The Project Story: Evolution of an Evidence-Governed Forecasting Platform

---

## 1. Why Did This Project Start?

Agricultural yield prediction in India is typically characterized by high uncertainty, decentralized administrative reporting, and vulnerability to severe monsoon oscillations. For government planners, buffer stock managers, and commodity traders, understanding district-level yields 3 to 6 months before harvest is vital for food security and market stabilization.

Most academic papers and commercial dashboards approach this problem by scraping regional statistics, feeding every available variable into a black-box gradient boosting or deep learning model, and claiming near-perfect predictive accuracy ($R^2 > 0.90$).

We initiated this project to answer a fundamental engineering and econometrics question:  
**"Can machine learning genuinely outperform simple historical persistence baselines under realistic, leakage-free operational constraints, and can we safely serve these models in production without risking catastrophic failure during climate shocks?"**

---

## 2. What Was the First Version? (Days 1–17)

The earliest version of the platform focused on single-crop Rice modeling using historical ICRISAT district tables (1966–2017) and an initial panel of 2,469 observations.

During early exploration, we noticed an immediate trap: if concurrent harvest production $P_{i, t}$ was included in feature matrices, regression models easily achieved $R^2 = 0.99$. However, because yield is algebraically computed as:
$$\text{Yield}_{i, t} = \frac{\text{Production}_{i, t}}{\text{Area}_{i, t}} \times 1000$$
predicting yield from harvest production is not forecasting—it is mathematical reconstruction. In an operational pre-season scenario, harvest production is unknown.

**The First Pivot**: We implemented an absolute leakage barrier, strictly restricting feature engineering to pre-sowing variables: autoregressive lags ($y_{t-1}$, $y_{t-2}$), 3-year rolling means, and pre-season crop cultivated area shares. We also expanded the data foundation from single-crop Rice to a unified panel of **71,601 records covering 29 crops across 20 states and 311 districts (2010–2017 active panel)**.

---

## 3. What Failed? The Single-Split Illusion (Days 18–20)

In Day 19, we trained multi-crop models on an initial chronological split: train on $\le 2015$, test on $2016–2017$. On that single split, machine learning models for multiple crops appeared to perform reasonably well.

However, in Day 20, we subjected the models to **4-fold expanding walk-forward temporal cross-validation** across origins $T \in \{2014, 2015, 2016, 2017\}$:
- In Fold 2 (Origin 2015), India suffered a massive pan-India drought (monsoon rainfall deficit ~14%).
- Complex tree models (Random Forest, Gradient Boosting) failed catastrophically. Because tree ensembles cannot extrapolate outside their observed feature manifold, they severely over-predicted yields in rainfed districts.
- For 12 out of 14 crops (including Rice and Wheat), simple historical persistence baselines (Historical District Mean) beat the machine learning models across the full four-fold tournament.
- In Sugarcane, raw unconstrained GBDT suffered a severe -10.19% loss in the 2015 drought fold, dragging its 4-fold mean performance down to a net **-1.60% loss**.

---

## 4. What Was Changed? The Governance Pivot (Days 21–25)

Instead of forcing machine learning into production for all crops or cherry-picking favorable splits, we embraced the empirical reality and built a **governed strategy registry**:

1. **Mandate Baselines for Losers (`BASELINE_PRODUCTION`)**:
   For the 12 crops where statistical district mean persistence proved more stable than ML (Rice, Wheat, Chickpea, Maize, Sorghum, etc.), we certified the baseline as the primary production strategy.
2. **Deploy Unconstrained ML for Proven Winners (`PRODUCTION_READY`)**:
   For **Oilseeds**, Random Forest won 3 out of 4 folds (75% win rate), achieved a **+12.79% mean MAE gain** over baseline, and bounded worst-fold degradation to -2.34%. It was certified for unconstrained production serving.
3. **Engineer Variance Clipping for Volatile Winners (`CONDITIONAL_PRODUCTION`)**:
   For **Sugarcane**, GBDT had strong predictive power during normal years but severe tail risks during droughts. We implemented an automated **3-sigma district variance clipping fallback**:
   $$\text{If } |\hat{y} - \mu_{\text{dist}}| > 3\sigma_{\text{dist}} \implies \text{Deploy } \mu_{\text{dist}}$$
   This fail-safe bounded the 2015 drought loss to -5.78% while preserving gains in 2016 (+9.75%) and 2017 (+8.44%), achieving an aggregate **+1.19% MAE gain** over baseline persistence across 1,193 test observations.

---

## 5. What Was the Weather Surprise? (Day 22 Ablation)

We hypothesized that incorporating pre-season gridded rainfall anomalies and temperature extremes would improve forecast accuracy. We executed a rigorous 5-tier ablation tournament ($EXP-22A$ through $EXP-22E$) across all 14 crops.

**The Finding**: Pre-season district weather aggregations degraded MAE or increased variance across **100% of the evaluated crops**. Historical autoregressive lags already encoded local soil moisture and technological inertia; adding noisy pre-season district weather signals degraded out-of-sample accuracy. We documented this negative result transparently and retained the historical-only feature tier ($EXP-22A$) as authoritative.

---

## 6. Production Hardening & Serving (Days 26–35)

With the scientific models frozen and certified, we engineered the production serving layer:
- **FastAPI Inference Microservice**: Sub-50ms P95 latency with Pydantic v2 schemas and pre-inference certification guards.
- **Cryptographic Provenance**: Every prediction generates an SHA-256 digital signature recorded in an append-only audit log.
- **MLOps Observability**: Continuous tracking of Population Stability Index (PSI) feature drift and post-harvest directional signed bias ($\hat{y} - y$).
- **Decision Workspace**: Interactive React/TypeScript dashboard synthesizing multi-evidence briefs (forecast, baseline, uncertainty, attribution, drift) and evaluating bounded what-if acreage allocation scenarios via SLSQP optimization.
- **Production Hardening**: Docker Compose deployment fronted by Nginx reverse proxy, non-root execution (`appuser`), fail-closed `/ready` probe (HTTP 503), WCAG 2.1 AA accessible UI, and 580+ automated tests.

---

## 7. Scientific Audit & Publication Packaging (Days 36–38)

In Days 36 and 37, we locked the codebase under an **absolute scientific freeze**:
- Verified all 25 empirical repository claims in a master claim register.
- Published a 15-chapter research paper manuscript, 11 publication tables, and 6 architecture figures.
- Established complete bitwise cryptographic reproducibility across all dataset panels and model weights.
- Productized the platform for recruiters, engineers, and researchers.

**The Final System**: A platform that proves responsible machine learning engineering is not about deploying the most complex model everywhere, but about knowing when to use ML, when to use a baseline, and how to govern decisions with empirical evidence.
