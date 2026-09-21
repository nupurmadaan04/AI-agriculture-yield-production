# Engineering & Scientific Trade-Offs: "Why Not Just X?"

This document provides technical justifications for key design and architectural choices made across the platform, contrasting chosen strategies against common alternatives.

---

## 1. Machine Learning & Econometric Decisions

### Q1: "Why crop-specific models instead of one single global model?"
- **Alternative**: Train a single large gradient boosting or neural network model pooling all 29 crops with a one-hot encoded `crop_id` feature.
- **Why Rejected**: Agricultural commodities operate under incompatible agronomic regimes. Sugarcane grows over 14 months under heavy irrigation, whereas Kharif pulses grow over 90 days under rainfed conditions. In Day 18 screening, pooling commodities diluted localized spatial signals and allowed large-volume crops (like Rice) to dominate gradient updates, obscuring catastrophic error spikes in smaller crops. Crop-specific models isolate spatial-temporal variance and permit independent governance.

### Q2: "Why benchmark against statistical persistence baselines?"
- **Alternative**: Benchmark machine learning models only against other complex models (e.g. XGBoost vs LightGBM vs Random Forest).
- **Why Rejected**: In agricultural economics, historical persistence (Historical District Mean) is exceptionally competitive due to soil characteristics, irrigation capital, and technological inertia. If an ML model cannot beat a naive historical district mean across sequential climate shocks, deploying it introduces operational complexity, latency, and failure modes with zero empirical gain.

### Q3: "Why walk-forward validation instead of standard K-Fold CV?"
- **Alternative**: Randomly split the 71,601 records into 80% train and 20% test partitions.
- **Why Rejected**: Random splitting causes severe temporal leakage. If 2015 was an extreme drought year, random splitting places 2015 records in both train and test sets, allowing the model to memorize the weather shock. Walk-forward validation enforces strict chronological ordering ($\text{Train} < T$, $\text{Test} = T$), faithfully simulating how models encounter future unforeseen climate regimes.

### Q4: "Why refuse to deploy machine learning for all 14 crops?"
- **Alternative**: Force the best-performing ML model into production for every crop regardless of baseline comparison.
- **Why Rejected**: Walk-forward testing proved that for 12 of 14 crops, ML degraded accuracy or suffered severe tail errors during the 2015 drought. Forcing ML into production when a simpler baseline is more accurate violates core engineering principles. Good MLOps means deploying the most reliable solution, not the most complex one.

### Q5: "Why implement automated fallback strategies?"
- **Alternative**: Return an HTTP 500 error or raise an exception whenever an input feature (e.g. lag-1 yield) is missing.
- **Why Rejected**: Agricultural reporting is decentralized; newly created districts or lagging administrative returns frequently lack prior-year data. Rather than failing in production, the system executes an automated fallback sequence ($\text{ML} \to \text{District Mean} \to \text{State Mean}$), ensuring 100% serving availability while transparently flagging the fallback in response metadata.

---

## 2. Explainability & Uncertainty Decisions

### Q6: "Why empirical P10–P90 ensemble intervals instead of Bayesian or Gaussian confidence intervals?"
- **Alternative**: Assume Gaussian residual distributions and report $\hat{y} \pm 1.96 \cdot \hat{\sigma}$, or train a Bayesian neural network.
- **Why Rejected**: Agricultural yield residuals exhibit heavy skewness, kurtosis, and heteroskedasticity during drought regimes. Gaussian assumptions produce invalid negative yield bounds. Conversely, empirical tree dispersion directly queries the 150 individual decision trees in the ensemble:
  $$\hat{y}_{P10} = \text{Quantile}_{0.10}(\{f_b(X)\}), \quad \hat{y}_{P90} = \text{Quantile}_{0.90}(\{f_b(X)\})$$
  This captures non-parametric ensemble disagreement and achieved 81.3% empirical coverage on benchmark test splits without ungrounded parametric assumptions.

### Q7: "Why Marginal Reference Perturbation Attribution instead of pure KernelSHAP?"
- **Alternative**: Run computationally heavy KernelSHAP across all features for every live inference request.
- **Why Rejected**: KernelSHAP requires generating thousands of background sample permutations per prediction, driving inference latency above 1,500ms. In contrast, Marginal Reference Perturbation Attribution compares features against the district's historical median vector $\mathbf{x}^{(0)}$ in closed form, delivering transparent local attributions within our sub-50ms latency budget.

### Q8: "Why strictly separate Scenarios (`[SCENARIO]`) from Forecasts (`[PREDICTED]`)?"
- **Alternative**: Allow users to tweak weather/acreage sliders and label the output as an "Updated AI Forecast".
- **Why Rejected**: Conflating hypothetical what-if simulations with empirically validated predictions misleads policy-makers. Simulations perturb parameters within a trained mathematical manifold but cannot simulate real-world agronomic interventions. Enforcing semantic badge separation (`[PREDICTED]` vs `[SCENARIO]`) prevents users from treating hypothetical calculations as empirical ground truth.

---

## 3. Software Architecture & Infrastructure Decisions

### Q9: "Why cryptographic SHA-256 provenance logging?"
- **Alternative**: Simply write log statements to stdout or log files using standard Python logging.
- **Why Rejected**: In public sector decision-making, agricultural forecasts influence procurement budgets and food reserve allocations. Months after a prediction is served, auditors must be able to verify whether a forecast was altered. Generating an immutable SHA-256 hash linking the request ID, inputs, model version, and dataset hash creates a tamper-evident audit trail.

### Q10: "Why FastAPI instead of Django or Flask?"
- **Alternative**: Build the backend in Django or Flask.
- **Why Rejected**: FastAPI provides native asynchronous request processing, automatic OpenAPI documentation, and Pydantic v2 validation. Its lightweight ASGI runtime easily achieves our sub-50ms P95 latency target with minimal memory overhead (~350 MB RSS).

### Q11: "Why Nginx reverse proxy fronting Docker containers?"
- **Alternative**: Expose the FastAPI backend port directly to the host machine.
- **Why Rejected**: Direct exposure bypasses perimeter rate limiting and leaves the application vulnerable to host port enumeration. Nginx handles client TLS termination, serves static frontend assets efficiently, injects OWASP security headers (`X-Frame-Options: DENY`, `nosniff`), and isolates backend port 8000 on an internal container bridge network.
