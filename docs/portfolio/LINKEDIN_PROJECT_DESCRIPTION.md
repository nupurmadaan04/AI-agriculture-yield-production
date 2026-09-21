# LinkedIn & Professional Portfolio Post

---

### Option 1: Long-Form Technical LinkedIn Post (Recommended)

Most applied machine learning projects in agriculture follow a predictable playbook: collect data, train an XGBoost or Random Forest model, report a 95% R² score on a random train/test split, and conclude that AI has solved yield prediction.

When deployed in the real world, these models frequently fail.

Over the past several weeks, I built the **Agricultural Forecasting & Decision Intelligence Platform** to address the actual challenges of district-level crop forecasting in India: spatial heterogeneity, data leakage, and out-of-distribution climate shocks.

Here is what the architecture actually looks like:

1. **Zero-Leakage Pre-Season Engineering**:
   Many papers predict yield using harvest-year production. Since Yield = Production / Area, that is 100% target leakage. We built a strict pre-season feature pipeline using only autoregressive lags and rolling multi-year moving averages across 71,601 panel records (29 crops, 311 districts, 1966–2017 historical context; 2010–2017 active panel).

2. **Walk-Forward Validation Over Random Splits**:
   Random train/test splits leak weather patterns across test sets. We evaluated models across 4 expanding walk-forward temporal origins (2014, 2015, 2016, 2017). Origin 2015 stress-tested models against a major pan-India drought.

3. **Responsible Model Governance**:
   We did not force machine learning into every crop. Models were benchmarked against statistical district mean persistence baselines.
   - For **Oilseeds**, Random Forest won 75% of folds and delivered a **+12.79% mean MAE improvement**.
   - For **Sugarcane**, raw GBDT failed during the 2015 drought (-10.19% fold loss). We implemented runtime 3-sigma variance clipping, restoring an aggregate **+1.19% MAE gain**.
   - For **12 major crops** (including Rice and Wheat), statistical baselines consistently outperformed ML across climate shocks, so we certified baselines for production deployment.

4. **Production Serving & Provenance**:
   The backend is built in FastAPI (<50ms P95 latency) and generates a cryptographic SHA-256 digital signature for every prediction, tracking exact model versions and feature hashes in an append-only audit log.

5. **MLOps & Decision Intelligence**:
   Live feature drift is tracked via Population Stability Index (PSI), and post-harvest census data evaluates directional signed bias. In the React/TypeScript Decision Workspace, users explore bounded what-if scenarios (SLSQP optimization) with non-causal feature attributions and empirical P10–P90 tree dispersion intervals.

The full repository includes a 15-chapter research paper draft, 11 publication tables, 6 system diagrams, formal model cards, and 580+ automated tests in Docker.

Check out the code, research paper, and architecture on GitHub: [Repository Link]

#MachineLearning #DataScience #MLOps #FastAPI #AgricultureAI #TimeSeries #ReproducibleResearch

---

### Option 2: Short-Form Featured Project Summary (For Portfolio / Resume Links)

**AI Agriculture Yield Forecasting & Decision Intelligence Platform**
- **Architecture**: Multi-crop district-level forecasting system integrating pre-season feature engineering, temporal walk-forward validation, model governance, explainability, and MLOps observability.
- **Dataset**: Harmonized longitudinal panel of 71,601 records across 29 crops, 20 states, and 311 districts (2010–2017 active window; 1966–2017 historical context).
- **Core Results**: Certified unconstrained ML for Oilseeds (+12.79% MAE gain, 75% win-rate) and conditional ML with 3-sigma variance clipping for Sugarcane (+1.19% gain). Mandated statistical district mean persistence for 12 crops where ML degraded under drought shocks.
- **Serving & Reliability**: Sub-50ms FastAPI inference, SHA-256 digital provenance hashing, Population Stability Index (PSI) drift monitoring, and Docker Compose deployment with Nginx reverse proxy and 580+ automated tests.
- **Tech Stack**: Python, FastAPI, Scikit-learn, Pandas, TypeScript, React 18, Vite, Docker, Nginx, Pytest.
