# 12. Research Limitations

To ensure full transparency, this section details the explicit limitations of the dataset, empirical methodology, models, and deployment architecture.

---

### 12.1 Dataset & Temporal Scope Limitations

1. **Temporal Horizon (2010–2017)**:
   The canonical multi-crop panel covers the agricultural years 2010 through 2017. While historical ICRISAT records provide longer-term context, post-2017 district yield statistics were not available in harmonized, cleaned format at the time of study. Consequently, ground-truth outcome evaluation is strictly constrained to historical test years.
2. **Spatial Aggregation Level**:
   Data is aggregated at the administrative district level (~300,000 to 1,000,000 hectares per district). District averages inevitably obscure farm-level micro-variation, soil heterogeneity, local topography, and localized farmer management practices.
3. **Data Sparsity in Secondary Crops**:
   While major crops (Rice, Wheat, Oilseeds) possess dense coverage across hundreds of districts, secondary crops (e.g. Minor Pulses, Sesamum) exhibit geographic sparsity and historical reporting discontinuities that limit statistical power.

---

### 12.2 Methodological & Modeling Limitations

1. **Pre-Season Information Boundary**:
   Forecasts are generated prior to planting using pre-season indicators. In-season extreme meteorological events—such as unseasonal monsoon breaks, localized hailstorms, flooding, or mid-season pest infestations (e.g. locust swarms)—occur after the pre-season decision origin and cannot be captured by pre-season models.
2. **Empirical Nature of Uncertainty Intervals**:
   The P10–P90 uncertainty intervals represent the empirical dispersion across 150 decision tree estimators. They capture epistemic model parameter variance within the trained feature manifold; they do not represent formal frequentist confidence intervals or Bayesian posterior probabilities.
3. **Absence of Agronomic Causality**:
   Feature attributions generated via Marginal Reference Perturbation or Tree SHAP describe mathematical sensitivities of the learned estimators. They reflect correlation in historical observational data and must not be used to infer agronomic cause-and-effect.
4. **Negative Exogenous Result Scope**:
   The finding that pre-season weather features yielded `NO_MEANINGFUL_GAIN` is specific to the evaluated district monthly aggregations, pre-sowing lead times, and tree-based model classes. It should not be extrapolated to mean that weather has no biological influence on plant growth.

---

### 12.3 Engineering & Infrastructure Limitations

1. **Host-Volume File Persistence**:
   Prediction audit trails and operational telemetry logs operate via local host-volume append-oriented CSV/JSONL files rather than a distributed cloud relational database management system (e.g. PostgreSQL/Spanner).
2. **Simulated Live Synchronization**:
   While production containerization and reverse proxy routing are fully operational, the system does not maintain live automated polling connections to real-time government telemetry feeds (e.g. live IMD radar or UPAg API).
