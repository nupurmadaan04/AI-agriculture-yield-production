# Explainable Agricultural AI & Decision Traceability Methodology

## 1. Purpose & Overview
The **Explainable AI (XAI) & Decision Traceability Layer** provides transparent, verifiable, and mathematically defensible model interpretability for all registered machine learning models in the Agricultural Decision Intelligence Platform.

It provides answers to 10 core governance questions:
1. **Why did the model produce this prediction?** &rarr; Local marginal reference attribution.
2. **Which features contributed most?** &rarr; Ranked directional feature contributions.
3. **Which features pushed the prediction higher or lower?** &rarr; Positive vs. negative attribution partitioning.
4. **How important is each feature globally?** &rarr; Dual model-native Gini importance vs. out-of-sample permutation importance.
5. **Why did this district/state receive this warning?** &rarr; Multi-signal monitoring evidence deconstruction.
6. **What changed between baseline and scenario?** &rarr; Input parameter differential analysis.
7. **Which model inputs are responsible for the difference?** &rarr; Scenario response sensitivity decomposition.
8. **How reliable is this explanation?** &rarr; Grounded in out-of-time $R^2=0.7866$, $\text{MAE}=353.01$, and PSI drift verification.
9. **Which model version generated the explanation?** &rarr; Explicitly tracked model version (`2.1.0`) and dataset provenance (`ICRISAT 1966–2017`).
10. **Can the entire explanation be reproduced later?** &rarr; Immutable SHA-256 decision certificates (`EXP-xxxx`).

---

## 2. Supported Models & Estimators

| Model ID | Pipeline Artifact | Task | Estimator Architecture | XAI Support |
| :--- | :--- | :--- | :--- | :--- |
| `exogenous_rf_forecaster` | `Models/forecasting_pipeline.pkl` | Yield Forecasting & Scenarios | `RandomForestRegressor (n=150, d=14)` | Native, Permutation, Marginal Attribution, Sensitivity |
| `pre_season_exogenous_pipeline` | `Models/pre_season_exogenous_pipeline.pkl` | Pre-Season Estimation | `GradientBoostingRegressor / RF` | Native, Permutation, Marginal Attribution |
| `isolation_forest_anomaly_detector` | `Models/agricultural_anomaly_pipeline.pkl` | Shock & Departure Flags | `IsolationForest (c=0.04)` | Feature-level Path Outlier Attribution |
| `kmeans_spatial_clusterer` | `Models/spatial_cluster_pipeline.pkl` | Agro-Climatic Grouping | `KMeans (k=4, StandardScaler)` | Cluster Centroid Dimension Distances |

---

## 3. Global Interpretability: Native vs. Permutation Importance

### 3.1 Model-Native Feature Importance
For tree ensembles, feature importance is calculated as the mean decrease in impurity (MDI / Gini):
$$I_{\text{native}}(f) = \frac{\sum_{t \in T} \Delta \text{Impurity}(t, f)}{\sum_{f'} \sum_{t \in T} \Delta \text{Impurity}(t, f')}$$

### 3.2 Out-of-Sample Permutation Importance
Permutation importance evaluates the loss increase on the holdout evaluation partition (Year > 2015) when feature $f$ is randomly permuted:
$$I_{\text{perm}}(f) = \frac{1}{K} \sum_{k=1}^K \left[ \text{MSE}(\mathbf{X}_{\text{perm}(f)}^{(k)}, \mathbf{y}) - \text{MSE}(\mathbf{X}_{\text{test}}, \mathbf{y}) \right]$$

### 3.3 Consensus vs. Divergence
Rank divergence occurs when a feature has high split frequency (high Gini) due to correlation with another feature, but low independent predictive loss penalty (low permutation). Divergences are highlighted with rank divergence badges rather than concealed.

---

## 4. Local Prediction Attribution: Marginal Reference Perturbation

To avoid synthetic or ungrounded additive decompositions, local attribution is computed via **Marginal Reference Perturbation** against empirical dataset median baselines $\mathbf{x}_{\text{ref}}$:
$$\Delta \hat{y}_j = \hat{y}(\mathbf{x}_{\text{ref}}^{j \leftarrow x_j}) - \hat{y}(\mathbf{x}_{\text{ref}})$$
- **Positive Driver**: $\Delta \hat{y}_j > 0$
- **Negative Driver**: $\Delta \hat{y}_j < 0$
- **Relative Influence**:
$$\text{Influence}_j = \frac{|\Delta \hat{y}_j|}{\sum_{k} |\Delta \hat{y}_k|} \times 100\%$$

---

## 5. Controlled Feature Sensitivity Analysis
Continuous $[-10\%, -5\%, 0\%, +5\%, +10\%]$ sweeps are applied to key features while strictly enforcing non-negative agricultural domain boundaries:
$$\hat{y}(\mathbf{x}_{j, \delta}) = f(\mathbf{x}_1, \dots, \max(0, x_j \cdot (1 + \delta)), \dots, \mathbf{x}_p)$$

---

## 6. Scientific Integrity Directives & Non-Causal Boundaries
1. **Explanations describe models, NOT reality**: An explanation explains why a statistical model produced an output within its trained feature space; it does not claim biological causation.
2. **Never assert causal impact**: Language such as *"increasing area causes yield to increase"* is forbidden. The platform strictly outputs *"model contribution"*, *"model sensitivity"*, and *"associated with the prediction"*.
3. **No fabricated explanations**: If an explanation cannot be computed from model artifacts, the system explicitly returns `"Explanation unavailable for this model configuration"`.
4. **Immutable decision certificates**: Every explanation is bound to an audit ID (`EXP-xxxx`), model version, and timestamp.
