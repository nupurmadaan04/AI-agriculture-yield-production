# 8. Model Explainability & Feature Attribution

### 8.1 Methodological Architecture

To deliver transparent, inspectable insights to agronomic stakeholders, the framework implements a dual-tier explainability architecture:

1. **Marginal Reference Perturbation Attribution**:
   Implements a model-agnostic local attribution mechanism in `src/explainability_engine.py`. For a given district feature vector $\mathbf{x} = (x_1, \dots, x_M)$, the marginal attribution $\phi_j(\mathbf{x})$ of feature $j$ is estimated by evaluating the model output shift relative to an empirical district reference baseline $\mathbf{x}^{(0)}$:
   $$\phi_j(\mathbf{x}) = f(x_1, \dots, x_j, \dots, x_M) - f(x_1, \dots, x_j^{(0)}, \dots, x_M)$$
   where $\mathbf{x}^{(0)}$ represents the district's median historical feature profile.
2. **Tree-Based Attribution**:
   For Random Forest and Gradient Boosting models, global feature importance is extracted via Mean Decrease in Impurity (MDI), while local attributions in the Decision Workspace are decomposed via Tree SHAP (Lundberg & Lee, 2017).

```
+----------------------------------------------------------------------------------------------------+
|                                    TABLE 3: GLOBAL FEATURE IMPORTANCE                              |
+------------------------------------+-----------------------+-------------------+-------------------+
| Feature Name                       | Oilseeds (RF)         | Sugarcane (GBDT)  | Primary Driver    |
+------------------------------------+-----------------------+-------------------+-------------------+
| **Yield Lag-1 ($y_{t-1}$)**        | 0.421                 | 0.385             | Autoregressive    |
| **3-Year Rolling Mean Yield**      | 0.284                 | 0.312             | Medium-Term Base  |
| **Cultivated Area Share**          | 0.165                 | 0.148             | Spatial Intensity |
| **District Cropping Intensity**    | 0.082                 | 0.091             | Technological     |
| **Pre-Season Weather Anomaly**     | 0.048                 | 0.064             | Pre-Sowing Hydro  |
+------------------------------------+-----------------------+-------------------+-------------------+
```

### 8.2 Baseline Transparency Principle

A core principle of this framework is that **statistical baselines do not fabricate explanations**. When a user queries a forecast for Rice, Wheat, or any other `BASELINE_PRODUCTION` commodity, the explainability engine does not execute synthetic tree perturbation routines. Instead, it explicitly and transparently reports:

> *"Certified Strategy is Historical District Mean Persistence. Forecast is derived as the expanding historical mean ($\bar{y}_{\text{dist}}$). Individual ML feature attributions are not applicable."*

### 8.3 Non-Causal Interpretation Boundary

Every explainability artifact, API response, and dashboard interface enforces a strict scientific boundary:
- **Model Attribution $\neq$ Biological Causality**: Attributions reflect the mathematical sensitivity of the trained estimator within the feature manifold.
- An attribution showing that a 10% increase in cultivated area corresponds to an expected +50 kg/ha yield output describes empirical correlation in the training distribution. It must **not** be interpreted as a causal promise that planting more area will biologically increase crop productivity.
