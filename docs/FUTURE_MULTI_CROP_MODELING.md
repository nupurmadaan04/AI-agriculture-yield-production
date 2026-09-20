# Future Multi-Crop Modeling Architecture

## Overview
This roadmap outlines the technical and statistical requirements for expanding predictive machine learning capabilities from the active Rice baseline to a complete suite of multi-crop agricultural forecasting models.

---

## 1. Architectural Strategy Comparison

| Approach | Architecture | Advantages | Disadvantages | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **A. Global Unified Model** | Single regressor with categorical crop embeddings | Single artifact to deploy | Conflates different agronomic dynamics; high cross-crop variance | Not Recommended |
| **B. Hierarchical Group Models** | Separate models for Cereals, Pulses, Oilseeds | Group-level weight sharing | Misses crop-specific physiological thresholds | Secondary Option |
| **C. Crop-Specific Models** | Dedicated Random Forest / LightGBM per crop | Uncompromised feature importance; distinct biological ranges | Requires independent tuning & validation artifacts | **Recommended Strategy** |

---

## 2. Ingestion & Feature Engineering Requirements

1. **Target Feature Sets**:
   - Lagged crop yield ($t-1, t-2, t-3$)
   - Cropping intensity and crop land share
   - Soil fertility indices (N, P, K)
   - Agro-climatic zone embeddings and spatial neighbor lags
2. **Evaluation Protocol**:
   - Out-of-time chronological train/test split (2010–2015 train, 2016–2017 test)
   - Spatial Group K-Fold cross-validation by state
   - 10th/90th percentile prediction interval calibration
