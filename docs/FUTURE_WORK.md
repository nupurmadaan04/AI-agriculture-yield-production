# Future Work & Research Roadmap

> **Important Notice**:
> The following items represent future research directions and enhancements. They are **not** currently implemented in the production platform.

---

## 1. Data Ingestion & Environmental Telemetry
- **Remote Sensing Integration**: Ingestion of multispectral satellite imagery (Sentinel-2 / Landsat-8) to compute in-season Normalized Difference Vegetation Index (NDVI) and Enhanced Vegetation Index (EVI).
- **High-Resolution Daily Meteorology**: Integration of daily gridded reanalysis datasets (ERA5-Land / IMD 0.25° grid) to model extreme intra-seasonal weather shocks (e.g. heatwaves during anthesis).
- **Soil Grids & Topography**: Integration of high-resolution soil organic carbon, pH, and digital elevation models (DEM) for enhanced local terrain modeling.

---

## 2. Advanced Methodologies & Uncertainty
- **Conformal Prediction & Bayesian Neural Networks**: Formal distribution-free prediction intervals with mathematically guaranteed finite-sample coverage rates.
- **Causal Inference & Double Machine Learning (DML)**: Estimating true heterogeneous treatment effects of specific agricultural interventions using orthogonalized DML estimators.
- **Multi-Crop Generalization**: Extending the panel architecture to wheat, maize, pulses, and oilseeds across South and Southeast Asia.

---

## 3. Production Infrastructure & Real-Time Capabilities
- **Sub-District & Field-Level Downscaling**: Developing spatial disaggregation models for block- and village-level agricultural extension advisory.
- **Edge Deployment & Offline Mobile Apps**: Distilling models via ONNX runtime for offline field worker operations in low-connectivity rural environments.
- **Streaming Telemetry & Automated Retraining**: Event-driven ingestion pipelines updating model drift statistics upon annual harvest data releases.
