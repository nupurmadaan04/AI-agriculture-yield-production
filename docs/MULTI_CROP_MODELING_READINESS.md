# Multi-Crop Modeling Readiness & Classification Specification

## 1. Classification Summary

The Multi-Crop Modeling Readiness Framework evaluated all **29 agricultural crop commodities** in the unified panel (`Datasets/processed/agricultural_panel.csv`, 71,601 records) across 7 quantitative criteria: total volume, temporal continuity, geographic breath, target completeness, zero-inflation, validation split feasibility, and baseline performance.

```
Total Crops Evaluated: 29
├── 🟢 MODEL_READY: 14 Crops (48.3%)
├── 🟡 ANALYTICS_READY: 9 Crops (31.0%)
└── 🔴 INSUFFICIENT_DATA: 6 Crops (20.7%)
```

---

## 2. Complete Crop Readiness Registry

| Crop Name | Readiness Status | Overall Score | Total Records | Active Districts | Median Continuity | Zero Yield % | Baseline MAE (kg/ha) | Recommended Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Minor Pulses** | `MODEL_READY` | **91.1** | 2,469 | 277 | 1.00 | 8.87% | 322.30 | Eligible for prospective ML model development |
| **Maize** | `MODEL_READY` | **90.4** | 2,469 | 290 | 1.00 | 9.07% | 763.91 | Eligible for prospective ML model development |
| **Rice** | `MODEL_READY` | **89.5** | 2,469 | 299 | 1.00 | 4.86% | 367.49 | Active validated model registered ($R^2=0.7866$) |
| **Sesamum** | `MODEL_READY` | **88.2** | 2,469 | 260 | 1.00 | 14.82% | 117.69 | Eligible for prospective ML model development |
| **Chickpea** | `MODEL_READY` | **86.4** | 2,469 | 237 | 1.00 | 15.63% | 274.42 | Eligible for prospective ML model development |
| **Pigeonpea** | `MODEL_READY` | **86.2** | 2,469 | 231 | 1.00 | 19.36% | 299.15 | Eligible for prospective ML model development |
| **Wheat** | `MODEL_READY` | **86.0** | 2,469 | 286 | 1.00 | 7.94% | 396.55 | Eligible for prospective ML model development |
| **Oilseeds** | `MODEL_READY` | **84.8** | 2,469 | 299 | 1.00 | 2.51% | 746.72 | Eligible for prospective ML model development |
| **Sugarcane** | `MODEL_READY` | **81.7** | 2,469 | 224 | 1.00 | 28.59% | 1126.31 | Eligible for prospective ML model development |
| **Rapeseed and Mustard** | `MODEL_READY` | **80.3** | 2,469 | 240 | 1.00 | 23.33% | 224.95 | Eligible for prospective ML model development |
| **Groundnut** | `MODEL_READY` | **79.9** | 2,469 | 239 | 1.00 | 27.62% | 302.18 | Eligible for prospective ML model development |
| **Sorghum** | `MODEL_READY` | **79.4** | 2,469 | 223 | 1.00 | 31.92% | 305.79 | Eligible for prospective ML model development |
| **Kharif Sorghum** | `MODEL_READY` | **76.9** | 2,469 | 175 | 1.00 | 44.92% | 331.18 | Eligible for prospective ML model development |
| **Pearl Millet** | `MODEL_READY` | **76.8** | 2,469 | 179 | 1.00 | 43.14% | 329.83 | Eligible for prospective ML model development |
| **Sunflower** | `ANALYTICS_READY` | **72.1** | 2,469 | 120 | 1.00 | 66.83% | 264.92 | Retain for historical descriptive analytics only |
| **Cotton** | `ANALYTICS_READY` | **71.7** | 2,469 | 143 | 1.00 | 61.40% | 262.14 | Retain for historical descriptive analytics only |
| **Linseed** | `ANALYTICS_READY` | **70.5** | 2,469 | 145 | 1.00 | 54.39% | 183.15 | Retain for historical descriptive analytics only |
| **Soyabean** | `ANALYTICS_READY` | **70.4** | 2,469 | 158 | 1.00 | 54.72% | 287.57 | Retain for historical descriptive analytics only |
| **Barley** | `ANALYTICS_READY` | **69.8** | 2,469 | 123 | 1.00 | 64.64% | 521.96 | Retain for historical descriptive analytics only |
| **Castor** | `ANALYTICS_READY` | **67.4** | 2,469 | 97 | 1.00 | 72.82% | 215.43 | Retain for historical descriptive analytics only |
| **Finger Millet** | `ANALYTICS_READY` | **66.8** | 2,469 | 96 | 1.00 | 73.63% | 315.39 | Retain for historical descriptive analytics only |
| **Rabi Sorghum** | `ANALYTICS_READY` | **66.6** | 2,469 | 85 | 1.00 | 74.85% | 379.55 | Retain for historical descriptive analytics only |
| **Safflower** | `ANALYTICS_READY` | **65.6** | 2,469 | 69 | 1.00 | 78.49% | 221.92 | Retain for historical descriptive analytics only |
| **Fruits and Vegetables** | `INSUFFICIENT_DATA` | **46.7** | 2,469 | 0 | 1.00 | 100.0% | N/A | Exclude from modeling (aggregate survey metric) |
| **Fodder** | `INSUFFICIENT_DATA` | **46.7** | 2,469 | 0 | 1.00 | 100.0% | N/A | Exclude from modeling (no yield recorded) |
| **Fruits** | `INSUFFICIENT_DATA` | **46.7** | 2,469 | 0 | 1.00 | 100.0% | N/A | Exclude from modeling (no yield recorded) |
| **Onion** | `INSUFFICIENT_DATA` | **46.7** | 2,469 | 0 | 1.00 | 100.0% | N/A | Exclude from modeling (no yield recorded) |
| **Potatoes** | `INSUFFICIENT_DATA` | **46.7** | 2,469 | 0 | 1.00 | 100.0% | N/A | Exclude from modeling (no yield recorded) |
| **Vegetables** | `INSUFFICIENT_DATA` | **46.7** | 2,469 | 0 | 1.00 | 100.0% | N/A | Exclude from modeling (no yield recorded) |
