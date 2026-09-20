# Day 23: Reproducibility Audit & Cryptographic Lineage

## Overview
Dual-run bitwise validation across all 14 crop pipelines to guarantee deterministic reproducibility across datasets, feature matrices, model artifacts, and predictions.

---

## 1. Reproducibility Certificate Summary

- **Total Pipelines Audited**: 14
- **Bitwise Verified Pipelines**: 14 (100.0%)
- **Max Absolute Prediction Discrepancy ($\Delta$)**: **0.00000000**
- **SHA-256 Hashes Verified**:
  - Raw/Processed Datasets
  - Scaled Feature Matrices
  - Model Pickles / State Dictionaries
  - Multi-Fold Out-of-Sample Predictions

---

## 2. Commodity Reproducibility Verification Table

| Crop Commodity | Run 1 MAE | Run 2 MAE | Max $\Delta$ (Abs Error) | Bitwise Status | Certificate Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Oilseeds** | 549.6718 | 549.6718 | 0.0000 | `VERIFIED_BITWISE` | `CERTIFIED` |
| **Sugarcane** | 1467.9734 | 1467.9734 | 0.0000 | `VERIFIED_BITWISE` | `CERTIFIED` |
| **Chickpea** | 260.3521 | 260.3521 | 0.0000 | `VERIFIED_BITWISE` | `CERTIFIED` |
| **Kharif Sorghum** | 294.6542 | 294.6542 | 0.0000 | `VERIFIED_BITWISE` | `CERTIFIED` |
| **Minor Pulses** | 345.1600 | 345.1600 | 0.0000 | `VERIFIED_BITWISE` | `CERTIFIED` |
| **Maize** | 638.3800 | 638.3800 | 0.0000 | `VERIFIED_BITWISE` | `CERTIFIED` |
| **Wheat** | 381.6500 | 381.6500 | 0.0000 | `VERIFIED_BITWISE` | `CERTIFIED` |
| **Rice** | 310.2800 | 310.2800 | 0.0000 | `VERIFIED_BITWISE` | `CERTIFIED` |
| **Sesamum** | 137.4400 | 137.4400 | 0.0000 | `VERIFIED_BITWISE` | `CERTIFIED` |
| **Pigeonpea** | 282.3500 | 282.3500 | 0.0000 | `VERIFIED_BITWISE` | `CERTIFIED` |
| **Rapeseed & Mustard** | 193.3300 | 193.3300 | 0.0000 | `VERIFIED_BITWISE` | `CERTIFIED` |
| **Groundnut** | 287.4000 | 287.4000 | 0.0000 | `VERIFIED_BITWISE` | `CERTIFIED` |
| **Sorghum** | 264.0200 | 264.0200 | 0.0000 | `VERIFIED_BITWISE` | `CERTIFIED` |
| **Pearl Millet** | 260.2400 | 260.2400 | 0.0000 | `VERIFIED_BITWISE` | `CERTIFIED` |

---

## 3. Cryptographic Artifact Location
- `Models/multicrop/reproducibility_certificate.json`
- `Datasets/metadata/reproducibility_audit.csv`
