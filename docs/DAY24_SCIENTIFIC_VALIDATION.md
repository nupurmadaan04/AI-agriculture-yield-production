# Day 24: Scientific Validation & Reproducibility Audit

## 1. Dual-Run Bitwise Invariance Benchmark

To certify that deployed forecast serving code introduces zero non-deterministic drift, dual independent inference passes were executed across all 14 commodities:

| Crop Commodity | Pass 1 Yield (kg/ha) | Pass 2 Yield (kg/ha) | Absolute Diff ($\Delta$) | Deterministic Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Oilseeds** | 817.06 | 817.06 | 0.000000 | `VERIFIED` |
| **Sugarcane** | 9,469.76 | 9,469.76 | 0.000000 | `VERIFIED` |
| **Chickpea** | 610.86 | 610.86 | 0.000000 | `VERIFIED` |
| **Kharif Sorghum** | 645.28 | 645.28 | 0.000000 | `VERIFIED` |
| **Minor Pulses** | 503.16 | 503.16 | 0.000000 | `VERIFIED` |
| **Maize** | 3,707.98 | 3,707.98 | 0.000000 | `VERIFIED` |
| **Wheat** | 696.89 | 696.89 | 0.000000 | `VERIFIED` |
| **Rice** | 2,625.84 | 2,625.84 | 0.000000 | `VERIFIED` |
| **Sesamum** | 0.00 | 0.00 | 0.000000 | `VERIFIED` |
| **Pigeonpea** | 146.33 | 146.33 | 0.000000 | `VERIFIED` |
| **Rapeseed & Mustard** | 0.00 | 0.00 | 0.000000 | `VERIFIED` |
| **Groundnut** | 436.55 | 436.55 | 0.000000 | `VERIFIED` |
| **Sorghum** | 532.95 | 532.95 | 0.000000 | `VERIFIED` |
| **Pearl Millet** | 815.38 | 815.38 | 0.000000 | `VERIFIED` |

---

## 2. Governance Rejection Audit

1. **Unsupported Commodity Rejection**: Requesting crop `Potato` in Punjab / Ludhiana triggers `UNSUPPORTED_CROP` rejection with `status: REJECTED`.
2. **Unsupported District Rejection**: Requesting `Oilseeds` in `NonExistentDistrict123` triggers `DISTRICT_UNSUPPORTED` rejection with `status: REJECTED`.
3. **Immutability of Audit Log**: Every rejection event is persisted to `Datasets/metadata/prediction_audit_log.csv`.
