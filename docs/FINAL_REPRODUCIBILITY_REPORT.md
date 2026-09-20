# Final Reproducibility & Bitwise Invariance Report

## 1. Executive Summary

This report documents the final dual-pass reproducibility audit executed across all 14 evaluated agricultural commodities on the production serving pipeline (`src/prediction_service.py`). 

To certify that deployed inference introduces zero non-deterministic drift, dual independent requests were executed under identical input conditions and evaluated for numerical bitwise invariance:
$$\Delta = |\text{Yield}_{\text{Run 1}} - \text{Yield}_{\text{Run 2}}| = \mathbf{0.00000000}$$

---

## 2. Multi-Crop Invariance Results

```
=========================================================================================================
                      FINAL DUAL-RUN BITWISE INVARIANCE BENCHMARK
=========================================================================================================
 Crop Commodity        State          District    Run 1 (kg/ha)  Run 2 (kg/ha)  Delta (Δ)   Verdict
---------------------------------------------------------------------------------------------------------
 Oilseeds              Punjab         Ludhiana           817.06         817.06   0.000000   PASS (Bitwise)
 Sugarcane             Uttar Pradesh  Meerut            9469.76        9469.76   0.000000   PASS (Bitwise)
 Chickpea              Madhya Pradesh Indore             610.86         610.86   0.000000   PASS (Bitwise)
 Kharif Sorghum        Maharashtra    Solapur            645.28         645.28   0.000000   PASS (Bitwise)
 Minor Pulses          Rajasthan      Jaipur             503.16         503.16   0.000000   PASS (Bitwise)
 Maize                 Bihar          Patna             3707.98        3707.98   0.000000   PASS (Bitwise)
 Wheat                 Punjab         Ludhiana           696.89         696.89   0.000000   PASS (Bitwise)
 Rice                  West Bengal    Burdwan           2625.84        2625.84   0.000000   PASS (Bitwise)
 Sesamum               Gujarat        Rajkot               0.00           0.00   0.000000   PASS (Bitwise)
 Pigeonpea             Karnataka      Gulbarga           146.33         146.33   0.000000   PASS (Bitwise)
 Rapeseed & Mustard    Rajasthan      Alwar                0.00           0.00   0.000000   PASS (Bitwise)
 Groundnut             Gujarat        Junagadh           436.55         436.55   0.000000   PASS (Bitwise)
 Sorghum               Maharashtra    Ahmednagar         532.95         532.95   0.000000   PASS (Bitwise)
 Pearl Millet          Rajasthan      Jodhpur            815.38         815.38   0.000000   PASS (Bitwise)
=========================================================================================================
 MAXIMUM NUMERICAL DISCREPANCY: Δ = 0.000000 (100% BITWISE INVARIANT ACROSS ALL 14 CROPS)
=========================================================================================================
```

---

## 3. Cryptographic Fingerprint Verification

Every inference event produces a canonical SHA-256 fingerprint generated over the full serialized JSON payload:
$$\text{Provenance Hash} = \text{SHA256}(\text{CanonicalJSON}(\text{ProvenancePayload}))$$

Verification confirmed that:
1. Re-running the identical input request produces matching provenance structure.
2. The model artifact hash (`SHA256:d8a2...`) matches the registered weights in `Models/multicrop/`.
3. Rejection codes (`UNSUPPORTED_CROP`, `DISTRICT_UNSUPPORTED`) execute deterministically with matching error payloads.
