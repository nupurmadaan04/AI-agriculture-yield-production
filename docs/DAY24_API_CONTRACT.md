# Day 24: REST API Contract & Specification

## 1. Overview

The Day 24 Forecast Serving and Governance subsystem exposes 7 REST API endpoints under `/api/forecast/*`.

---

## 2. Endpoints

### 1. `GET /api/forecast/strategies`
Returns the compiled forecast strategy registry for all 14 commodities.

### 2. `GET /api/forecast/certification`
Returns certification status counts and high-level governance policy directives.

### 3. `GET /api/forecast/coverage`
Returns supported crops, states, districts, and observation counts across 9,019 geographic mappings.

### 4. `POST /api/forecast/predict`
Executes governed forecasting inference.
**Payload:**
```json
{
  "crop": "Oilseeds",
  "state": "Punjab",
  "district": "Ludhiana",
  "forecast_year": 2018,
  "yield_lag_1": null,
  "yield_rolling_3yr_mean": null,
  "area_lag_1": null
}
```

### 5. `GET /api/forecast/provenance/{request_id}`
Returns the full cryptographic provenance payload for a specific request ID.

### 6. `GET /api/forecast/audit?limit=50`
Returns recent immutable prediction audit log entries.

### 7. `GET /api/forecast/health`
Returns operational health status, governance guard state, and registered strategy count.
