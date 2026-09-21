# Forecast Serving Runtime Architecture

## 1. Request Handling & Latency Profiling

The forecast serving engine is implemented in FastAPI (`backend/main.py` and `src/prediction_service.py`):
- **Throughput & Latency**: Sub-50ms $P95$ latency for point forecasts and empirical uncertainty generation.
- **Request Validation**: Pydantic v2 schemas enforce strict typing, boundary validation, and coordinate checks on all incoming requests.

---

## 2. Dynamic Router & Guard Execution Flow

When a forecast request arrives for crop $C$, district $D$, and year $Y$:
1. **Strategy Guard Lookup**: Consults `Models/multicrop/forecast_strategy_registry.json`.
2. **Strategy Assignment**:
   - If $C = \text{Oilseeds}$: Executes trained `RandomForestRegressor`. If district lags are unavailable, falls back to Historical District Mean.
   - If $C = \text{Sugarcane}$: Executes `GradientBoostingRegressor`. If predicted yield deviates $> 3\sigma$ from district mean, activates 3-$\sigma$ variance clipping fallback.
   - If $C \in \{\text{Rice, Wheat, Chickpea, } \dots\}$: Executes certified `HistoricalDistrictMean` persistence.
3. **Provenance Signature**: Hashes request ID, model version, and inputs into SHA-256 digital fingerprint.
