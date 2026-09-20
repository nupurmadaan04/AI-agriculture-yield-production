# Day 22: Temporal Alignment & Pre-Season Availability Contract

## 1. The Pre-Season Forecast Origin Mandate
In agricultural yield forecasting, temporal availability is the strict non-negotiable rule. An operational pre-season model makes forecasts **prior to sowing** (June 1 for Kharif commodities).

$$\text{Forecast Origin: } t_{\text{forecast}} \le \text{May 31, year } t$$

Any meteorological, biophysical, or vegetative observation occurring after this origin is classified as **UNSAFE** and strictly forbidden.

---

## 2. Temporal Contract Registry

| Feature Name | Observation Period | Availability Date | Forecast Horizon | Lag | Timing Status | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `preseason_rainfall_total` | Jan 1 – May 31 | May 31 | Pre-Season | 0 | `SAFE` | Pre-monsoon shower accumulation prior to field preparation |
| `preseason_rainfall_anomaly`| Jan 1 – May 31 | May 31 | Pre-Season | 0 | `SAFE` | Departure from 30-year historical baseline |
| `preseason_temp_mean` | Mar 1 – May 31 | May 31 | Pre-Season | 0 | `SAFE` | Mean pre-sowing thermal regime |
| `preseason_temp_max` | Mar 1 – May 31 | May 31 | Pre-Season | 0 | `SAFE` | Daytime maximum heat wave exposure |
| `preseason_soil_moisture` | May 1 – May 25 | May 25 | Pre-Season | 0 | `SAFE` | Near-real-time satellite root-zone saturation |
| `preseason_aridity_index` | Jan 1 – May 31 | May 31 | Pre-Season | 0 | `SAFE` | Standardized precipitation-evapotranspiration index |
| `rainfall_lag1_total` | Jan 1 – Dec 31 (t-1) | Dec 31 (t-1) | Pre-Season | 1 | `SAFE` | Trailing year hydrological reservoir carryover |
| `irrigation_ratio_lag1` | Year t-1 | Dec 31 (t-1) | Pre-Season | 1 | `SAFE` | Observed infrastructure capacity from prior agricultural season |
| `monsoon_rainfall_june_sept`| Jun 1 – Sep 30 (t) | Sep 30 (t) | Concurrent | 0 | `UNSAFE` | **Lookahead Leakage**: Concurrent monsoon rain occurs during crop growth |
| `harvest_ndvi_max` | Aug 1 – Oct 31 (t) | Oct 31 (t) | Post-Sowing | 0 | `UNSAFE` | **Lookahead Leakage**: Peak vegetative vigor occurs months post-forecast |
