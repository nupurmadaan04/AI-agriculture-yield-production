# Authoritative Agricultural Data Sources

This document specifies the authoritative public agricultural data sources integrated or registered in the **AI Agriculture Intelligence Platform**.

---

## 1. Primary Ingested Source: ICRISAT District Level Database (DLD)

- **Source ID**: `ICRISAT_DLD_1966_2017`
- **Source Name**: ICRISAT District Level Database (DLD)
- **Provider**: International Crops Research Institute for the Semi-Arid Tropics (ICRISAT)
- **Source URL**: [http://data.icrisat.org/dld/](http://data.icrisat.org/dld/)
- **Access Method**: Direct Local Archive Ingestion
- **Download Date**: 2026-09-02
- **License / Terms**: Open Access for Academic, Research, and Public Decision Support
- **Geographic Level**: District Level (311 agricultural districts across 20 Indian states)
- **Temporal Coverage**: 1966–2017 (51 continuous agricultural seasons)
- **Crop Coverage**: 29 crop categories including:
  - *Cereals*: Rice, Wheat, Kharif Sorghum, Rabi Sorghum, Total Sorghum, Pearl Millet, Maize, Finger Millet, Barley
  - *Pulses*: Chickpea, Pigeonpea, Minor Pulses
  - *Oilseeds*: Groundnut, Sesamum, Rapeseed & Mustard, Safflower, Castor, Linseed, Sunflower, Soyabean, Total Oilseeds
  - *Commercial*: Sugarcane, Cotton
  - *Horticulture & Forage*: Fruits, Vegetables, Fruits & Vegetables, Potatoes, Onion, Fodder
- **Variables**: Area, Production, Yield
- **Original Units**:
  - Area: 1,000 hectares
  - Production: 1,000 metric tons
  - Yield: kg/hectare
- **Standardized Units**:
  - Area: Hectares (`ha`)
  - Production: Metric tonnes (`tonnes`)
  - Yield: Kilograms per hectare (`kg/ha`)
- **Known Limitations**: Harmonized to 1966 district boundaries to ensure longitudinal panel integrity across district bifurcations over the 51-year horizon.

---

## 2. Secondary Registered Source: Government of India Open Government Data (OGD)

- **Source ID**: `GOVT_INDIA_OGD_DES`
- **Source Name**: Open Government Data (OGD) - Directorate of Economics and Statistics (DES)
- **Provider**: Ministry of Agriculture and Farmers Welfare, Government of India
- **Source URL**: [https://data.gov.in/](https://data.gov.in/)
- **Access Method**: Ingestion Specification Registered
- **License / Terms**: National Data Sharing and Accessibility Policy (NDSAP) - Government Open Data License India (GODL)
- **Geographic Level**: State and District Level
- **Temporal Coverage**: 1997–2020
- **Crop Coverage**: Major national crops (Rice, Wheat, Maize, Cotton, Sugarcane, Gram, Groundnut, Arhar/Tur)
- **Variables**: Area (ha), Production (tonnes), Yield (kg/ha)
- **Known Limitations**: Variations in seasonal reporting across states; requires district boundary harmonization before temporal concatenation.

---

## 3. Reference Source: FAOSTAT Crop and Livestock Products

- **Source ID**: `FAOSTAT_CROP_PRODUCTION`
- **Source Name**: FAOSTAT Crop and Livestock Products
- **Provider**: Food and Agriculture Organization of the United Nations (FAO)
- **Source URL**: [https://www.fao.org/faostat/en/#data/QCL](https://www.fao.org/faostat/en/#data/QCL)
- **Access Method**: API Client Specification
- **License / Terms**: CC BY-NC-SA 3.0 IGO
- **Geographic Level**: Country Level (National totals)
- **Temporal Coverage**: 1961–2022
- **Crop Coverage**: Global agricultural commodities
- **Known Limitations**: National totals only; useful for macro benchmarking but does not provide district-level granularity.
