# Geospatial Intelligence & Regional Clustering Methodology

## 1. Overview
The **Geospatial Intelligence Engine** models spatial heterogeneity and geographic patterns across India's 311 agricultural districts without relying on synthetic geospatial interpolations.

---

## 2. Spatial Clustering Framework

### 2.1 Coordinate Normalization & K-Means Clustering
Districts are clustered based on normalized geographic centroid coordinates $(\text{Latitude}_i, \text{Longitude}_i)$:

$$\mathbf{x}_{\text{geo}} = \left( \frac{\text{Lat} - \mu_{\text{lat}}}{\sigma_{\text{lat}}}, \frac{\text{Lon} - \mu_{\text{lon}}}{\sigma_{\text{lon}}} \right)$$

- **Cluster Count:** $K=5$ macro-agricultural zones (Indo-Gangetic Plain, Southern Deccan, Coastal Delta, Western Arid, Central Plateau).
- **Cluster Properties:** Captures contiguous agro-ecological zones with similar baseline climatic regimes.

---

## 3. Within-State Spatial Z-Scores
To detect localized agricultural deviations relative to regional peers:

$$z_{\text{state}}(i, t) = \frac{y_{i, t} - \mu_{\text{state}}(t)}{\sigma_{\text{state}}(t)}$$

- **Spatial Outlier Threshold:** $|z_{\text{state}}| \ge 2.0\sigma$
- **Regional Similarity Metric:** Euclidean distance in standardized feature space weighted by geographic adjacency.
