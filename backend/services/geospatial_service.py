"""
Geospatial Intelligence Service.

Serves GIS choropleth datasets, spatial clusters, state/district spatial profiles,
and multi-dimensional cosine similarity searches across India's agricultural panel.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from backend.utils.data_loader import data_loader
from backend.services.ml_service import ml_service, STATE_TO_CODE
from backend.services.risk_service import risk_service
from backend.services.early_warning_service import early_warning_service
from backend.services.trend_service import trend_service
from backend.services.forecast_service import forecast_service
from backend.services.spatial_outlier_service import spatial_outlier_service
from src.build_geo_dataset import STATE_GEO_REGISTRY

class GeospatialService:
    _instance: Optional['GeospatialService'] = None
    _manifest: Optional[Dict[str, Any]] = None
    _cluster_metadata: Optional[Dict[str, Any]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GeospatialService, cls).__new__(cls)
        return cls._instance

    def load_metadata(self) -> None:
        base_dir = Path(__file__).resolve().parent.parent.parent
        manifest_path = base_dir / 'Models' / 'geographic_feature_manifest.json'
        cluster_path = base_dir / 'Models' / 'spatial_cluster_metadata.json'

        if manifest_path.exists():
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    self._manifest = json.load(f)
            except Exception as e:
                print(f"[GeospatialService Error] Failed to load manifest: {e}")

        if cluster_path.exists():
            try:
                with open(cluster_path, 'r', encoding='utf-8') as f:
                    self._cluster_metadata = json.load(f)
            except Exception as e:
                print(f"[GeospatialService Error] Failed to load cluster metadata: {e}")

    def get_spatial_overview(self) -> Dict[str, Any]:
        """Provides executive geospatial statistics."""
        df = data_loader.dataframe
        outliers = spatial_outlier_service.get_spatial_outliers()
        states_data = self.get_states_spatial()

        high_risk_states = sum(1 for s in states_data if s['risk_score'] >= 50.0)
        anomalous_states = sum(1 for s in states_data if s['anomaly_count'] > 0)

        return {
            'total_states_monitored': len(states_data),
            'total_districts_monitored': int(df['Dist Name'].nunique()),
            'total_observations': len(df),
            'national_average_yield_kg_ha': round(float(df['RICE YIELD (Kg per ha)'].mean()), 1),
            'national_average_risk_score': round(float(np.mean([s['risk_score'] for s in states_data])), 1),
            'high_risk_states_count': high_risk_states,
            'anomalous_states_count': anomalous_states,
            'spatial_outliers_count': len(outliers),
            'clusters_count': len(self.get_spatial_clusters())
        }

    def get_states_spatial(self) -> List[Dict[str, Any]]:
        """Returns state spatial data with coordinates, risk, yield, and cluster mapping."""
        df = data_loader.dataframe
        results = []

        for s_name, geo in sorted(STATE_GEO_REGISTRY.items()):
            s_data = df[df['State Name'].str.lower() == s_name.lower()]
            if s_data.empty:
                continue

            avg_y = float(round(s_data['RICE YIELD (Kg per ha)'].mean(), 1))
            avg_area = float(round(s_data['RICE AREA (1000 ha)'].mean(), 1))

            # Risk & Warning
            risk_prof = risk_service.get_state_risk_profile(s_name)
            risk_score = float(risk_prof.get('risk_score', 38.5))
            risk_level = str(risk_prof.get('risk_level', 'MODERATE'))

            # Trend
            trend_res = trend_service.analyze_region_trend(state=s_name)

            # Forecast 1-year
            fc_res = forecast_service.forecast_region(state_val=s_name, horizons=[1])
            fc_1yr = fc_res['forecasts'][0]['predicted_yield'] if fc_res.get('forecasts') else avg_y

            # Cluster Assignment (based on yield & risk)
            if avg_y >= 3500.0:
                c_id = 0
            elif avg_y >= 2600.0:
                c_id = 1
            elif risk_score >= 50.0:
                c_id = 3
            else:
                c_id = 2

            results.append({
                'state': s_name,
                'state_code': int(s_data['State Code'].iloc[0]),
                'lat': geo['lat'],
                'lon': geo['lon'],
                'region': geo['region'],
                'agro_zone': geo['zone'],
                'district_count': int(s_data['Dist Name'].nunique()),
                'average_yield_kg_ha': avg_y,
                'average_area_k_ha': avg_area,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'trend_direction': trend_res['direction'],
                'theil_sen_slope': trend_res['theil_sen_slope'],
                'forecast_1yr_kg_ha': fc_1yr,
                'anomaly_count': int(len(s_data[s_data['RICE YIELD (Kg per ha)'] < avg_y * 0.6])),
                'cluster_id': c_id
            })

        return results

    def get_state_spatial_profile(self, state: str) -> Dict[str, Any]:
        """Provides detailed state spatial profile with member districts."""
        state_code, state_name = ml_service.resolve_state(state)
        df = data_loader.dataframe
        s_data = df[df['State Code'] == state_code]

        if s_data.empty:
            s_data = df[df['State Name'].str.lower() == state.lower()]

        geo = STATE_GEO_REGISTRY.get(state_name, {'lat': 20.5937, 'lon': 78.9629, 'region': 'Central', 'zone': 'India'})
        risk_prof = risk_service.get_state_risk_profile(state_name)
        trend_res = trend_service.analyze_region_trend(state=state_name)
        fc_res = forecast_service.forecast_region(state_val=state_name, horizons=[1, 2, 3])
        outliers = spatial_outlier_service.get_spatial_outliers(state=state_name)

        # District breakdowns
        dist_features = spatial_outlier_service.get_district_features()
        s_dist_df = dist_features[dist_features['state_name'].str.lower() == state_name.lower()]
        district_items = s_dist_df.to_dict(orient='records')

        return {
            'state': state_name,
            'state_code': state_code,
            'lat': geo['lat'],
            'lon': geo['lon'],
            'region': geo['region'],
            'agro_zone': geo['zone'],
            'district_count': len(district_items),
            'average_yield_kg_ha': round(float(s_data['RICE YIELD (Kg per ha)'].mean()), 1),
            'risk_score': risk_prof.get('risk_score', 38.5),
            'risk_level': risk_prof.get('risk_level', 'MODERATE'),
            'trend_direction': trend_res['direction'],
            'theil_sen_slope': trend_res['theil_sen_slope'],
            'forecasts': fc_res.get('forecasts', []),
            'spatial_outliers': outliers,
            'districts': district_items
        }

    def get_district_spatial_profile(self, state: str, district: str) -> Dict[str, Any]:
        """Detailed district profile."""
        dist_features = spatial_outlier_service.get_district_features()
        match = dist_features[
            (dist_features['state_name'].str.lower() == state.lower()) &
            (dist_features['district_name'].str.lower() == district.lower())
        ]

        if match.empty:
            match = dist_features[dist_features['district_name'].str.lower() == district.lower()]

        if match.empty:
            return {
                'state': state,
                'district': district,
                'status': 'NOT_FOUND'
            }

        rec = match.iloc[0].to_dict()
        fc_res = forecast_service.forecast_region(state_val=rec['state_name'], district=rec['district_name'], horizons=[1, 2, 3])
        rec['forecasts'] = fc_res.get('forecasts', [])
        return rec

    def get_risk_map(self) -> List[Dict[str, Any]]:
        """Returns risk choropleth state data."""
        return self.get_states_spatial()

    def get_yield_map(self) -> List[Dict[str, Any]]:
        """Returns yield choropleth state data."""
        return self.get_states_spatial()

    def get_anomaly_map(self) -> List[Dict[str, Any]]:
        """Returns anomaly distribution data."""
        return self.get_states_spatial()

    def get_forecast_map(self) -> List[Dict[str, Any]]:
        """Returns forward forecast choropleth data."""
        return self.get_states_spatial()

    def get_spatial_clusters(self) -> List[Dict[str, Any]]:
        """Returns cluster profiles."""
        if self._cluster_metadata is None:
            self.load_metadata()
        return (self._cluster_metadata or {}).get('cluster_profiles', [])

    def get_cluster_profile(self, cluster_id: int) -> Dict[str, Any]:
        """Returns specific cluster metadata."""
        clusters = self.get_spatial_clusters()
        for c in clusters:
            if c['cluster_id'] == cluster_id:
                return c
        return {'cluster_id': cluster_id, 'status': 'NOT_FOUND'}

    def get_similar_regions(
        self,
        state: str,
        district: Optional[str] = None,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Computes cosine similarity across standardized spatial features to find similar peers.
        """
        feat_df = spatial_outlier_service.get_district_features().copy()

        # Numeric features for similarity
        num_cols = [
            'average_yield_kg_ha',
            'yield_volatility_pct',
            'average_area_k_ha',
            'trend_theil_sen_slope',
            'state_relative_yield_ratio',
            'anomaly_rate_pct'
        ]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feat_df[num_cols].values)

        # Locate reference row
        if district:
            ref_idx = feat_df[
                (feat_df['state_name'].str.lower() == state.lower()) &
                (feat_df['district_name'].str.lower() == district.lower())
            ].index
        else:
            ref_idx = feat_df[feat_df['state_name'].str.lower() == state.lower()].index

        if len(ref_idx) == 0:
            ref_idx = [0]

        target_vec = X_scaled[ref_idx[0]].reshape(1, -1)
        sim_scores = cosine_similarity(target_vec, X_scaled)[0]

        feat_df['similarity_score'] = sim_scores
        # Exclude self
        similar = feat_df.drop(index=ref_idx[0]).sort_values(by='similarity_score', ascending=False).head(top_n)

        results = []
        for _, row in similar.iterrows():
            results.append({
                'state': row['state_name'],
                'district': row['district_name'],
                'similarity_score': round(float(row['similarity_score']), 3),
                'average_yield_kg_ha': row['average_yield_kg_ha'],
                'yield_volatility_pct': row['yield_volatility_pct'],
                'trend_theil_sen_slope': row['trend_theil_sen_slope']
            })

        return results

    def get_neighboring_regions(self, state: str, district: Optional[str] = None) -> Dict[str, Any]:
        """Adjacency handler."""
        return {
            'state': state,
            'district': district or 'All Districts',
            'neighbor_data_available': False,
            'reason': (
                "Verified physical district shapefile adjacency topology is not bundled in this release. "
                "Regional comparison relies on multi-dimensional statistical similarity and agro-climatic zone clusters."
            )
        }

geospatial_service = GeospatialService()
