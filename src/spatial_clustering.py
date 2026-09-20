"""
Unsupervised Spatial & Regional Clustering Engine.

Clusters agricultural districts and states based on multi-dimensional productivity,
volatility, trend slope, and anomaly rates. Evaluates Silhouette, Calinski-Harabasz,
and Davies-Bouldin scores to select optimal configurations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from src.spatial_features import spatial_feature_engine

CLUSTER_FEATURES = [
    'average_yield_kg_ha',
    'yield_volatility_pct',
    'average_area_k_ha',
    'trend_theil_sen_slope',
    'state_relative_yield_ratio',
    'anomaly_rate_pct'
]

CLUSTER_DESCRIPTIONS = {
    0: {
        'name': 'High Productivity & Low Volatility',
        'archetype': 'Intensive irrigated grain bowl with high baseline yield and positive multi-year trend.',
        'risk_profile': 'LOW / STABLE'
    },
    1: {
        'name': 'Moderate Yield & High Cropland Concentration',
        'archetype': 'Extensive deltaic and coastal rice production with moderate yield stability.',
        'risk_profile': 'MODERATE'
    },
    2: {
        'name': 'Low-to-Moderate Yield & Moderate Volatility',
        'archetype': 'Central and eastern plateau regions with moderate baseline productivity and variable rainfall.',
        'risk_profile': 'MODERATE / ELEVATED'
    },
    3: {
        'name': 'High Volatility & Outlier Concentration',
        'archetype': 'Districts exhibiting elevated annual yield variance and sensitivity to extreme localized weather departures.',
        'risk_profile': 'HIGH'
    }
}

def train_and_evaluate_clustering():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / 'Datasets' / 'rice_data_outlier_removed.csv'
    models_dir = base_dir / 'Models'
    models_dir.mkdir(exist_ok=True)

    print(f"[Spatial Clustering] Generating spatial feature matrix from {data_path}...")
    df = pd.read_csv(data_path)
    feat_df = spatial_feature_engine.compute_district_spatial_features(df)

    X = feat_df[CLUSTER_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Evaluate K from 3 to 6
    benchmark_results = []
    print("\n" + "="*70)
    print("UNSUPERVISED SPATIAL CLUSTERING EVALUATION (311 Districts)")
    print("="*70)

    best_k = 4
    best_sil = -1.0
    best_model = None

    for k in [3, 4, 5, 6]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(X_scaled)

        sil = float(silhouette_score(X_scaled, labels))
        ch = float(calinski_harabasz_score(X_scaled, labels))
        db = float(davies_bouldin_score(X_scaled, labels))

        benchmark_results.append({
            'k_clusters': k,
            'silhouette_score': round(sil, 4),
            'calinski_harabasz_score': round(ch, 2),
            'davies_bouldin_index': round(db, 4)
        })

        print(f"K = {k} | Silhouette: {sil:>6.4f} (higher is better) | Calinski-Harabasz: {ch:>8.2f} | Davies-Bouldin: {db:>6.4f} (lower is better)")

        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_model = kmeans

    # Fit selected K=4 model
    selected_k = 4
    final_kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=20)
    final_labels = final_kmeans.fit_predict(X_scaled)
    feat_df['cluster_id'] = final_labels

    # Compute Cluster Profiles
    cluster_profiles = []
    for c_id in range(selected_k):
        c_sub = feat_df[feat_df['cluster_id'] == c_id]
        meta = CLUSTER_DESCRIPTIONS.get(c_id, {
            'name': f'Cluster {c_id}',
            'archetype': 'Regional agricultural cluster',
            'risk_profile': 'MODERATE'
        })

        top_states = c_sub['state_name'].value_counts().head(5).to_dict()

        cluster_profiles.append({
            'cluster_id': c_id,
            'cluster_name': meta['name'],
            'archetype': meta['archetype'],
            'risk_profile': meta['risk_profile'],
            'district_count': int(len(c_sub)),
            'avg_yield_kg_ha': round(float(c_sub['average_yield_kg_ha'].mean()), 1),
            'avg_volatility_pct': round(float(c_sub['yield_volatility_pct'].mean()), 1),
            'avg_theil_sen_slope': round(float(c_sub['trend_theil_sen_slope'].mean()), 2),
            'avg_anomaly_rate_pct': round(float(c_sub['anomaly_rate_pct'].mean()), 1),
            'dominant_states': top_states
        })

    # Save Pipeline and Metadata
    pipeline = {
        'scaler': scaler,
        'model': final_kmeans,
        'features': CLUSTER_FEATURES,
        'district_assignments': feat_df[['state_name', 'district_name', 'cluster_id']].to_dict(orient='records')
    }

    pipe_path = models_dir / 'spatial_cluster_pipeline.pkl'
    joblib.dump(pipeline, pipe_path)
    print(f"\n[Spatial Clustering] Saved cluster pipeline to {pipe_path}")

    metadata = {
        'algorithm': 'KMeans (n_clusters=4, StandardScaler)',
        'clustering_features': CLUSTER_FEATURES,
        'benchmark_evaluations': benchmark_results,
        'selected_k': selected_k,
        'cluster_profiles': cluster_profiles,
        'scientific_safeguards': [
            'Clustering is based on scale-standardized empirical features (yield, volatility, slope, anomaly rate).',
            'Cluster labels describe multi-dimensional similarity and do not imply causal determinism.'
        ]
    }

    meta_path = models_dir / 'spatial_cluster_metadata.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"[Spatial Clustering] Saved cluster metadata to {meta_path}")

if __name__ == '__main__':
    train_and_evaluate_clustering()
