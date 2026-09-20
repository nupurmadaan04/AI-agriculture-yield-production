import pytest
import pandas as pd
from backend.utils.data_loader import data_loader
from src.spatial_features import spatial_feature_engine

def test_spatial_feature_generation():
    df = data_loader.dataframe
    feat_df = spatial_feature_engine.compute_district_spatial_features(df)
    
    assert len(feat_df) == 311
    required_cols = [
        'state_name', 'district_name', 'average_yield_kg_ha',
        'yield_volatility_pct', 'district_yield_zscore_state',
        'state_relative_yield_ratio', 'trend_theil_sen_slope',
        'anomaly_rate_pct'
    ]
    for col in required_cols:
        assert col in feat_df.columns

def test_within_state_zscore_properties():
    df = data_loader.dataframe
    feat_df = spatial_feature_engine.compute_district_spatial_features(df)
    
    # State mean z-score for a large state like Punjab should be close to 0
    pb = feat_df[feat_df['state_name'].str.lower() == 'punjab']
    assert len(pb) > 10
    mean_z = pb['district_yield_zscore_state'].mean()
    assert abs(mean_z) < 0.2
