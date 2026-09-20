import pytest
import json
from pathlib import Path
import joblib

def test_spatial_clustering_artifacts():
    base_dir = Path(__file__).resolve().parent.parent
    pipeline_path = base_dir / 'Models' / 'spatial_cluster_pipeline.pkl'
    meta_path = base_dir / 'Models' / 'spatial_cluster_metadata.json'
    
    assert pipeline_path.exists(), "Pipeline pickle must exist"
    assert meta_path.exists(), "Metadata json must exist"
    
    pipe = joblib.load(pipeline_path)
    assert 'scaler' in pipe
    assert 'model' in pipe
    assert 'features' in pipe
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    assert meta['selected_k'] == 4
    assert len(meta['cluster_profiles']) == 4
    assert len(meta['benchmark_evaluations']) >= 4
