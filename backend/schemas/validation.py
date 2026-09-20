from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class ModelMetricsDetail(BaseModel):
    sample_count: int
    mae: float
    rmse: float
    r2: float
    mape: float
    median_absolute_error: float
    mean_residual: float
    residual_std: float
    overprediction_rate_pct: float
    underprediction_rate_pct: float
    bias_direction: str

class BenchmarkComparisonItem(BaseModel):
    model: str
    mae: float
    rmse: float
    r2: float
    mape: float

class ValidationOverviewResponse(BaseModel):
    primary_model: str
    evaluation_type: str
    evaluation_period: str
    training_period: str
    metrics: ModelMetricsDetail
    benchmark_comparison: List[BenchmarkComparisonItem]

class StatePerformanceItem(BaseModel):
    state: str
    sample_count: int
    mean_observed_yield: float
    mean_predicted_yield: float
    mae: float
    rmse: float
    r2: float
    mape: float
    mean_residual: float
    bias_direction: str

class StatesPerformanceResponse(BaseModel):
    data: List[StatePerformanceItem]

class ScatterPointItem(BaseModel):
    year: int
    state: str
    district: str
    observed: float
    predicted: float
    residual: float
    absolute_error: float

class ResidualBinItem(BaseModel):
    bin_min: float
    bin_max: float
    bin_label: str
    count: int
    percentage: float

class ErrorPercentiles(BaseModel):
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float

class LargestErrorItem(BaseModel):
    year: int
    state: str
    district: str
    observed_yield: float
    predicted_yield: float
    residual: float
    absolute_error: float
    relative_error_pct: float

class ErrorSummaryResponse(BaseModel):
    total_test_samples: int
    mean_absolute_error: float
    median_absolute_error: float
    residual_bins: List[ResidualBinItem]
    percentiles: ErrorPercentiles
    severity_breakdown: Dict[str, Any]
    largest_errors: List[LargestErrorItem]
    state_error_rankings: List[StatePerformanceItem]

class CalibrationBucketItem(BaseModel):
    bucket_label: str
    spread_min_pct: float
    spread_max_pct: float
    sample_count: int
    percentage_of_test_set: float
    mean_spread_kg_ha: float
    mean_absolute_error_kg_ha: float
    median_absolute_error_kg_ha: float
    observed_error_rate_pct: float

class CalibrationSummaryResponse(BaseModel):
    total_evaluated_samples: int
    spread_error_correlation: float
    calibration_buckets: List[CalibrationBucketItem]
    scientific_disclaimer: str

class DriftFeatureItem(BaseModel):
    feature_name: str
    psi_score: float
    ks_statistic: float
    ks_pvalue: float
    reference_mean: float
    evaluation_mean: float
    mean_shift_pct: float
    reference_std: float
    evaluation_std: float
    status: str

class DriftOverviewResponse(BaseModel):
    reference_period: str
    evaluation_period: str
    reference_samples: int
    evaluation_samples: int
    overall_status: str
    summary_counts: Dict[str, int]
    features: List[DriftFeatureItem]
    interpretation: str

class DataQualitySubScore(BaseModel):
    score: float
    weight: float

class DataQualityResponse(BaseModel):
    overall_quality_score: float
    status: str
    records_evaluated: int
    total_features: int
    sub_scores: Dict[str, Any]
    dataset_temporal_bounds: Dict[str, int]

class ModelRegistryItem(BaseModel):
    model_id: str
    model_name: str
    version: str
    model_type: str
    task: str
    target: str
    training_period: str
    evaluation_period: str
    test_metrics: Dict[str, Any]
    features: List[str]
    artifact_path: str
    status: str
    is_primary: bool

class ModelRegistryResponse(BaseModel):
    total_registered_models: int
    models: List[ModelRegistryItem]
