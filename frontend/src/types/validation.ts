export interface ModelMetricsDetail {
  sample_count: number
  mae: number
  rmse: number
  r2: number
  mape: number
  median_absolute_error: number
  mean_residual: number
  residual_std: number
  overprediction_rate_pct: number
  underprediction_rate_pct: number
  bias_direction: string
}

export interface BenchmarkComparisonItem {
  model: string
  mae: number
  rmse: number
  r2: number
  mape: number
}

export interface ValidationOverviewResponse {
  primary_model: string
  evaluation_type: string
  evaluation_period: string
  training_period: string
  metrics: ModelMetricsDetail
  benchmark_comparison: BenchmarkComparisonItem[]
}

export interface StatePerformanceItem {
  state: string
  sample_count: number
  mean_observed_yield: number
  mean_predicted_yield: number
  mae: number
  rmse: number
  r2: number
  mape: number
  mean_residual: number
  bias_direction: string
}

export interface StatesPerformanceResponse {
  data: StatePerformanceItem[]
}

export interface ScatterPointItem {
  year: number
  state: string
  district: string
  observed: number
  predicted: number
  residual: number
  absolute_error: number
}

export interface ResidualBinItem {
  bin_min: number
  bin_max: number
  bin_label: string
  count: number
  percentage: number
}

export interface ErrorPercentiles {
  p25: number
  p50: number
  p75: number
  p90: number
  p95: number
  p99: number
}

export interface LargestErrorItem {
  year: number
  state: string
  district: string
  observed_yield: number
  predicted_yield: number
  residual: number
  absolute_error: number
  relative_error_pct: number
}

export interface ErrorSummaryResponse {
  total_test_samples: number
  mean_absolute_error: number
  median_absolute_error: number
  residual_bins: ResidualBinItem[]
  percentiles: ErrorPercentiles
  severity_breakdown: {
    low_error_count: number
    low_error_pct: number
    moderate_error_count: number
    moderate_error_pct: number
    high_error_count: number
    high_error_pct: number
    thresholds: {
      low_threshold: number
      moderate_threshold: number
    }
  }
  largest_errors: LargestErrorItem[]
  state_error_rankings: StatePerformanceItem[]
}

export interface CalibrationBucketItem {
  bucket_label: string
  spread_min_pct: number
  spread_max_pct: number
  sample_count: number
  percentage_of_test_set: number
  mean_spread_kg_ha: number
  mean_absolute_error_kg_ha: number
  median_absolute_error_kg_ha: number
  observed_error_rate_pct: number
}

export interface CalibrationSummaryResponse {
  total_evaluated_samples: number
  spread_error_correlation: number
  calibration_buckets: CalibrationBucketItem[]
  scientific_disclaimer: string
}

export interface DriftFeatureItem {
  feature_name: string
  psi_score: number
  ks_statistic: number
  ks_pvalue: number
  reference_mean: number
  evaluation_mean: number
  mean_shift_pct: number
  reference_std: number
  evaluation_std: number
  status: string
}

export interface DriftOverviewResponse {
  reference_period: string
  evaluation_period: string
  reference_samples: number
  evaluation_samples: number
  overall_status: string
  summary_counts: Record<string, number>
  features: DriftFeatureItem[]
  interpretation: string
}

export interface DataQualityResponse {
  overall_quality_score: number
  status: string
  records_evaluated: number
  total_features: number
  sub_scores: {
    completeness: { score: number; missing_cells: number; total_cells: number; weight: number }
    validity: { score: number; invalid_area_count: number; invalid_prod_count: number; out_of_range_yield_count: number; weight: number }
    consistency: { score: number; duplicate_keys_count: number; weight: number }
    temporal_integrity: { score: number; monitored_districts: number; districts_with_full_8yr_history: number; weight: number }
  }
  dataset_temporal_bounds: {
    start_year: number
    end_year: number
    total_years: number
  }
}

export interface ModelRegistryItem {
  model_id: string
  model_name: string
  version: string
  model_type: string
  task: string
  target: string
  training_period: string
  evaluation_period: string
  test_metrics: Record<string, any>
  features: string[]
  artifact_path: string
  status: string
  is_primary: boolean
}

export interface ModelRegistryResponse {
  total_registered_models: number
  models: ModelRegistryItem[]
}
