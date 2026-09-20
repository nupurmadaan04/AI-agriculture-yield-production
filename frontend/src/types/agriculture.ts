export interface HealthStatus {
  status: string
  dataset_loaded: boolean
  records: number
}

export interface SummaryMetrics {
  total_records: number
  total_states: number
  total_districts: number
  min_year: number
  max_year: number
  average_yield: number
  median_yield: number
  average_area: number
  average_production: number
  total_area: number
  total_production: number
  zero_yield_records: number
  missing_values: number
  duplicate_rows: number
  columns_count: number
}

export interface DatasetSummary {
  totalRecords: number
  cleanedRecords: number
  statesCovered: number
  districtsCovered: number
  yearsCovered: string
  avgYield: number
  medianYield: number
  minYield: number
  maxYield: number
  totalProduction: number
  totalArea: number
  zeroYieldCount: number
  missingValues: number
  duplicateRows: number
  columnsCount: number
}

export interface FilterOptions {
  years: number[]
  states: string[]
  districts: string[]
  crops: string[]
}

export interface FilterState {
  crop: string
  state: string
  district: string
  year: number | 'all'
}

export interface AgriculturalRecord {
  id: string
  state?: string
  district?: string
  stateName?: string
  distName?: string
  year: number
  area: number // 1000 ha
  production: number // 1000 tons
  yield: number // kg/ha
  stateCode?: number
  distCode?: number
  cropName?: string
  isZeroYield?: boolean
}

export interface Pagination {
  page: number
  page_size: number
  total: number
  total_pages: number
}

export interface PaginatedRecords {
  data: AgriculturalRecord[]
  pagination: Pagination
}

export interface TrendPoint {
  year: number
  average_yield: number
  average_area: number
  average_production: number
  total_production: number
  total_area: number
  record_count: number
}

export interface YearlyTrendPoint {
  year: number
  avgYield: number
  totalProduction: number
  totalArea: number
  recordCount: number
}

export interface TrendsResponse {
  data: TrendPoint[]
}

export interface StateSummary {
  state: string
  rank: number
  record_count: number
  district_count: number
  average_yield: number
  median_yield: number
  total_area: number
  total_production: number
}

export interface StatePerformanceSummary {
  stateCode?: number
  stateName: string
  avgYield: number
  totalProduction?: number
  totalArea?: number
  districtCount?: number
  mae?: number
  rank?: number
}

export interface StatesResponse {
  data: StateSummary[]
}

export interface DistrictSummary {
  district: string
  state: string
  year: number
  area: number
  production: number
  yield: number
}

export interface DistrictsResponse {
  data: DistrictSummary[]
}

export interface CropInfo {
  id: string
  name: string
  season: string
  coverage: string
  isPrimary: boolean
  available: boolean
}

export interface ModelMetricItem {
  id: string
  model_name: string
  model_type: string
  feature_set: string
  train_r2?: number
  random_r2?: number
  random_mae?: number
  random_rmse?: number
  temporal_r2?: number
  temporal_mae?: number
  temporal_rmse?: number
  cv_r2?: number
  cv_mae?: number
  cv_rmse?: number
  status: string
  notes?: string
}

export interface FeatureImportanceData {
  native_mdi: Record<string, number>
  permutation_importance: Record<string, any>
}

export interface AblationExperiment {
  config_name: string
  feature_set: string
  num_features: number
  random_r2: number
  random_mae: number
  temporal_r2: number
  temporal_mae: number
  group_kfold_r2: number
  interpretation: string
}

export interface ModelMetricsResponse {
  leaderboard: ModelMetricItem[]
  feature_importance?: FeatureImportanceData
  ablation_experiments: AblationExperiment[]
}

export interface DeterministicEstimateRequest {
  area: number
  production: number
}

export interface DeterministicEstimateResponse {
  estimated_yield: number
  method: string
  type: string
  unit: string
  area: number
  production: number
}

// =========================================================================
// DAY 17 MULTI-CROP AGRICULTURAL TYPES
// =========================================================================

export interface CropItem {
  crop: string
  records: number
  first_year: number
  last_year: number
  states: number
  districts: number
  has_area: boolean
  has_production: boolean
  has_yield: boolean
  forecasting_supported: boolean
  scenario_supported: boolean
  xai_supported: boolean
}

export interface CropsResponse {
  total_crops: number
  crops: CropItem[]
}

export interface CropDetailResponse {
  crop: string
  records: number
  first_year: number
  last_year: number
  states: string[]
  districts_count: number
  total_production_tonnes: number
  average_yield_kg_ha: number
  forecasting_model_status: string
  available_years: number[]
}

export interface CropAvailabilityResponse {
  crop: string
  available: boolean
  state_available: boolean
  district_available: boolean
  year_available: boolean
  matching_records: number
  supported_models: string[]
}

export interface DatasetMetadataResponse {
  dataset_version: string
  sources: string[]
  record_count: number
  crop_count: number
  state_count: number
  district_count: number
  year_range: string
  available_crops: string[]
  available_states: string[]
  quality_status: string
}
