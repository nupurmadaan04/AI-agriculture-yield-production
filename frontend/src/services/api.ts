import { useQuery, useMutation } from '@tanstack/react-query'
import {
  HealthStatus,
  SummaryMetrics,
  FilterOptions,
  PaginatedRecords,
  TrendsResponse,
  StatesResponse,
  DistrictsResponse,
  ModelMetricsResponse,
  DeterministicEstimateRequest,
  DeterministicEstimateResponse,
  CropItem,
  CropsResponse,
  CropDetailResponse,
  CropAvailabilityResponse,
  DatasetMetadataResponse,
} from '../types/agriculture'
import {
  CropReadinessResponse,
  CropReadinessItem,
  CropBaselinesResponse,
  ReadinessSummaryResponse,
  FeatureCompatibilityResponse,
  ArchitectureDecisionResponse,
  MultiCropModelItem,
  MultiCropModelsResponse,
  CropModelComparisonResponse,
  CropModelMetricsResponse,
  CropModelFeaturesResponse,
  MultiCropLeaderboardItem,
  MultiCropLeaderboardResponse,
  CropPredictionRequest,
  CropPredictionResponse,
  FoldResultItem,
  CropRobustnessItem,
  CropRobustnessResponse,
  CropFoldsResponse,
  CropRobustnessDetailResponse,
  RobustnessSummaryResponse,
  CropDiagnosisItem,
  CropDiagnosisSummaryResponse,
  CropErrorRegimeItem,
  CropErrorRegimesResponse,
  CropDistrictErrorItem,
  CropDistrictErrorsResponse,
  CropYearErrorItem,
  CropYearErrorsResponse,
  CropFeatureStabilityItem,
  CropFeatureStabilityResponse,
  CropModelSelectionItem,
  CropModelSelectionResponse,
  CropForecastingStrategyItem,
  CropForecastingStrategyResponse,
  ExogenousSourceItem,
  ExogenousSourcesResponse,
  ExogenousCoverageItem,
  ExogenousCoverageResponse,
  ExogenousFeatureItem,
  ExogenousFeaturesResponse,
  ExogenousAblationItem,
  ExogenousAblationResponse,
  ExogenousFoldResultItem,
  ExogenousCropFoldsResponse,
  ExogenousCropResultItem,
  ExogenousCropResultResponse,
  ExogenousModelSelectionItem,
  ExogenousModelSelectionResponse,
  ExogenousSummaryResponse,
  FinalStrategyItem,
  FinalValidationFoldItem,
  FinalValidationResponse,
  SingleCropFinalValidationResponse,
  ResidualQuantileItem,
  ResidualYearItem,
  ResidualDiagnosticsResponse,
  PredictionBiasItem,
  PredictionBiasResponse,
  ReproducibilityItem,
  ReproducibilityResponse,
  FinalModelCertificationItem,
  FinalModelCertificationResponse,
  ForecastStrategiesResponse,
  ForecastCoverageResponse,
  ForecastCertificationSummaryResponse,
  ForecastPredictRequest,
  ForecastPredictResponse,
  ForecastAuditResponse,
  ForecastHealthResponse,
} from '../types/modeling'
import {
  SystemHealthItem,
  RuntimeMetricsResponse,
  ForecastOperationsMetrics,
  StrategyMonitoringResponse,
  PredictionTraceResponse,
  ModelIntegrityResponse,
  DatasetIntegrityResponse,
  StrategyRegistryHealthResponse,
  OperationalErrorsResponse,
  AlertsResponse,
  DriftMonitoringResponse,
  ObservabilitySummaryResponse,
} from '../types/observability'
import {
  PostHarvestPredictRequest,
  PostHarvestPredictResponse,
  PreSeasonPredictRequest,
  PreSeasonPredictResponse,
  PreSeasonAdvancedPredictRequest,
  PreSeasonAdvancedPredictResponse,
  ModelsListResponse,
  ErrorAnalysisResponse,
  RiskAssessmentRequest,
  RiskAssessmentResponse,
  ExplainabilityRequest,
  ExplainabilityResponse,
  AnomalyDetectionRequest,
  AnomalyDetectionResponse,
  StateRiskResponse,
  AnomalyFeedResponse,
  IntelligenceDashboardResponse,
} from '../types/model'
import {
  ScenarioSimulationRequest,
  ScenarioSimulationResponse,
  CopilotQueryRequest,
  CopilotQueryResponse,
  ReportGenerateRequest,
  ReportGenerateResponse,
  DecisionSupportResponse,
} from '../types/intelligence'
import {
  ForecastYieldRequest,
  ForecastYieldResponse,
  TrendAnalyzeResponse,
  StateTrendItem,
  EarlyWarningAssessResponse,
  EarlyWarningDashboardResponse,
} from '../types/temporal'
import {
  StateSpatialItem,
  GeospatialOverviewResponse,
  SpatialClusterItem,
  StateSpatialProfileResponse,
} from '../types/geospatial'
import {
  ValidationOverviewResponse,
  StatesPerformanceResponse,
  ScatterPointItem,
  ErrorSummaryResponse,
  CalibrationSummaryResponse,
  DriftOverviewResponse,
  DriftFeatureItem,
  DataQualityResponse,
  ModelRegistryResponse,
  ModelRegistryItem,
} from '../types/validation'
import {
  ScenarioResult,
  ScenarioComparisonResult,
  SensitivityResult,
  OptimizationResult,
  ScenarioAuditItem,
  ScenarioHistoryResult,
} from '../types/scenario'
import {
  MonitoringOverview,
  TemporalSignal,
  Alert,
  StateWarningMapItem,
  ChangeDetectionResult,
  WarningBacktest,
  MonitoringHealth,
  BacktestRequest,
} from '../types/monitoring'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL !== undefined
    ? import.meta.env.VITE_API_BASE_URL
    : import.meta.env.PROD
    ? '/api'
    : 'http://localhost:8000/api'

// Generic fetch wrapper with clean error extraction
async function fetchJson<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  })

  if (!response.ok) {
    let errorMessage = `HTTP error ${response.status}: ${response.statusText}`
    try {
      const errBody = await response.json()
      if (errBody.detail) errorMessage = errBody.detail
    } catch {
      // ignore
    }
    throw new Error(errorMessage)
  }

  return response.json()
}

// API Service functions
export const api = {
  getHealth: (): Promise<HealthStatus> => fetchJson<HealthStatus>('/health'),

  getSummary: (crop?: string): Promise<SummaryMetrics> =>
    fetchJson<SummaryMetrics>(`/summary${crop ? `?crop=${encodeURIComponent(crop)}` : ''}`),

  getFilters: (crop?: string): Promise<FilterOptions> =>
    fetchJson<FilterOptions>(`/filters${crop ? `?crop=${encodeURIComponent(crop)}` : ''}`),

  getAgricultureCrops: (): Promise<CropsResponse> =>
    fetchJson<CropsResponse>('/agriculture/crops'),

  getCropDetail: (crop: string): Promise<CropDetailResponse> =>
    fetchJson<CropDetailResponse>(`/agriculture/crops/${encodeURIComponent(crop)}`),

  getAgricultureCoverage: (): Promise<CropsResponse> =>
    fetchJson<CropsResponse>('/agriculture/coverage'),

  getAgricultureSummary: (crop?: string): Promise<SummaryMetrics> =>
    fetchJson<SummaryMetrics>(`/agriculture/summary${crop ? `?crop=${encodeURIComponent(crop)}` : ''}`),

  getAgricultureAvailability: (params?: {
    crop?: string
    state?: string
    district?: string
    year?: number
    season?: string
  }): Promise<CropAvailabilityResponse> => {
    const query = new URLSearchParams()
    if (params?.crop) query.set('crop', params.crop)
    if (params?.state && params.state !== 'all') query.set('state', params.state)
    if (params?.district && params.district !== 'all') query.set('district', params.district)
    if (params?.year) query.set('year', params.year.toString())
    if (params?.season) query.set('season', params.season)
    const qs = query.toString()
    return fetchJson<CropAvailabilityResponse>(`/agriculture/availability${qs ? `?${qs}` : ''}`)
  },

  getDatasetMetadata: (): Promise<DatasetMetadataResponse> =>
    fetchJson<DatasetMetadataResponse>('/agriculture/metadata'),

  getRecords: (params?: {
    page?: number
    page_size?: number
    year?: number | string
    state?: string
    district?: string
    search?: string
    crop?: string
  }): Promise<PaginatedRecords> => {
    const query = new URLSearchParams()
    if (params?.page) query.set('page', params.page.toString())
    if (params?.page_size) query.set('page_size', params.page_size.toString())
    if (params?.year && params.year !== 'all') query.set('year', params.year.toString())
    if (params?.state && params.state !== 'all') query.set('state', params.state)
    if (params?.district && params.district !== 'all') query.set('district', params.district)
    if (params?.search) query.set('search', params.search)
    if (params?.crop) query.set('crop', params.crop)

    const qs = query.toString()
    return fetchJson<PaginatedRecords>(`/records${qs ? `?${qs}` : ''}`)
  },

  getTrends: (params?: {
    state?: string
    district?: string
    year_start?: number
    year_end?: number
    crop?: string
  }): Promise<TrendsResponse> => {
    const query = new URLSearchParams()
    if (params?.state && params.state !== 'all') query.set('state', params.state)
    if (params?.district && params.district !== 'all') query.set('district', params.district)
    if (params?.year_start) query.set('year_start', params.year_start.toString())
    if (params?.year_end) query.set('year_end', params.year_end.toString())
    if (params?.crop) query.set('crop', params.crop)

    const qs = query.toString()
    return fetchJson<TrendsResponse>(`/trends${qs ? `?${qs}` : ''}`)
  },

  getStates: (crop?: string): Promise<StatesResponse> =>
    fetchJson<StatesResponse>(`/states${crop ? `?crop=${encodeURIComponent(crop)}` : ''}`),

  getDistricts: (params?: { state?: string; year?: number; crop?: string }): Promise<DistrictsResponse> => {
    const query = new URLSearchParams()
    if (params?.state && params.state !== 'all') query.set('state', params.state)
    if (params?.year) query.set('year', params.year.toString())
    if (params?.crop) query.set('crop', params.crop)

    const qs = query.toString()
    return fetchJson<DistrictsResponse>(`/districts${qs ? `?${qs}` : ''}`)
  },

  getModelMetrics: (): Promise<ModelMetricsResponse> =>
    fetchJson<ModelMetricsResponse>('/model-metrics'),

  estimateDeterministic: (
    payload: DeterministicEstimateRequest
  ): Promise<DeterministicEstimateResponse> =>
    fetchJson<DeterministicEstimateResponse>('/estimate/deterministic', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  // ML Endpoints
  predictPostHarvest: (
    payload: PostHarvestPredictRequest
  ): Promise<PostHarvestPredictResponse> =>
    fetchJson<PostHarvestPredictResponse>('/predict/post-harvest', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  predictPreSeason: (
    payload: PreSeasonPredictRequest
  ): Promise<PreSeasonPredictResponse> =>
    fetchJson<PreSeasonPredictResponse>('/predict/pre-season', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  predictPreSeasonAdvanced: (
    payload: PreSeasonAdvancedPredictRequest
  ): Promise<PreSeasonAdvancedPredictResponse> =>
    fetchJson<PreSeasonAdvancedPredictResponse>('/predict/pre-season/advanced', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  getModels: (): Promise<ModelsListResponse> =>
    fetchJson<ModelsListResponse>('/models'),

  getErrorAnalysis: (): Promise<ErrorAnalysisResponse> =>
    fetchJson<ErrorAnalysisResponse>('/error-analysis'),

  // Day 5 Agricultural Intelligence Endpoints
  predictRisk: (
    payload: RiskAssessmentRequest
  ): Promise<RiskAssessmentResponse> =>
    fetchJson<RiskAssessmentResponse>('/intelligence/risk', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  explainPrediction: (
    payload: ExplainabilityRequest
  ): Promise<ExplainabilityResponse> =>
    fetchJson<ExplainabilityResponse>('/intelligence/explain', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  detectAnomaly: (
    payload: AnomalyDetectionRequest
  ): Promise<AnomalyDetectionResponse> =>
    fetchJson<AnomalyDetectionResponse>('/intelligence/anomaly', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  getIntelligenceDashboard: (): Promise<IntelligenceDashboardResponse> =>
    fetchJson<IntelligenceDashboardResponse>('/intelligence/dashboard'),

  getStateRisk: (): Promise<StateRiskResponse> =>
    fetchJson<StateRiskResponse>('/intelligence/state-risk'),

  getAnomalies: (limit = 50): Promise<AnomalyFeedResponse> =>
    fetchJson<AnomalyFeedResponse>(`/intelligence/anomalies?limit=${limit}`),

  // Day 6 Decision Intelligence Endpoints
  predictScenario: (
    payload: ScenarioSimulationRequest
  ): Promise<ScenarioSimulationResponse> =>
    fetchJson<ScenarioSimulationResponse>('/scenario/simulate', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  queryCopilot: (
    payload: CopilotQueryRequest
  ): Promise<CopilotQueryResponse> =>
    fetchJson<CopilotQueryResponse>('/copilot/query', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  generateReport: (
    payload: ReportGenerateRequest
  ): Promise<ReportGenerateResponse> =>
    fetchJson<ReportGenerateResponse>('/reports/generate', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  getDecisionSupport: (): Promise<DecisionSupportResponse> =>
    fetchJson<DecisionSupportResponse>('/decision-support'),

  // Day 7 Temporal & Early Warning Endpoints
  forecastYield: (payload: ForecastYieldRequest): Promise<ForecastYieldResponse> =>
    fetchJson<ForecastYieldResponse>('/forecast/yield', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  getForecastState: (state: string): Promise<ForecastYieldResponse> =>
    fetchJson<ForecastYieldResponse>(`/forecast/state/${encodeURIComponent(state)}`),

  getForecastDistrict: (district: string, state?: string): Promise<ForecastYieldResponse> =>
    fetchJson<ForecastYieldResponse>(`/forecast/district/${encodeURIComponent(district)}${state ? `?state=${encodeURIComponent(state)}` : ''}`),

  analyzeTrends: (payload: { state?: string; district?: string }): Promise<TrendAnalyzeResponse> =>
    fetchJson<TrendAnalyzeResponse>('/trends/analyze', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  getStatesTrends: (): Promise<{ data: StateTrendItem[] }> =>
    fetchJson<{ data: StateTrendItem[] }>('/trends/states'),

  assessEarlyWarning: (payload: { state?: string; district?: string }): Promise<EarlyWarningAssessResponse> =>
    fetchJson<EarlyWarningAssessResponse>('/early-warning/assess', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  getEarlyWarningDashboard: (): Promise<EarlyWarningDashboardResponse> =>
    fetchJson<EarlyWarningDashboardResponse>('/early-warning/dashboard'),

  getStatesEarlyWarning: (): Promise<{ data: EarlyWarningAssessResponse[] }> =>
    fetchJson<{ data: EarlyWarningAssessResponse[] }>('/early-warning/states'),

  // Day 8 Geospatial Intelligence
  getGeospatialOverview: (): Promise<GeospatialOverviewResponse> =>
    fetchJson<GeospatialOverviewResponse>('/geospatial/overview'),

  getGeospatialStates: (): Promise<StateSpatialItem[]> =>
    fetchJson<StateSpatialItem[]>('/geospatial/states'),

  getGeospatialStateProfile: (state: string): Promise<StateSpatialProfileResponse> =>
    fetchJson<StateSpatialProfileResponse>(`/geospatial/state/${encodeURIComponent(state)}`),

  getGeospatialDistrictProfile: (district: string, state?: string): Promise<any> =>
    fetchJson<any>(`/geospatial/district/${encodeURIComponent(district)}${state ? `?state=${encodeURIComponent(state)}` : ''}`),

  getGeospatialRiskMap: (): Promise<StateSpatialItem[]> =>
    fetchJson<StateSpatialItem[]>('/geospatial/risk-map'),

  getGeospatialYieldMap: (): Promise<StateSpatialItem[]> =>
    fetchJson<StateSpatialItem[]>('/geospatial/yield-map'),

  getGeospatialAnomalyMap: (): Promise<StateSpatialItem[]> =>
    fetchJson<StateSpatialItem[]>('/geospatial/anomaly-map'),

  getGeospatialForecastMap: (): Promise<StateSpatialItem[]> =>
    fetchJson<StateSpatialItem[]>('/geospatial/forecast-map'),

  getGeospatialClusters: (): Promise<SpatialClusterItem[]> =>
    fetchJson<SpatialClusterItem[]>('/geospatial/clusters'),

  getGeospatialClusterDetail: (clusterId: number): Promise<SpatialClusterItem> =>
    fetchJson<SpatialClusterItem>(`/geospatial/clusters/${clusterId}`),

  queryGeospatial: (payload: any): Promise<any> =>
    fetchJson<any>('/geospatial/query', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  // Day 9 Model Reliability, Validation & Monitoring
  getValidationOverview: (): Promise<ValidationOverviewResponse> =>
    fetchJson<ValidationOverviewResponse>('/validation/overview'),

  getValidationStates: (): Promise<StatesPerformanceResponse> =>
    fetchJson<StatesPerformanceResponse>('/validation/states'),

  getValidationScatter: (): Promise<ScatterPointItem[]> =>
    fetchJson<ScatterPointItem[]>('/validation/prediction', { method: 'POST' }),

  getErrorsSummary: (): Promise<ErrorSummaryResponse> =>
    fetchJson<ErrorSummaryResponse>('/errors/summary'),

  getCalibrationSummary: (): Promise<CalibrationSummaryResponse> =>
    fetchJson<CalibrationSummaryResponse>('/calibration/summary'),

  getDriftOverview: (): Promise<DriftOverviewResponse> =>
    fetchJson<DriftOverviewResponse>('/drift/overview'),

  getDriftFeatures: (): Promise<DriftFeatureItem[]> =>
    fetchJson<DriftFeatureItem[]>('/drift/features'),

  getDataQuality: (): Promise<DataQualityResponse> =>
    fetchJson<DataQualityResponse>('/data-quality'),

  getModelRegistry: (): Promise<ModelRegistryResponse> =>
    fetchJson<ModelRegistryResponse>('/models/registry'),

  getModelDetail: (modelName: string): Promise<ModelRegistryItem> =>
    fetchJson<ModelRegistryItem>(`/models/${encodeURIComponent(modelName)}`),

  // Day 10 Scenario Intelligence Endpoints
  simulateScenario: (payload: any): Promise<ScenarioResult> =>
    fetchJson<ScenarioResult>('/scenario/simulate', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  compareScenarios: (payload: any): Promise<ScenarioComparisonResult> =>
    fetchJson<ScenarioComparisonResult>('/scenario/compare', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  analyzeSensitivity: (payload: any): Promise<SensitivityResult> =>
    fetchJson<SensitivityResult>('/scenario/sensitivity', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  optimizeDecision: (payload: any): Promise<OptimizationResult> =>
    fetchJson<OptimizationResult>('/scenario/optimize', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  getScenarioTemplates: (): Promise<any> =>
    fetchJson<any>('/scenario/templates'),

  getScenarioHistory: (limit = 20): Promise<ScenarioHistoryResult> =>
    fetchJson<ScenarioHistoryResult>(`/scenario/history?limit=${limit}`),

  getScenarioAudit: (scenarioId: string): Promise<ScenarioAuditItem> =>
    fetchJson<ScenarioAuditItem>(`/scenario/${encodeURIComponent(scenarioId)}/audit`),

  // Day 12 Monitoring & Early Warning Endpoints
  getMonitoringOverview: (): Promise<MonitoringOverview> =>
    fetchJson<MonitoringOverview>('/monitoring/overview'),

  getMonitoringTimeline: (state = 'Punjab', district?: string, metric = 'yield'): Promise<TemporalSignal> =>
    fetchJson<TemporalSignal>(`/monitoring/timeline?state=${encodeURIComponent(state)}&metric=${encodeURIComponent(metric)}${district ? `&district=${encodeURIComponent(district)}` : ''}`),

  getMonitoringStates: (): Promise<StateWarningMapItem[]> =>
    fetchJson<StateWarningMapItem[]>('/monitoring/states'),

  getMonitoringDistricts: (state?: string): Promise<Alert[]> =>
    fetchJson<Alert[]>(`/monitoring/districts${state ? `?state=${encodeURIComponent(state)}` : ''}`),

  getMonitoringAlerts: (params?: { state?: string; district?: string; severity?: string; signal_type?: string; year?: number; limit?: number }): Promise<Alert[]> => {
    const q = new URLSearchParams()
    if (params?.state) q.append('state', params.state)
    if (params?.district) q.append('district', params.district)
    if (params?.severity) q.append('severity', params.severity)
    if (params?.signal_type) q.append('signal_type', params.signal_type)
    if (params?.year) q.append('year', String(params.year))
    if (params?.limit) q.append('limit', String(params.limit))
    return fetchJson<Alert[]>(`/monitoring/alerts${q.toString() ? `?${q.toString()}` : ''}`)
  },

  getAlertById: (alertId: string): Promise<Alert> =>
    fetchJson<Alert>(`/monitoring/alerts/${encodeURIComponent(alertId)}`),

  getWarningMap: (): Promise<StateWarningMapItem[]> =>
    fetchJson<StateWarningMapItem[]>('/monitoring/warning-map'),

  getMonitoringHealth: (): Promise<MonitoringHealth> =>
    fetchJson<MonitoringHealth>('/monitoring/health'),

  queryMonitoringAlerts: (payload: any): Promise<Alert[]> =>
    fetchJson<Alert[]>('/monitoring/query', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  runWarningBacktest: (payload: BacktestRequest): Promise<WarningBacktest> =>
    fetchJson<WarningBacktest>('/monitoring/backtest', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  // Day 13 Explainable AI (XAI) Endpoints
  getGlobalImportance: (): Promise<import('../types/explainability').GlobalImportanceResponse> =>
    fetchJson<import('../types/explainability').GlobalImportanceResponse>('/explainability/global'),

  explainPredictionXAI: (payload: import('../types/explainability').LocalExplanationRequest): Promise<import('../types/explainability').LocalExplanationResponse> =>
    fetchJson<import('../types/explainability').LocalExplanationResponse>('/explainability/prediction', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  getSensitivityAnalysis: (payload: import('../types/explainability').SensitivityRequest): Promise<import('../types/explainability').SensitivityResponse> =>
    fetchJson<import('../types/explainability').SensitivityResponse>('/explainability/sensitivity', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  explainAlert: (alertId: string): Promise<import('../types/explainability').AlertExplanationResponse> =>
    fetchJson<import('../types/explainability').AlertExplanationResponse>(`/explainability/alert/${encodeURIComponent(alertId)}`),

  explainScenario: (
    scenarioId: string,
    params?: { state?: string; baseline_yield?: number; simulated_yield?: number; changed_features?: Record<string, number> }
  ): Promise<import('../types/explainability').ScenarioExplanationResponse> => {
    const q = new URLSearchParams()
    if (params?.state) q.append('state', params.state)
    if (params?.baseline_yield) q.append('baseline_yield', String(params.baseline_yield))
    if (params?.simulated_yield) q.append('simulated_yield', String(params.simulated_yield))
    return fetchJson<import('../types/explainability').ScenarioExplanationResponse>(
      `/explainability/scenario/${encodeURIComponent(scenarioId)}${q.toString() ? `?${q.toString()}` : ''}`,
      {
        method: 'POST',
        body: JSON.stringify(params?.changed_features || {}),
      }
    )
  },

  getExplanationAudit: (explanationId: string): Promise<import('../types/explainability').ExplanationAuditResponse> =>
    fetchJson<import('../types/explainability').ExplanationAuditResponse>(`/explainability/audit/${encodeURIComponent(explanationId)}`),

  getExplainabilityValidation: (): Promise<import('../types/explainability').ExplanationValidationResponse> =>
    fetchJson<import('../types/explainability').ExplanationValidationResponse>('/explainability/validation'),

  // Day 18: Multi-Crop Modeling Readiness & Baselines
  getReadinessSummary: (): Promise<ReadinessSummaryResponse> =>
    fetchJson<ReadinessSummaryResponse>('/modeling/readiness/summary'),

  getCropReadinessAll: (status?: string): Promise<CropReadinessResponse> =>
    fetchJson<CropReadinessResponse>(`/modeling/crops/readiness${status ? `?status=${encodeURIComponent(status)}` : ''}`),

  getCropReadinessSingle: (crop: string): Promise<CropReadinessItem> =>
    fetchJson<CropReadinessItem>(`/modeling/crops/${encodeURIComponent(crop)}/readiness`),

  getCropBaselines: (crop: string): Promise<CropBaselinesResponse> =>
    fetchJson<CropBaselinesResponse>(`/modeling/crops/${encodeURIComponent(crop)}/baselines`),

  getFeatureCompatibility: (): Promise<FeatureCompatibilityResponse> =>
    fetchJson<FeatureCompatibilityResponse>('/modeling/feature-compatibility'),

  getArchitectureDecision: (): Promise<ArchitectureDecisionResponse> =>
    fetchJson<ArchitectureDecisionResponse>('/modeling/architecture-decision'),

  // Day 19: Multi-Crop Forecasting Endpoints
  getMultiCropModels: (status?: string): Promise<MultiCropModelsResponse> =>
    fetchJson<MultiCropModelsResponse>(`/modeling/models${status ? `?status=${encodeURIComponent(status)}` : ''}`),

  getCropModelDetails: (crop: string): Promise<MultiCropModelItem> =>
    fetchJson<MultiCropModelItem>(`/modeling/models/${encodeURIComponent(crop)}`),

  getCropModelComparison: (crop: string): Promise<CropModelComparisonResponse> =>
    fetchJson<CropModelComparisonResponse>(`/modeling/models/${encodeURIComponent(crop)}/comparison`),

  getCropModelMetrics: (crop: string): Promise<CropModelMetricsResponse> =>
    fetchJson<CropModelMetricsResponse>(`/modeling/models/${encodeURIComponent(crop)}/metrics`),

  getCropModelFeatures: (crop: string): Promise<CropModelFeaturesResponse> =>
    fetchJson<CropModelFeaturesResponse>(`/modeling/models/${encodeURIComponent(crop)}/features`),

  getMultiCropLeaderboard: (): Promise<MultiCropLeaderboardResponse> =>
    fetchJson<MultiCropLeaderboardResponse>('/modeling/leaderboard'),

  predictMultiCropYield: (crop: string, payload: CropPredictionRequest): Promise<CropPredictionResponse> =>
    fetchJson<CropPredictionResponse>(`/modeling/models/${encodeURIComponent(crop)}/predict`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  // Day 20: Temporal Robustness & Walk-Forward Validation Endpoints
  getCropRobustnessAll: (): Promise<CropRobustnessResponse> =>
    fetchJson<CropRobustnessResponse>('/modeling/robustness'),

  getRobustnessSummary: (): Promise<RobustnessSummaryResponse> =>
    fetchJson<RobustnessSummaryResponse>('/modeling/robustness/summary'),

  getCropRobustness: (crop: string): Promise<CropRobustnessItem> =>
    fetchJson<CropRobustnessItem>(`/modeling/robustness/${encodeURIComponent(crop)}`),

  getCropFolds: (crop: string): Promise<CropFoldsResponse> =>
    fetchJson<CropFoldsResponse>(`/modeling/robustness/${encodeURIComponent(crop)}/folds`),

  getCropRobustnessDetail: (crop: string): Promise<CropRobustnessDetailResponse> =>
    fetchJson<CropRobustnessDetailResponse>(`/modeling/robustness/${encodeURIComponent(crop)}/comparison`),

  // Day 21: Model Diagnosis & Strategy Endpoints
  getCropDiagnosisAll: (): Promise<CropDiagnosisSummaryResponse> =>
    fetchJson<CropDiagnosisSummaryResponse>('/modeling/diagnosis'),

  getCropDiagnosis: (crop: string): Promise<CropDiagnosisItem> =>
    fetchJson<CropDiagnosisItem>(`/modeling/diagnosis/${encodeURIComponent(crop)}`),

  getCropErrorRegimes: (crop: string): Promise<CropErrorRegimesResponse> =>
    fetchJson<CropErrorRegimesResponse>(`/modeling/diagnosis/${encodeURIComponent(crop)}/errors`),

  getCropDistrictErrors: (crop: string): Promise<CropDistrictErrorsResponse> =>
    fetchJson<CropDistrictErrorsResponse>(`/modeling/diagnosis/${encodeURIComponent(crop)}/districts`),

  getCropYearErrors: (crop: string): Promise<CropYearErrorsResponse> =>
    fetchJson<CropYearErrorsResponse>(`/modeling/diagnosis/${encodeURIComponent(crop)}/years`),

  getCropFeatureStability: (crop: string): Promise<CropFeatureStabilityResponse> =>
    fetchJson<CropFeatureStabilityResponse>(`/modeling/diagnosis/${encodeURIComponent(crop)}/features`),

  getCropModelSelectionAll: (): Promise<CropModelSelectionResponse> =>
    fetchJson<CropModelSelectionResponse>('/modeling/selection'),

  getCropModelSelection: (crop: string): Promise<CropModelSelectionItem> =>
    fetchJson<CropModelSelectionItem>(`/modeling/selection/${encodeURIComponent(crop)}`),

  getCropForecastingStrategyAll: (): Promise<CropForecastingStrategyResponse> =>
    fetchJson<CropForecastingStrategyResponse>('/modeling/forecasting-strategy'),

  getCropForecastingStrategy: (crop: string): Promise<CropForecastingStrategyItem> =>
    fetchJson<CropForecastingStrategyItem>(`/modeling/forecasting-strategy/${encodeURIComponent(crop)}`),

  // Day 22: Exogenous Data & Pre-Season Feature Expansion Endpoints
  getExogenousSummary: (): Promise<ExogenousSummaryResponse> =>
    fetchJson<ExogenousSummaryResponse>('/modeling/exogenous'),

  getExogenousSources: (): Promise<ExogenousSourcesResponse> =>
    fetchJson<ExogenousSourcesResponse>('/modeling/exogenous/sources'),

  getExogenousCoverage: (): Promise<ExogenousCoverageResponse> =>
    fetchJson<ExogenousCoverageResponse>('/modeling/exogenous/coverage'),

  getExogenousFeatures: (): Promise<ExogenousFeaturesResponse> =>
    fetchJson<ExogenousFeaturesResponse>('/modeling/exogenous/features'),

  getExogenousAblation: (): Promise<ExogenousAblationResponse> =>
    fetchJson<ExogenousAblationResponse>('/modeling/exogenous/ablation'),

  getExogenousSelectionAll: (): Promise<ExogenousModelSelectionResponse> =>
    fetchJson<ExogenousModelSelectionResponse>('/modeling/exogenous/selection'),

  getExogenousCropResult: (crop: string): Promise<ExogenousCropResultResponse> =>
    fetchJson<ExogenousCropResultResponse>(`/modeling/exogenous/${encodeURIComponent(crop)}`),

  getExogenousCropFolds: (crop: string): Promise<ExogenousCropFoldsResponse> =>
    fetchJson<ExogenousCropFoldsResponse>(`/modeling/exogenous/${encodeURIComponent(crop)}/folds`),

  // Day 23: Final Temporal Validation, Residual Diagnostics & Model Certification
  getFinalValidationSummary: (): Promise<FinalValidationResponse> =>
    fetchJson<FinalValidationResponse>('/modeling/final-validation'),

  getSingleCropFinalValidation: (crop: string): Promise<SingleCropFinalValidationResponse> =>
    fetchJson<SingleCropFinalValidationResponse>(`/modeling/final-validation/${encodeURIComponent(crop)}`),

  getCropResidualDiagnostics: (crop: string): Promise<ResidualDiagnosticsResponse> =>
    fetchJson<ResidualDiagnosticsResponse>(`/modeling/final-validation/${encodeURIComponent(crop)}/residuals`),

  getCropPredictionBias: (crop: string): Promise<PredictionBiasResponse> =>
    fetchJson<PredictionBiasResponse>(`/modeling/final-validation/${encodeURIComponent(crop)}/bias`),

  getCropFinalStrategy: (crop: string): Promise<FinalStrategyItem> =>
    fetchJson<FinalStrategyItem>(`/modeling/final-validation/${encodeURIComponent(crop)}/strategy`),

  getReproducibilityAudit: (): Promise<ReproducibilityResponse> =>
    fetchJson<ReproducibilityResponse>('/modeling/final-validation/reproducibility'),

  getFinalCertification: (): Promise<FinalModelCertificationResponse> =>
    fetchJson<FinalModelCertificationResponse>('/modeling/certification'),

  // Day 24: Production Forecast Serving & Governance Endpoints
  getForecastStrategies: (): Promise<ForecastStrategiesResponse> =>
    fetchJson<ForecastStrategiesResponse>('/forecast/strategies'),

  getForecastCertificationSummary: (): Promise<ForecastCertificationSummaryResponse> =>
    fetchJson<ForecastCertificationSummaryResponse>('/forecast/certification'),

  getForecastCoverage: (): Promise<ForecastCoverageResponse> =>
    fetchJson<ForecastCoverageResponse>('/forecast/coverage'),

  predictForecast: (payload: ForecastPredictRequest): Promise<ForecastPredictResponse> =>
    fetchJson<ForecastPredictResponse>('/forecast/predict', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  getForecastProvenance: (requestId: string): Promise<any> =>
    fetchJson<any>(`/forecast/provenance/${encodeURIComponent(requestId)}`),

  getForecastAuditLogs: (limit = 50): Promise<ForecastAuditResponse> =>
    fetchJson<ForecastAuditResponse>(`/forecast/audit?limit=${limit}`),

  getForecastHealth: (): Promise<ForecastHealthResponse> =>
    fetchJson<ForecastHealthResponse>('/forecast/health'),

  // Day 28: Production Observability & Operations Endpoints
  getObservabilitySummary: (): Promise<ObservabilitySummaryResponse> =>
    fetchJson<ObservabilitySummaryResponse>('/observability/summary'),

  getObservabilityHealth: (): Promise<SystemHealthItem> =>
    fetchJson<SystemHealthItem>('/observability/health'),

  getObservabilityMetrics: (): Promise<RuntimeMetricsResponse> =>
    fetchJson<RuntimeMetricsResponse>('/observability/metrics'),

  getObservabilityForecasts: (): Promise<ForecastOperationsMetrics> =>
    fetchJson<ForecastOperationsMetrics>('/observability/forecasts'),

  getObservabilityStrategies: (): Promise<StrategyMonitoringResponse> =>
    fetchJson<StrategyMonitoringResponse>('/observability/strategies'),

  getObservabilityModels: (): Promise<ModelIntegrityResponse> =>
    fetchJson<ModelIntegrityResponse>('/observability/models'),

  getObservabilityDataset: (): Promise<DatasetIntegrityResponse> =>
    fetchJson<DatasetIntegrityResponse>('/observability/dataset'),

  getObservabilityRegistry: (): Promise<StrategyRegistryHealthResponse> =>
    fetchJson<StrategyRegistryHealthResponse>('/observability/registry'),

  getPredictionTrace: (requestId: string): Promise<PredictionTraceResponse> =>
    fetchJson<PredictionTraceResponse>(`/observability/trace/${encodeURIComponent(requestId)}`),

  getObservabilityErrors: (limit = 50): Promise<OperationalErrorsResponse> =>
    fetchJson<OperationalErrorsResponse>(`/observability/errors?limit=${limit}`),

  getObservabilityAlerts: (): Promise<AlertsResponse> =>
    fetchJson<AlertsResponse>('/observability/alerts'),

  getObservabilityDrift: (): Promise<DriftMonitoringResponse> =>
    fetchJson<DriftMonitoringResponse>('/observability/drift'),
}

// TanStack Query Custom Hooks
export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: api.getHealth,
    staleTime: 1000 * 30,
  })
}

export function useAgricultureCrops() {
  return useQuery({
    queryKey: ['agriculture-crops'],
    queryFn: api.getAgricultureCrops,
    staleTime: 1000 * 60 * 30,
  })
}

export function useCropDetail(crop: string) {
  return useQuery({
    queryKey: ['crop-detail', crop],
    queryFn: () => api.getCropDetail(crop),
    enabled: Boolean(crop),
    staleTime: 1000 * 60 * 10,
  })
}

export function useAgricultureAvailability(params?: {
  crop?: string
  state?: string
  district?: string
  year?: number
  season?: string
}) {
  return useQuery({
    queryKey: ['agriculture-availability', params],
    queryFn: () => api.getAgricultureAvailability(params),
    staleTime: 1000 * 60 * 5,
  })
}

export function useDatasetMetadata() {
  return useQuery({
    queryKey: ['dataset-metadata'],
    queryFn: api.getDatasetMetadata,
    staleTime: 1000 * 60 * 60,
  })
}

export function useSummary(crop?: string) {
  return useQuery({
    queryKey: ['summary', crop],
    queryFn: () => api.getSummary(crop),
    staleTime: 1000 * 60 * 10,
  })
}

export function useFilters(crop?: string) {
  return useQuery({
    queryKey: ['filters', crop],
    queryFn: () => api.getFilters(crop),
    staleTime: 1000 * 60 * 30,
  })
}

export function useRecords(params?: {
  page?: number
  page_size?: number
  year?: number | string
  state?: string
  district?: string
  search?: string
  crop?: string
}) {
  return useQuery({
    queryKey: ['records', params],
    queryFn: () => api.getRecords(params),
    staleTime: 1000 * 60,
  })
}

export function useTrends(params?: {
  state?: string
  district?: string
  year_start?: number
  year_end?: number
  crop?: string
}) {
  return useQuery({
    queryKey: ['trends', params],
    queryFn: () => api.getTrends(params),
    staleTime: 1000 * 60 * 5,
  })
}

export function useStates(crop?: string) {
  return useQuery({
    queryKey: ['states', crop],
    queryFn: () => api.getStates(crop),
    staleTime: 1000 * 60 * 10,
  })
}

export function useDistricts(params?: { state?: string; year?: number }) {
  return useQuery({
    queryKey: ['districts', params],
    queryFn: () => api.getDistricts(params),
    staleTime: 1000 * 60 * 5,
  })
}

export function useModelMetrics() {
  return useQuery({
    queryKey: ['model-metrics'],
    queryFn: api.getModelMetrics,
    staleTime: 1000 * 60 * 30,
  })
}

export function useModels() {
  return useQuery({
    queryKey: ['models'],
    queryFn: api.getModels,
    staleTime: 1000 * 60 * 30,
  })
}

export function useErrorAnalysis() {
  return useQuery({
    queryKey: ['error-analysis'],
    queryFn: api.getErrorAnalysis,
    staleTime: 1000 * 60 * 30,
  })
}

export function usePredictPostHarvest() {
  return useMutation({
    mutationFn: (payload: PostHarvestPredictRequest) => api.predictPostHarvest(payload),
  })
}

export function usePredictPreSeason() {
  return useMutation({
    mutationFn: (payload: PreSeasonPredictRequest) => api.predictPreSeason(payload),
  })
}

export function usePredictPreSeasonAdvanced() {
  return useMutation({
    mutationFn: (payload: PreSeasonAdvancedPredictRequest) => api.predictPreSeasonAdvanced(payload),
  })
}

export function usePredictRisk() {
  return useMutation({
    mutationFn: (payload: RiskAssessmentRequest) => api.predictRisk(payload),
  })
}

export function useExplainPrediction() {
  return useMutation({
    mutationFn: (payload: ExplainabilityRequest) => api.explainPrediction(payload),
  })
}

export function useDetectAnomaly() {
  return useMutation({
    mutationFn: (payload: AnomalyDetectionRequest) => api.detectAnomaly(payload),
  })
}

export function useIntelligenceDashboard() {
  return useQuery({
    queryKey: ['intelligence-dashboard'],
    queryFn: api.getIntelligenceDashboard,
    staleTime: 1000 * 60 * 10,
  })
}

export function useStateRisk() {
  return useQuery({
    queryKey: ['state-risk'],
    queryFn: api.getStateRisk,
    staleTime: 1000 * 60 * 10,
  })
}

export function useAnomalies(limit = 50) {
  return useQuery({
    queryKey: ['anomalies', limit],
    queryFn: () => api.getAnomalies(limit),
    staleTime: 1000 * 60 * 10,
  })
}

export function usePredictScenario() {
  return useMutation({
    mutationFn: (payload: ScenarioSimulationRequest) => api.predictScenario(payload),
  })
}

export function useCopilot() {
  return useMutation({
    mutationFn: (payload: CopilotQueryRequest) => api.queryCopilot(payload),
  })
}

export function useGenerateReport() {
  return useMutation({
    mutationFn: (payload: ReportGenerateRequest) => api.generateReport(payload),
  })
}

export function useDecisionSupport() {
  return useQuery({
    queryKey: ['decision-support'],
    queryFn: api.getDecisionSupport,
    staleTime: 1000 * 60 * 5,
  })
}

// Day 7 Temporal & Early Warning Hooks
export function useForecastYield() {
  return useMutation({
    mutationFn: (payload: ForecastYieldRequest) => api.forecastYield(payload),
  })
}

export function useForecastState(state: string) {
  return useQuery({
    queryKey: ['forecast-state', state],
    queryFn: () => api.getForecastState(state),
    enabled: Boolean(state),
    staleTime: 1000 * 60 * 10,
  })
}

export function useForecastDistrict(district: string, state?: string) {
  return useQuery({
    queryKey: ['forecast-district', district, state],
    queryFn: () => api.getForecastDistrict(district, state),
    enabled: Boolean(district),
    staleTime: 1000 * 60 * 10,
  })
}

export function useAnalyzeTrends() {
  return useMutation({
    mutationFn: (payload: { state?: string; district?: string }) => api.analyzeTrends(payload),
  })
}

export function useStatesTrends() {
  return useQuery({
    queryKey: ['states-trends'],
    queryFn: api.getStatesTrends,
    staleTime: 1000 * 60 * 15,
  })
}

export function useAssessEarlyWarning() {
  return useMutation({
    mutationFn: (payload: { state?: string; district?: string }) => api.assessEarlyWarning(payload),
  })
}

export function useEarlyWarningDashboard() {
  return useQuery({
    queryKey: ['early-warning-dashboard'],
    queryFn: api.getEarlyWarningDashboard,
    staleTime: 1000 * 60 * 5,
  })
}

export function useStatesEarlyWarning() {
  return useQuery({
    queryKey: ['states-early-warning'],
    queryFn: api.getStatesEarlyWarning,
    staleTime: 1000 * 60 * 10,
  })
}

// Day 8 Geospatial Hooks
export function useGeospatialOverview() {
  return useQuery({
    queryKey: ['geospatial-overview'],
    queryFn: api.getGeospatialOverview,
    staleTime: 1000 * 60 * 10,
  })
}

export function useGeospatialStates() {
  return useQuery({
    queryKey: ['geospatial-states'],
    queryFn: api.getGeospatialStates,
    staleTime: 1000 * 60 * 10,
  })
}

export function useGeospatialStateProfile(state: string) {
  return useQuery({
    queryKey: ['geospatial-state-profile', state],
    queryFn: () => api.getGeospatialStateProfile(state),
    enabled: Boolean(state),
    staleTime: 1000 * 60 * 10,
  })
}

export function useGeospatialDistrictProfile(district: string, state?: string) {
  return useQuery({
    queryKey: ['geospatial-district-profile', district, state],
    queryFn: () => api.getGeospatialDistrictProfile(district, state),
    enabled: Boolean(district),
    staleTime: 1000 * 60 * 10,
  })
}

export function useGeospatialClusters() {
  return useQuery({
    queryKey: ['geospatial-clusters'],
    queryFn: api.getGeospatialClusters,
    staleTime: 1000 * 60 * 20,
  })
}

export function useGeospatialQuery() {
  return useMutation({
    mutationFn: (payload: any) => api.queryGeospatial(payload),
  })
}

// Day 9 Model Reliability, Validation & Monitoring Hooks
export function useValidationOverview() {
  return useQuery({
    queryKey: ['validation-overview'],
    queryFn: api.getValidationOverview,
    staleTime: 1000 * 60 * 15,
  })
}

export function useValidationStates() {
  return useQuery({
    queryKey: ['validation-states'],
    queryFn: api.getValidationStates,
    staleTime: 1000 * 60 * 15,
  })
}

export function useValidationScatter() {
  return useQuery({
    queryKey: ['validation-scatter'],
    queryFn: api.getValidationScatter,
    staleTime: 1000 * 60 * 15,
  })
}

export function useErrorsSummary() {
  return useQuery({
    queryKey: ['errors-summary'],
    queryFn: api.getErrorsSummary,
    staleTime: 1000 * 60 * 15,
  })
}

export function useCalibrationSummary() {
  return useQuery({
    queryKey: ['calibration-summary'],
    queryFn: api.getCalibrationSummary,
    staleTime: 1000 * 60 * 15,
  })
}

export function useDriftOverview() {
  return useQuery({
    queryKey: ['drift-overview'],
    queryFn: api.getDriftOverview,
    staleTime: 1000 * 60 * 15,
  })
}

export function useDriftFeatures() {
  return useQuery({
    queryKey: ['drift-features'],
    queryFn: api.getDriftFeatures,
    staleTime: 1000 * 60 * 15,
  })
}

export function useDataQuality() {
  return useQuery({
    queryKey: ['data-quality'],
    queryFn: api.getDataQuality,
    staleTime: 1000 * 60 * 30,
  })
}

export function useModelRegistry() {
  return useQuery({
    queryKey: ['model-registry'],
    queryFn: api.getModelRegistry,
    staleTime: 1000 * 60 * 30,
  })
}

// Day 10 Scenario Intelligence Hooks
export function useSimulateScenario() {
  return useMutation({
    mutationFn: (payload: any) => api.simulateScenario(payload),
  })
}

export function useCompareScenarios() {
  return useMutation({
    mutationFn: (payload: any) => api.compareScenarios(payload),
  })
}

export function useAnalyzeSensitivity() {
  return useMutation({
    mutationFn: (payload: any) => api.analyzeSensitivity(payload),
  })
}

export function useOptimizeDecision() {
  return useMutation({
    mutationFn: (payload: any) => api.optimizeDecision(payload),
  })
}

export function useScenarioTemplates() {
  return useQuery({
    queryKey: ['scenario-templates'],
    queryFn: api.getScenarioTemplates,
    staleTime: 1000 * 60 * 60,
  })
}

export function useScenarioHistory(limit = 20) {
  return useQuery({
    queryKey: ['scenario-history', limit],
    queryFn: () => api.getScenarioHistory(limit),
    staleTime: 1000 * 60 * 2,
  })
}

export function useScenarioAudit(scenarioId: string) {
  return useQuery({
    queryKey: ['scenario-audit', scenarioId],
    queryFn: () => api.getScenarioAudit(scenarioId),
    enabled: Boolean(scenarioId),
    staleTime: 1000 * 60 * 30,
  })
}

// Day 12 Monitoring & Early Warning Hooks
export function useMonitoringOverview() {
  return useQuery({
    queryKey: ['monitoring-overview'],
    queryFn: api.getMonitoringOverview,
    staleTime: 1000 * 60 * 2,
  })
}

export function useMonitoringTimeline(state = 'Punjab', district?: string, metric = 'yield') {
  return useQuery({
    queryKey: ['monitoring-timeline', state, district, metric],
    queryFn: () => api.getMonitoringTimeline(state, district, metric),
    staleTime: 1000 * 60 * 5,
  })
}

export function useMonitoringStates() {
  return useQuery({
    queryKey: ['monitoring-states'],
    queryFn: api.getMonitoringStates,
    staleTime: 1000 * 60 * 10,
  })
}

export function useMonitoringDistricts(state?: string) {
  return useQuery({
    queryKey: ['monitoring-districts', state],
    queryFn: () => api.getMonitoringDistricts(state),
    staleTime: 1000 * 60 * 5,
  })
}

export function useMonitoringAlerts(params?: { state?: string; district?: string; severity?: string; signal_type?: string; year?: number; limit?: number }) {
  return useQuery({
    queryKey: ['monitoring-alerts', params],
    queryFn: () => api.getMonitoringAlerts(params),
    staleTime: 1000 * 60 * 2,
  })
}

export function useAlertById(alertId: string) {
  return useQuery({
    queryKey: ['monitoring-alert-detail', alertId],
    queryFn: () => api.getAlertById(alertId),
    enabled: Boolean(alertId),
    staleTime: 1000 * 60 * 10,
  })
}

export function useWarningMap() {
  return useQuery({
    queryKey: ['monitoring-warning-map'],
    queryFn: api.getWarningMap,
    staleTime: 1000 * 60 * 5,
  })
}

export function useMonitoringHealth() {
  return useQuery({
    queryKey: ['monitoring-health'],
    queryFn: api.getMonitoringHealth,
    staleTime: 1000 * 60 * 5,
  })
}

export function useRunWarningBacktest() {
  return useMutation({
    mutationFn: (payload: BacktestRequest) => api.runWarningBacktest(payload),
  })
}

// Day 13 Explainable AI (XAI) Hooks
export function useGlobalImportance() {
  return useQuery({
    queryKey: ['explainability-global-importance'],
    queryFn: api.getGlobalImportance,
    staleTime: 1000 * 60 * 30,
  })
}

export function useExplainPredictionXAI() {
  return useMutation({
    mutationFn: (payload: import('../types/explainability').LocalExplanationRequest) =>
      api.explainPredictionXAI(payload),
  })
}

export function useSensitivityAnalysis() {
  return useMutation({
    mutationFn: (payload: import('../types/explainability').SensitivityRequest) =>
      api.getSensitivityAnalysis(payload),
  })
}

export function useAlertExplanation(alertId: string) {
  return useQuery({
    queryKey: ['explainability-alert', alertId],
    queryFn: () => api.explainAlert(alertId),
    enabled: Boolean(alertId),
    staleTime: 1000 * 60 * 10,
  })
}

export function useScenarioExplanation() {
  return useMutation({
    mutationFn: (args: {
      scenarioId: string
      state?: string
      baseline_yield?: number
      simulated_yield?: number
      changed_features?: Record<string, number>
    }) =>
      api.explainScenario(args.scenarioId, {
        state: args.state,
        baseline_yield: args.baseline_yield,
        simulated_yield: args.simulated_yield,
        changed_features: args.changed_features,
      }),
  })
}

export function useExplanationAudit(explanationId: string) {
  return useQuery({
    queryKey: ['explainability-audit', explanationId],
    queryFn: () => api.getExplanationAudit(explanationId),
    enabled: Boolean(explanationId),
    staleTime: 1000 * 60 * 30,
  })
}

export function useExplainabilityValidation() {
  return useQuery({
    queryKey: ['explainability-validation'],
    queryFn: api.getExplainabilityValidation,
    staleTime: 1000 * 60 * 60,
  })
}

// Day 18: Multi-Crop Modeling Readiness & Baselines Hooks
export function useReadinessSummary() {
  return useQuery({
    queryKey: ['readiness-summary'],
    queryFn: api.getReadinessSummary,
    staleTime: 1000 * 60 * 30,
  })
}

export function useCropReadinessAll(status?: string) {
  return useQuery({
    queryKey: ['crop-readiness-all', status],
    queryFn: () => api.getCropReadinessAll(status),
    staleTime: 1000 * 60 * 30,
  })
}

export function useCropReadinessSingle(crop: string) {
  return useQuery({
    queryKey: ['crop-readiness-single', crop],
    queryFn: () => api.getCropReadinessSingle(crop),
    enabled: Boolean(crop),
    staleTime: 1000 * 60 * 30,
  })
}

export function useCropBaselines(crop: string) {
  return useQuery({
    queryKey: ['crop-baselines', crop],
    queryFn: () => api.getCropBaselines(crop),
    enabled: Boolean(crop),
    staleTime: 1000 * 60 * 30,
  })
}

export function useFeatureCompatibility() {
  return useQuery({
    queryKey: ['feature-compatibility'],
    queryFn: api.getFeatureCompatibility,
    staleTime: 1000 * 60 * 60,
  })
}

export function useArchitectureDecision() {
  return useQuery({
    queryKey: ['architecture-decision'],
    queryFn: api.getArchitectureDecision,
    staleTime: 1000 * 60 * 60,
  })
}

// Day 19: Multi-Crop Forecasting Hooks
export function useMultiCropModels(status?: string) {
  return useQuery({
    queryKey: ['multicrop-models', status],
    queryFn: () => api.getMultiCropModels(status),
    staleTime: 1000 * 60 * 30,
  })
}

export function useCropModelDetails(crop: string) {
  return useQuery({
    queryKey: ['multicrop-model-details', crop],
    queryFn: () => api.getCropModelDetails(crop),
    enabled: Boolean(crop),
    staleTime: 1000 * 60 * 30,
  })
}

export function useCropModelComparison(crop: string) {
  return useQuery({
    queryKey: ['multicrop-model-comparison', crop],
    queryFn: () => api.getCropModelComparison(crop),
    enabled: Boolean(crop),
    staleTime: 1000 * 60 * 30,
  })
}

export function useCropModelMetrics(crop: string) {
  return useQuery({
    queryKey: ['multicrop-model-metrics', crop],
    queryFn: () => api.getCropModelMetrics(crop),
    enabled: Boolean(crop),
    staleTime: 1000 * 60 * 30,
  })
}

export function useCropModelFeatures(crop: string) {
  return useQuery({
    queryKey: ['multicrop-model-features', crop],
    queryFn: () => api.getCropModelFeatures(crop),
    enabled: Boolean(crop),
    staleTime: 1000 * 60 * 30,
  })
}

export function useMultiCropLeaderboard() {
  return useQuery({
    queryKey: ['multicrop-leaderboard'],
    queryFn: api.getMultiCropLeaderboard,
    staleTime: 1000 * 60 * 30,
  })
}

// Day 20: Temporal Robustness Hooks
export function useCropRobustnessAll() {
  return useQuery({
    queryKey: ['multicrop-robustness-all'],
    queryFn: api.getCropRobustnessAll,
    staleTime: 1000 * 60 * 30,
  })
}

export function useRobustnessSummary() {
  return useQuery({
    queryKey: ['robustness-summary'],
    queryFn: api.getRobustnessSummary,
    staleTime: 1000 * 60 * 30,
  })
}

export function useCropRobustness(crop: string) {
  return useQuery({
    queryKey: ['crop-robustness', crop],
    queryFn: () => api.getCropRobustness(crop),
    enabled: Boolean(crop),
    staleTime: 1000 * 60 * 30,
  })
}

export function useCropFolds(crop: string) {
  return useQuery({
    queryKey: ['crop-folds', crop],
    queryFn: () => api.getCropFolds(crop),
    enabled: Boolean(crop),
    staleTime: 1000 * 60 * 30,
  })
}

export function useCropRobustnessDetail(crop: string) {
  return useQuery({
    queryKey: ['crop-robustness-detail', crop],
    queryFn: () => api.getCropRobustnessDetail(crop),
    enabled: Boolean(crop),
    staleTime: 1000 * 60 * 30,
  })
}

// Day 21: Model Diagnosis & Strategy Hooks
export function useCropDiagnosisAll() {
  return useQuery({
    queryKey: ['crop-diagnosis-all'],
    queryFn: api.getCropDiagnosisAll,
    staleTime: 1000 * 60 * 30,
  })
}

export function useCropDiagnosis(crop: string) {
  return useQuery({
    queryKey: ['crop-diagnosis', crop],
    queryFn: () => api.getCropDiagnosis(crop),
    enabled: Boolean(crop),
    staleTime: 1000 * 60 * 30,
  })
}

export function useCropErrorRegimes(crop: string) {
  return useQuery({
    queryKey: ['crop-error-regimes', crop],
    queryFn: () => api.getCropErrorRegimes(crop),
    enabled: Boolean(crop),
    staleTime: 1000 * 60 * 30,
  })
}

export function useCropDistrictErrors(crop: string) {
  return useQuery({
    queryKey: ['crop-district-errors', crop],
    queryFn: () => api.getCropDistrictErrors(crop),
    enabled: Boolean(crop),
    staleTime: 1000 * 60 * 30,
  })
}

export function useCropYearErrors(crop: string) {
  return useQuery({
    queryKey: ['crop-year-errors', crop],
    queryFn: () => api.getCropYearErrors(crop),
    enabled: Boolean(crop),
    staleTime: 1000 * 60 * 30,
  })
}

export function useCropFeatureStability(crop: string) {
  return useQuery({
    queryKey: ['crop-feature-stability', crop],
    queryFn: () => api.getCropFeatureStability(crop),
    enabled: Boolean(crop),
    staleTime: 1000 * 60 * 30,
  })
}

export function useCropModelSelectionAll() {
  return useQuery({
    queryKey: ['crop-model-selection-all'],
    queryFn: api.getCropModelSelectionAll,
    staleTime: 1000 * 60 * 30,
  })
}

export function useCropModelSelection(crop: string) {
  return useQuery({
    queryKey: ['crop-model-selection', crop],
    queryFn: () => api.getCropModelSelection(crop),
    enabled: Boolean(crop),
    staleTime: 1000 * 60 * 30,
  })
}

export function useCropForecastingStrategyAll() {
  return useQuery({
    queryKey: ['crop-forecasting-strategy-all'],
    queryFn: api.getCropForecastingStrategyAll,
    staleTime: 1000 * 60 * 30,
  })
}

export function useCropForecastingStrategy(crop: string) {
  return useQuery({
    queryKey: ['crop-forecasting-strategy', crop],
    queryFn: () => api.getCropForecastingStrategy(crop),
    enabled: Boolean(crop),
    staleTime: 1000 * 60 * 30,
  })
}

// ---------------------------------------------------------------------------
// Day 22 Exogenous React Query Hooks
// ---------------------------------------------------------------------------

export function useExogenousSummary() {
  return useQuery({
    queryKey: ['exogenous-summary'],
    queryFn: api.getExogenousSummary,
    staleTime: 1000 * 60 * 30,
  })
}

export function useExogenousSources() {
  return useQuery({
    queryKey: ['exogenous-sources'],
    queryFn: api.getExogenousSources,
    staleTime: 1000 * 60 * 30,
  })
}

export function useExogenousCoverage() {
  return useQuery({
    queryKey: ['exogenous-coverage'],
    queryFn: api.getExogenousCoverage,
    staleTime: 1000 * 60 * 30,
  })
}

export function useExogenousFeatures() {
  return useQuery({
    queryKey: ['exogenous-features'],
    queryFn: api.getExogenousFeatures,
    staleTime: 1000 * 60 * 30,
  })
}

export function useExogenousAblation() {
  return useQuery({
    queryKey: ['exogenous-ablation'],
    queryFn: api.getExogenousAblation,
    staleTime: 1000 * 60 * 30,
  })
}

export function useExogenousSelectionAll() {
  return useQuery({
    queryKey: ['exogenous-selection-all'],
    queryFn: api.getExogenousSelectionAll,
    staleTime: 1000 * 60 * 30,
  })
}

export function useExogenousCropResult(crop: string) {
  return useQuery({
    queryKey: ['exogenous-crop-result', crop],
    queryFn: () => api.getExogenousCropResult(crop),
    enabled: Boolean(crop),
    staleTime: 1000 * 60 * 30,
  })
}

export function useExogenousCropFolds(crop: string) {
  return useQuery({
    queryKey: ['exogenous-crop-folds', crop],
    queryFn: () => api.getExogenousCropFolds(crop),
    enabled: Boolean(crop),
    staleTime: 1000 * 60 * 30,
  })
}

// Day 23 Custom Hooks
export function useFinalValidationSummary() {
  return useQuery({
    queryKey: ['final-validation-summary'],
    queryFn: api.getFinalValidationSummary,
    staleTime: 1000 * 60 * 30,
  })
}

export function useSingleCropFinalValidation(crop: string) {
  return useQuery({
    queryKey: ['single-crop-final-validation', crop],
    queryFn: () => api.getSingleCropFinalValidation(crop),
    enabled: Boolean(crop),
    staleTime: 1000 * 60 * 30,
  })
}

export function useCropResidualDiagnostics(crop: string) {
  return useQuery({
    queryKey: ['crop-residual-diagnostics', crop],
    queryFn: () => api.getCropResidualDiagnostics(crop),
    enabled: Boolean(crop),
    staleTime: 1000 * 60 * 30,
  })
}

export function useCropPredictionBias(crop: string) {
  return useQuery({
    queryKey: ['crop-prediction-bias', crop],
    queryFn: () => api.getCropPredictionBias(crop),
    enabled: Boolean(crop),
    staleTime: 1000 * 60 * 30,
  })
}

export function useCropFinalStrategy(crop: string) {
  return useQuery({
    queryKey: ['crop-final-strategy', crop],
    queryFn: () => api.getCropFinalStrategy(crop),
    enabled: Boolean(crop),
    staleTime: 1000 * 60 * 30,
  })
}

export function useReproducibilityAudit() {
  return useQuery({
    queryKey: ['reproducibility-audit'],
    queryFn: api.getReproducibilityAudit,
    staleTime: 1000 * 60 * 30,
  })
}

export function useFinalCertification() {
  return useQuery({
    queryKey: ['final-model-certification'],
    queryFn: api.getFinalCertification,
    staleTime: 1000 * 60 * 30,
  })
}

// Day 24 Production Forecast Serving & Governance Hooks
export function useForecastStrategies() {
  return useQuery({
    queryKey: ['forecast-strategies'],
    queryFn: api.getForecastStrategies,
    staleTime: 1000 * 60 * 30,
  })
}

export function useForecastCertificationSummary() {
  return useQuery({
    queryKey: ['forecast-certification-summary'],
    queryFn: api.getForecastCertificationSummary,
    staleTime: 1000 * 60 * 30,
  })
}

export function useForecastCoverage() {
  return useQuery({
    queryKey: ['forecast-coverage'],
    queryFn: api.getForecastCoverage,
    staleTime: 1000 * 60 * 30,
  })
}

export function useForecastPredict() {
  return useMutation({
    mutationFn: (payload: ForecastPredictRequest) => api.predictForecast(payload),
  })
}

export function useForecastProvenance(requestId: string) {
  return useQuery({
    queryKey: ['forecast-provenance', requestId],
    queryFn: () => api.getForecastProvenance(requestId),
    enabled: Boolean(requestId),
    staleTime: 1000 * 60 * 30,
  })
}

export function useForecastAuditLogs(limit = 50) {
  return useQuery({
    queryKey: ['forecast-audit-logs', limit],
    queryFn: () => api.getForecastAuditLogs(limit),
    refetchInterval: 10000,
    staleTime: 1000 * 5,
  })
}

export function useForecastHealth() {
  return useQuery({
    queryKey: ['forecast-health'],
    queryFn: api.getForecastHealth,
    staleTime: 1000 * 60 * 5,
  })
}

// Day 28 Production Observability & Operational Intelligence Hooks
export function useObservabilitySummary() {
  return useQuery({
    queryKey: ['observability-summary'],
    queryFn: api.getObservabilitySummary,
    refetchInterval: 5000,
    staleTime: 1000 * 3,
  })
}

export function useObservabilityHealth() {
  return useQuery({
    queryKey: ['observability-health'],
    queryFn: api.getObservabilityHealth,
    refetchInterval: 5000,
    staleTime: 1000 * 3,
  })
}

export function useObservabilityMetrics() {
  return useQuery({
    queryKey: ['observability-metrics'],
    queryFn: api.getObservabilityMetrics,
    refetchInterval: 5000,
    staleTime: 1000 * 3,
  })
}

export function useObservabilityForecasts() {
  return useQuery({
    queryKey: ['observability-forecasts'],
    queryFn: api.getObservabilityForecasts,
    refetchInterval: 10000,
    staleTime: 1000 * 5,
  })
}

export function useObservabilityStrategies() {
  return useQuery({
    queryKey: ['observability-strategies'],
    queryFn: api.getObservabilityStrategies,
    refetchInterval: 10000,
    staleTime: 1000 * 5,
  })
}

export function useObservabilityModels() {
  return useQuery({
    queryKey: ['observability-models'],
    queryFn: api.getObservabilityModels,
    staleTime: 1000 * 30,
  })
}

export function useObservabilityDataset() {
  return useQuery({
    queryKey: ['observability-dataset'],
    queryFn: api.getObservabilityDataset,
    staleTime: 1000 * 60,
  })
}

export function useObservabilityRegistry() {
  return useQuery({
    queryKey: ['observability-registry'],
    queryFn: api.getObservabilityRegistry,
    staleTime: 1000 * 60,
  })
}

export function usePredictionTrace(requestId: string) {
  return useQuery({
    queryKey: ['prediction-trace', requestId],
    queryFn: () => api.getPredictionTrace(requestId),
    enabled: Boolean(requestId && requestId.trim().length > 0),
    staleTime: 1000 * 60 * 5,
  })
}

export function useObservabilityErrors(limit = 50) {
  return useQuery({
    queryKey: ['observability-errors', limit],
    queryFn: () => api.getObservabilityErrors(limit),
    refetchInterval: 10000,
    staleTime: 1000 * 5,
  })
}

export function useObservabilityAlerts() {
  return useQuery({
    queryKey: ['observability-alerts'],
    queryFn: api.getObservabilityAlerts,
    refetchInterval: 5000,
    staleTime: 1000 * 3,
  })
}

export function useObservabilityDrift() {
  return useQuery({
    queryKey: ['observability-drift'],
    queryFn: api.getObservabilityDrift,
    staleTime: 1000 * 60 * 5,
  })
}







