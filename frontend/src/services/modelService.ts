import { modelLeaderboard, featureImportances, ablationSummary } from '../data/mockModelData'
import { ModelLeaderboardEntry, FeatureImportanceItem, EstimationParams, EstimationResult } from '../types/model'

export const modelService = {
  getLeaderboard: async (): Promise<ModelLeaderboardEntry[]> => {
    return Promise.resolve(modelLeaderboard)
  },

  getFeatureImportances: async (): Promise<FeatureImportanceItem[]> => {
    return Promise.resolve(featureImportances)
  },

  getAblationSummary: async () => {
    return Promise.resolve(ablationSummary)
  },

  estimateYield: async (params: EstimationParams): Promise<EstimationResult> => {
    if (params.mode === 'post-harvest') {
      const area = params.area || 0
      const prod = params.production || 0
      const detYield = area > 0 ? (prod / area) * 1000 : 0
      // ML estimate slightly approximates deterministic
      return Promise.resolve({
        mode: 'post-harvest',
        estimatedYield: Math.round(detYield * 10) / 10,
        deterministicYield: Math.round(detYield * 100) / 100,
        confidenceRange: [Math.max(0, detYield - 96), detYield + 96],
        method: 'Post-Harvest ML Reconstruction (HistGradientBoosting / RandomForest) + Exact Ratio Baseline',
        statusMessage: 'Yield estimated from reported cultivated area and production statistics.'
      })
    } else {
      // Pre-season estimation placeholder
      return Promise.resolve({
        mode: 'pre-season',
        estimatedYield: 2150.0,
        confidenceRange: [1800.0, 2500.0],
        method: 'Pre-Season Agro-Climatic Model (Experimental / Coming Soon)',
        statusMessage: 'Pre-season forecasting model requires additional weather and soil datasets.',
        requiresAdditionalData: true
      })
    }
  }
}
