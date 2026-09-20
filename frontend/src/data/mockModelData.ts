import { ModelLeaderboardEntry, FeatureImportanceItem } from '../types/model'

export const modelLeaderboard: ModelLeaderboardEntry[] = [
  {
    id: 'det-baseline',
    modelName: 'Deterministic Agricultural Baseline',
    modelType: 'deterministic',
    featureSet: 'Production / Area × 1000 (Exact Ratio)',
    randomR2: 0.9893,
    randomMae: 6.56,
    randomRmse: 114.58,
    temporalR2: 0.9771,
    temporalMae: 16.68,
    temporalRmse: 168.15,
    groupKFoldR2: 0.9942,
    groupKFoldMae: 4.21,
    groupKFoldRmse: 84.13,
    status: 'baseline',
    notes: 'Non-ML post-harvest mathematical identity (Production / Area × 1000). Algebraic reconstruction of reported yield, not an out-of-sample forecast.'
  },
  {
    id: 'hist-gb',
    modelName: 'HistGradientBoosting Regressor',
    modelType: 'ml',
    featureSet: 'Year, State, Area, Production',
    randomR2: 0.9697,
    randomMae: 93.57,
    randomRmse: 192.87,
    temporalR2: 0.9533,
    temporalMae: 119.05,
    temporalRmse: 240.02,
    groupKFoldR2: 0.8394,
    groupKFoldMae: 251.71,
    groupKFoldRmse: 377.69,
    status: 'production',
    notes: 'Top performing ML estimator. Fast histogram-based gradient boosting.'
  },
  {
    id: 'extra-trees',
    modelName: 'ExtraTrees Regressor',
    modelType: 'ml',
    featureSet: 'Year, State, Area, Production',
    randomR2: 0.9661,
    randomMae: 76.68,
    randomRmse: 203.86,
    temporalR2: 0.9526,
    temporalMae: 101.68,
    temporalRmse: 241.89,
    groupKFoldR2: 0.7519,
    groupKFoldMae: 330.77,
    groupKFoldRmse: 483.24,
    status: 'benchmark',
    notes: 'Extremely randomized trees with lowest random holdout MAE among ML models.'
  },
  {
    id: 'rf-tuned',
    modelName: 'RandomForest (Tuned)',
    modelType: 'ml',
    featureSet: 'Year, State, Area, Production',
    randomR2: 0.9570,
    randomMae: 96.27,
    randomRmse: 229.75,
    temporalR2: 0.9395,
    temporalMae: 122.82,
    temporalRmse: 273.16,
    groupKFoldR2: 0.8024,
    groupKFoldMae: 230.48,
    groupKFoldRmse: 423.40,
    status: 'benchmark',
    notes: 'Tuned Random Forest with 150 estimators, max_depth=30.'
  },
  {
    id: 'rf-baseline',
    modelName: 'RandomForest (Baseline)',
    modelType: 'ml',
    featureSet: 'Year, State, Area, Production',
    randomR2: 0.9568,
    randomMae: 97.00,
    randomRmse: 230.26,
    temporalR2: 0.9400,
    temporalMae: 122.88,
    temporalRmse: 271.98,
    groupKFoldR2: 0.7940,
    groupKFoldMae: 236.44,
    groupKFoldRmse: 434.82,
    status: 'benchmark',
    notes: 'Default scikit-learn random forest configuration.'
  },
  {
    id: 'gb-regressor',
    modelName: 'GradientBoosting Regressor',
    modelType: 'ml',
    featureSet: 'Year, State, Area, Production',
    randomR2: 0.9612,
    randomMae: 106.26,
    randomRmse: 218.28,
    temporalR2: 0.9445,
    temporalMae: 130.92,
    temporalRmse: 261.73,
    groupKFoldR2: 0.8474,
    groupKFoldMae: 239.93,
    groupKFoldRmse: 371.85,
    status: 'benchmark',
    notes: 'Standard sequential boosting trees.'
  },
  {
    id: 'svr-rbf',
    modelName: 'Support Vector Regressor (SVR)',
    modelType: 'ml',
    featureSet: 'Year, State, Area, Production',
    randomR2: 0.6405,
    randomMae: 400.19,
    randomRmse: 663.97,
    temporalR2: 0.6096,
    temporalMae: 487.03,
    temporalRmse: 693.96,
    groupKFoldR2: 0.5288,
    groupKFoldMae: 454.80,
    groupKFoldRmse: 676.93,
    status: 'benchmark',
    notes: 'RBF kernel SVR. Non-linear ratio approximation is difficult for RBF distance metric.'
  }
]

export const featureImportances: FeatureImportanceItem[] = [
  {
    feature: 'RICE PRODUCTION (1000 tons)',
    nativeMdi: 52.4,
    permutationDeltaR2: 0.864,
    description: 'Direct numerator in mathematical yield formulation. Strongest single predictor.'
  },
  {
    feature: 'RICE AREA (1000 ha)',
    nativeMdi: 43.1,
    permutationDeltaR2: 0.802,
    description: 'Direct denominator in mathematical yield formulation. Scale normalizing variable.'
  },
  {
    feature: 'State Code',
    nativeMdi: 3.1,
    permutationDeltaR2: 0.041,
    description: 'Regional geographic identifier capturing agro-climatic baseline differences.'
  },
  {
    feature: 'Year',
    nativeMdi: 1.4,
    permutationDeltaR2: 0.018,
    description: 'Temporal variable capturing technological advancements and secular yield trends.'
  }
]

export const ablationSummary = [
  {
    featureSet: 'Full (Area + Prod + State + Year)',
    numFeatures: 4,
    randomR2: 0.9570,
    randomMae: 96.27,
    temporalR2: 0.9395,
    temporalMae: 122.82,
    groupKFoldR2: 0.8024,
    interpretation: 'Approximates mathematical ratio P/A with high precision.'
  },
  {
    featureSet: 'No Production (Area + State + Year)',
    numFeatures: 3,
    randomR2: 0.7412,
    randomMae: 345.81,
    temporalR2: 0.6479,
    temporalMae: 468.03,
    groupKFoldR2: -0.0038,
    interpretation: 'True pre-season forecasting. Error increases 3.6x; cannot generalize to unseen states.'
  },
  {
    featureSet: 'No Area (Prod + State + Year)',
    numFeatures: 3,
    randomR2: 0.7674,
    randomMae: 335.21,
    temporalR2: 0.7243,
    temporalMae: 400.71,
    groupKFoldR2: 0.1384,
    interpretation: 'Moderate correlation with overall district production volume.'
  },
  {
    featureSet: 'State + Year only',
    numFeatures: 2,
    randomR2: 0.4971,
    randomMae: 581.78,
    temporalR2: 0.4293,
    temporalMae: 672.34,
    groupKFoldR2: -0.9263,
    interpretation: 'Coarse regional historical baseline with poor predictive fidelity.'
  }
]
