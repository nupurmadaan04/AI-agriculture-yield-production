import { ReportItem } from '../types/reports'

export const reportService = {
  getReports: async (): Promise<ReportItem[]> => {
    return Promise.resolve([
      {
        id: 'rep-scientific-validation',
        title: 'Scientific Validation & Model Integrity Review',
        category: 'scientific',
        summary: 'Rigorous empirical analysis of mathematical target leakage, deterministic baseline superiority, and pre-season ablation.',
        generatedDate: '2026-08-30',
        fileFormat: 'MD',
        status: 'available',
        fileSize: '16.8 KB'
      },
      {
        id: 'rep-project-audit',
        title: 'Complete Repository & Pipeline Audit',
        category: 'dataset',
        summary: 'Baseline reproduction, metric correction (MSE vs RMSE), dataset restoration, and code quality assessment.',
        generatedDate: '2026-08-30',
        fileFormat: 'MD',
        status: 'available',
        fileSize: '24.2 KB'
      },
      {
        id: 'rep-model-comparison',
        title: 'Cross-Model Multi-Split Benchmark Report',
        category: 'benchmark',
        summary: 'Comprehensive evaluation of 6 estimators across Random, Temporal, and GroupKFold cross-validation.',
        generatedDate: '2026-08-30',
        fileFormat: 'CSV',
        status: 'available',
        fileSize: '1.2 KB'
      },
      {
        id: 'rep-state-error-analysis',
        title: 'State & District-Level Error Breakdown',
        category: 'benchmark',
        summary: 'Detailed residual distributions, best/worst performing states, and extreme outlier diagnosis.',
        generatedDate: '2026-08-30',
        fileFormat: 'CSV',
        status: 'available',
        fileSize: '3.4 KB'
      },
      {
        id: 'rep-pre-season-feasibility',
        title: 'Pre-Season Forecasting Feasibility Study',
        category: 'scientific',
        summary: 'Exploration of weather, soil, and satellite data requirements for authentic pre-harvest prediction.',
        generatedDate: '2026-09-01',
        fileFormat: 'PDF',
        status: 'coming-soon',
        fileSize: 'TBD'
      }
    ])
  }
}
