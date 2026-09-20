import React from 'react'
import {
  useValidationOverview,
  useValidationStates,
  useValidationScatter,
  useErrorsSummary,
  useCalibrationSummary,
  useDriftOverview,
  useDataQuality,
  useModelRegistry
} from '../services/api'
import { ModelScoreCard } from '../components/validation/ModelScoreCard'
import { ModelComparisonTable } from '../components/validation/ModelComparisonTable'
import { ErrorDistributionChart } from '../components/validation/ErrorDistributionChart'
import { PredictionComparisonChart } from '../components/validation/PredictionComparisonChart'
import { RegionalPerformanceTable } from '../components/validation/RegionalPerformanceTable'
import { CalibrationPanel } from '../components/validation/CalibrationPanel'
import { DriftMonitor } from '../components/validation/DriftMonitor'
import { DataQualityCard } from '../components/validation/DataQualityCard'
import { ModelRegistryTable } from '../components/validation/ModelRegistryTable'
import { ShieldCheck, Loader2, Sparkles, RefreshCw } from 'lucide-react'
import { Button } from '../components/ui/Button'

export const ModelReliability: React.FC = () => {
  const { data: validation, isLoading: loadingVal, refetch: refetchVal } = useValidationOverview()
  const { data: statesRes, isLoading: loadingStates } = useValidationStates()
  const { data: scatterPoints, isLoading: loadingScatter } = useValidationScatter()
  const { data: errors, isLoading: loadingErrors } = useErrorsSummary()
  const { data: calibration, isLoading: loadingCal } = useCalibrationSummary()
  const { data: drift, isLoading: loadingDrift } = useDriftOverview()
  const { data: dataQuality, isLoading: loadingQuality } = useDataQuality()
  const { data: registry, isLoading: loadingReg } = useModelRegistry()

  const isLoading = loadingVal || loadingStates || loadingScatter || loadingErrors || loadingCal || loadingDrift || loadingQuality || loadingReg

  const handleRefresh = () => {
    refetchVal()
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold tracking-tight text-foreground">Model Reliability & Validation Monitor</h1>
            <span className="flex items-center gap-1 text-[11px] font-semibold text-emerald-600 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
              <ShieldCheck className="w-3.5 h-3.5" />
              Audited & Calibrated
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            Continuous out-of-time validation (2016–2017), Population Stability Index (PSI) drift tracking, and dataset integrity governance
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={handleRefresh} className="text-xs h-8">
            <RefreshCw className={`w-3.5 h-3.5 mr-1.5 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh Diagnostics
          </Button>
        </div>
      </div>

      {isLoading && !validation ? (
        <div className="flex items-center justify-center p-16 text-muted-foreground gap-2">
          <Loader2 className="w-6 h-6 animate-spin text-primary" />
          <span className="text-sm">Calculating out-of-time validation metrics and drift distributions...</span>
        </div>
      ) : (
        <>
          {/* Top KPI row */}
          <ModelScoreCard validation={validation} dataQuality={dataQuality} drift={drift} />

          {/* Model Benchmarks */}
          {validation?.benchmark_comparison && (
            <ModelComparisonTable benchmarks={validation.benchmark_comparison} />
          )}

          {/* Charts Row: Observed vs Predicted & Residuals */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <PredictionComparisonChart scatterPoints={scatterPoints || []} />
            <ErrorDistributionChart errorSummary={errors} />
          </div>

          {/* Regional Slices */}
          {statesRes?.data && (
            <RegionalPerformanceTable states={statesRes.data} />
          )}

          {/* Calibration & Spread Analysis */}
          <CalibrationPanel calibration={calibration} />

          {/* Feature Drift Monitor */}
          <DriftMonitor drift={drift} />

          {/* 4-Pillar Data Quality Card */}
          <DataQualityCard dataQuality={dataQuality} />

          {/* Model Registry & Governance */}
          {registry?.models && (
            <ModelRegistryTable models={registry.models} />
          )}
        </>
      )}
    </div>
  )
}
export default ModelReliability
