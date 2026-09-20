import React, { useState } from 'react'
import { Sliders, AlertTriangle, RefreshCw, Layers } from 'lucide-react'
import { SensitivityResponse } from '../../types/explainability'
import { useSensitivityAnalysis } from '../../services/api'

interface SensitivityAnalysisProps {
  states?: string[]
}

export const SensitivityAnalysis: React.FC<SensitivityAnalysisProps> = ({
  states = ['Punjab', 'Haryana', 'Andhra Pradesh', 'Uttar Pradesh', 'West Bengal', 'Tamil Nadu']
}) => {
  const [selectedState, setSelectedState] = useState<string>('Punjab')
  const [activeFeature, setActiveFeature] = useState<string>('RICE_YIELD_LAG1')
  const [data, setData] = useState<SensitivityResponse | null>(null)

  const sensitivityMutation = useSensitivityAnalysis()

  const handleRun = () => {
    sensitivityMutation.mutate(
      {
        state: selectedState,
        target_features: ['RICE_YIELD_LAG1', 'RICE AREA (1000 ha)', 'RICE_AREA_SHARE', 'TOTAL_CROPPED_AREA']
      },
      {
        onSuccess: (res) => {
          setData(res)
          if (res.tested_features.length > 0 && !res.tested_features.includes(activeFeature)) {
            setActiveFeature(res.tested_features[0])
          }
        }
      }
    )
  }

  React.useEffect(() => {
    handleRun()
  }, [])

  const currentCurve = data?.sensitivity_curves[activeFeature] || []

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 mb-6">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4 mb-6">
        <div>
          <h3 className="text-base font-semibold text-white flex items-center gap-2">
            <Sliders className="w-5 h-5 text-emerald-400" />
            Controlled Feature Sensitivity Analysis
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Evaluate how the registered model responds to systematic input perturbations ([-10%, +10%])
          </p>
        </div>

        <div className="flex items-center gap-3">
          <select
            value={selectedState}
            onChange={(e) => setSelectedState(e.target.value)}
            className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-emerald-500"
          >
            {states.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>

          <button
            onClick={handleRun}
            disabled={sensitivityMutation.isPending}
            className="px-3.5 py-1.5 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-semibold rounded-lg text-xs flex items-center gap-1.5 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${sensitivityMutation.isPending ? 'animate-spin' : ''}`} />
            Run Sweep
          </button>
        </div>
      </div>

      {data && (
        <div>
          {/* Feature Selector Tabs */}
          <div className="flex flex-wrap gap-2 mb-6">
            {data.tested_features.map((feat) => (
              <button
                key={feat}
                onClick={() => setActiveFeature(feat)}
                className={`px-3 py-1.5 text-xs rounded-lg font-medium transition-colors ${
                  activeFeature === feat
                    ? 'bg-emerald-500 text-slate-950 font-semibold shadow-sm'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700'
                }`}
              >
                {feat}
              </button>
            ))}
          </div>

          {/* Curve Visualization Table & Delta Chart */}
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 mb-6">
            {/* Perturbation Grid */}
            <div className="lg:col-span-3 bg-slate-800/40 p-4 rounded-xl border border-slate-800">
              <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3">
                Perturbation Response Curve: <span className="text-emerald-400 font-mono">{activeFeature}</span>
              </h4>

              <div className="space-y-2.5">
                {currentCurve.map((point) => {
                  const isBaseline = point.step_pct === 0
                  const isPosDelta = point.prediction_delta_kg_ha >= 0

                  return (
                    <div
                      key={point.step_pct}
                      className={`p-2.5 rounded-lg border text-xs flex items-center justify-between gap-3 ${
                        isBaseline
                          ? 'bg-emerald-500/10 border-emerald-500/30'
                          : 'bg-slate-900/60 border-slate-800'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span
                          className={`font-mono font-bold w-12 ${
                            point.step_pct > 0
                              ? 'text-emerald-400'
                              : point.step_pct < 0
                              ? 'text-rose-400'
                              : 'text-slate-200'
                          }`}
                        >
                          {point.step_pct > 0 ? `+${point.step_pct}%` : `${point.step_pct}%`}
                        </span>
                        <span className="text-slate-400">
                          Val: <strong className="text-slate-200">{point.perturbed_value.toLocaleString()}</strong>
                        </span>
                      </div>

                      <div className="flex items-center gap-3">
                        <span className="text-slate-200 font-medium">
                          {point.predicted_yield_kg_ha.toLocaleString()} kg/ha
                        </span>
                        <span
                          className={`font-mono font-semibold w-24 text-right ${
                            isPosDelta ? 'text-emerald-400' : 'text-rose-400'
                          }`}
                        >
                          {isPosDelta ? '+' : ''}
                          {point.prediction_delta_kg_ha.toFixed(1)} ({point.relative_delta_pct > 0 ? '+' : ''}
                          {point.relative_delta_pct}%)
                        </span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Metric Overview Card */}
            <div className="lg:col-span-2 bg-slate-800/40 p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
              <div>
                <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
                  Baseline Model Prediction
                </h4>
                <p className="text-2xl font-bold text-white mb-4">
                  {data.base_prediction_kg_ha.toLocaleString()}{' '}
                  <span className="text-xs font-normal text-slate-400">kg/ha</span>
                </p>

                <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
                  Elasticity Summary
                </h4>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Modifying <span className="text-emerald-400 font-semibold">{activeFeature}</span> by ±10% shifts the model
                  prediction across a range of{' '}
                  <span className="text-white font-semibold">
                    {Math.min(...currentCurve.map((c) => c.predicted_yield_kg_ha)).toFixed(0)} –{' '}
                    {Math.max(...currentCurve.map((c) => c.predicted_yield_kg_ha)).toFixed(0)} kg/ha
                  </span>
                  .
                </p>
              </div>

              <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg text-[11px] text-amber-300/90 flex items-start gap-2 mt-4">
                <AlertTriangle className="w-4 h-4 shrink-0 text-amber-400 mt-0.5" />
                <span>
                  <strong>Non-Causal Note:</strong> Sensitivity curves quantify mathematical gradient responses of the Random Forest
                  regressor; they do not imply agronomic physical causation.
                </span>
              </div>
            </div>
          </div>

          <p className="text-[11px] text-slate-500 italic">{data.scientific_disclaimer}</p>
        </div>
      )}
    </div>
  )
}
