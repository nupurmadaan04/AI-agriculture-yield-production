import React, { useState } from 'react'
import { Sparkles, HelpCircle, ArrowRight, CheckCircle, TrendingUp, TrendingDown, RefreshCw } from 'lucide-react'
import { LocalExplanationResponse } from '../../types/explainability'
import { FeatureContributionChart } from './FeatureContributionChart'
import { useExplainPredictionXAI } from '../../services/api'

interface PredictionExplanationProps {
  states?: string[]
}

export const PredictionExplanation: React.FC<PredictionExplanationProps> = ({
  states = ['Punjab', 'Haryana', 'Andhra Pradesh', 'Uttar Pradesh', 'West Bengal', 'Tamil Nadu', 'Odisha', 'Bihar']
}) => {
  const [selectedState, setSelectedState] = useState<string>('Punjab')
  const [district, setDistrict] = useState<string>('')
  const [area, setArea] = useState<number>(310.0)
  const [year, setYear] = useState<number>(2017)

  const [explanation, setExplanation] = useState<LocalExplanationResponse | null>(null)
  const explainMutation = useExplainPredictionXAI()

  const handleExplain = () => {
    explainMutation.mutate(
      {
        state: selectedState,
        district: district || undefined,
        area_1000_ha: area,
        year: year
      },
      {
        onSuccess: (data) => {
          setExplanation(data)
        }
      }
    )
  }

  // Run on mount
  React.useEffect(() => {
    handleExplain()
  }, [])

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 mb-6">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4 mb-6">
        <div>
          <h3 className="text-base font-semibold text-white flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-emerald-400" />
            Local Prediction Attribution & Driver Deconstruction
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Decompose any individual prediction into directional feature contributions against empirical dataset medians
          </p>
        </div>

        <button
          onClick={handleExplain}
          disabled={explainMutation.isPending}
          className="px-4 py-2 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-semibold rounded-lg text-xs flex items-center gap-2 transition-colors disabled:opacity-50"
        >
          {explainMutation.isPending ? (
            <>
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              Attributing...
            </>
          ) : (
            <>
              <RefreshCw className="w-3.5 h-3.5" />
              Re-Calculate Attribution
            </>
          )}
        </button>
      </div>

      {/* Query Controls */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 p-4 bg-slate-800/40 rounded-lg border border-slate-800 mb-6">
        <div>
          <label className="block text-xs font-medium text-slate-300 mb-1.5">State</label>
          <select
            value={selectedState}
            onChange={(e) => setSelectedState(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-emerald-500"
          >
            {states.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-300 mb-1.5">District (Optional)</label>
          <input
            type="text"
            placeholder="e.g. Ludhiana"
            value={district}
            onChange={(e) => setDistrict(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-300 mb-1.5">Rice Area (1000 ha)</label>
          <input
            type="number"
            min={1}
            max={5000}
            value={area}
            onChange={(e) => setArea(Number(e.target.value))}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-emerald-500"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-300 mb-1.5">Survey Year</label>
          <input
            type="number"
            min={1966}
            max={2030}
            value={year}
            onChange={(e) => setYear(Number(e.target.value))}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-emerald-500"
          />
        </div>
      </div>

      {/* Attribution Result */}
      {explanation && (
        <div>
          {/* Top Level Summary Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
            <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-800">
              <p className="text-xs text-slate-400 mb-1">Model Prediction</p>
              <p className="text-xl font-bold text-white">
                {explanation.prediction_kg_ha.toLocaleString()} <span className="text-xs font-normal text-slate-400">kg/ha</span>
              </p>
              <p className="text-[11px] text-slate-400 mt-1">
                Entity: <span className="text-slate-200 font-medium">{explanation.entity}</span>
              </p>
            </div>

            <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-800">
              <p className="text-xs text-slate-400 mb-1">Median Dataset Baseline</p>
              <p className="text-xl font-bold text-slate-300">
                {explanation.baseline_reference_kg_ha.toLocaleString()} <span className="text-xs font-normal text-slate-400">kg/ha</span>
              </p>
              <p className="text-[11px] text-slate-400 mt-1">Empirical Reference Vector</p>
            </div>

            <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-800">
              <p className="text-xs text-slate-400 mb-1">Attributed Prediction Delta</p>
              <p
                className={`text-xl font-bold ${
                  explanation.prediction_delta_kg_ha >= 0 ? 'text-emerald-400' : 'text-rose-400'
                }`}
              >
                {explanation.prediction_delta_kg_ha >= 0 ? '+' : ''}
                {explanation.prediction_delta_kg_ha.toFixed(1)} <span className="text-xs font-normal text-slate-400">kg/ha</span>
              </p>
              <p className="text-[11px] text-slate-400 mt-1">
                Certificate: <span className="font-mono text-emerald-400">{explanation.explanation_id || 'EXP-0001'}</span>
              </p>
            </div>
          </div>

          {/* Key Drivers Summary */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
            <div className="p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
              <h4 className="text-xs font-semibold text-emerald-400 mb-1.5 flex items-center gap-1.5">
                <TrendingUp className="w-3.5 h-3.5" /> Top Positive Model Drivers
              </h4>
              <ul className="text-xs text-slate-300 space-y-1">
                {explanation.top_positive_features.map((f, i) => (
                  <li key={i} className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                    {f}
                  </li>
                ))}
              </ul>
            </div>

            <div className="p-3 bg-rose-500/10 border border-rose-500/20 rounded-lg">
              <h4 className="text-xs font-semibold text-rose-400 mb-1.5 flex items-center gap-1.5">
                <TrendingDown className="w-3.5 h-3.5" /> Top Negative Model Drivers
              </h4>
              <ul className="text-xs text-slate-300 space-y-1">
                {explanation.top_negative_features.length > 0 ? (
                  explanation.top_negative_features.map((f, i) => (
                    <li key={i} className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-rose-400" />
                      {f}
                    </li>
                  ))
                ) : (
                  <li className="text-slate-400 italic">No features contributed negatively to this prediction.</li>
                )}
              </ul>
            </div>
          </div>

          {/* Detailed Contribution Waterfall / Bar List */}
          <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3">
            Individual Feature Attribution Waterfall
          </h4>
          <FeatureContributionChart contributions={explanation.feature_contributions} />

          <p className="text-[11px] text-slate-500 mt-4 italic">{explanation.scientific_disclaimer}</p>
        </div>
      )}
    </div>
  )
}
