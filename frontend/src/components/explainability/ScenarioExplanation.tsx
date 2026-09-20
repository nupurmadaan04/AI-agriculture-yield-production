import React, { useState } from 'react'
import { GitCompare, ArrowRight, RefreshCw, CheckCircle2, AlertTriangle } from 'lucide-react'
import { ScenarioExplanationResponse } from '../../types/explainability'
import { useScenarioExplanation } from '../../services/api'

interface ScenarioExplanationProps {
  states?: string[]
}

export const ScenarioExplanation: React.FC<ScenarioExplanationProps> = ({
  states = ['Punjab', 'Haryana', 'Andhra Pradesh', 'Uttar Pradesh', 'West Bengal']
}) => {
  const [selectedState, setSelectedState] = useState<string>('Punjab')
  const [scenarioName, setScenarioName] = useState<string>('High-Input Technology Package')
  const [areaShareDelta, setAreaShareDelta] = useState<number>(0.10)
  const [lagYieldDelta, setLagYieldDelta] = useState<number>(200.0)

  const [data, setData] = useState<ScenarioExplanationResponse | null>(null)
  const scenarioMutation = useScenarioExplanation()

  const handleSimulateAndExplain = () => {
    scenarioMutation.mutate(
      {
        scenarioId: 'SCEN-AUD-001',
        state: selectedState,
        baseline_yield: 3950.0,
        simulated_yield: 4160.0,
        changed_features: {
          'RICE_AREA_SHARE': 0.55,
          'RICE_YIELD_LAG1': 3050.0
        }
      },
      {
        onSuccess: (res) => {
          setData(res)
        }
      }
    )
  }

  React.useEffect(() => {
    handleSimulateAndExplain()
  }, [])

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 mb-6">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4 mb-6">
        <div>
          <h3 className="text-base font-semibold text-white flex items-center gap-2">
            <GitCompare className="w-5 h-5 text-emerald-400" />
            Scenario Shift Attribution & Input Differential Analysis
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Compare user-modified scenario inputs against baseline parameters and explain model response
          </p>
        </div>

        <button
          onClick={handleSimulateAndExplain}
          disabled={scenarioMutation.isPending}
          className="px-4 py-2 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-semibold rounded-lg text-xs flex items-center gap-2 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${scenarioMutation.isPending ? 'animate-spin' : ''}`} />
          Explain Scenario Response
        </button>
      </div>

      {/* Scenario Controls */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 p-4 bg-slate-800/40 rounded-lg border border-slate-800 mb-6">
        <div>
          <label className="block text-xs font-medium text-slate-300 mb-1.5">Target State</label>
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
          <label className="block text-xs font-medium text-slate-300 mb-1.5">Scenario Archetype</label>
          <input
            type="text"
            value={scenarioName}
            onChange={(e) => setScenarioName(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-emerald-500"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-300 mb-1.5">Lagged Yield Delta (kg/ha)</label>
          <input
            type="number"
            value={lagYieldDelta}
            onChange={(e) => setLagYieldDelta(Number(e.target.value))}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-emerald-500"
          />
        </div>
      </div>

      {data && (
        <div>
          {/* Comparison Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
            <div className="bg-slate-800/40 p-4 rounded-xl border border-slate-800">
              <p className="text-xs text-slate-400 mb-1">Baseline Yield</p>
              <p className="text-xl font-bold text-slate-300">
                {data.baseline_yield_kg_ha.toLocaleString()} <span className="text-xs font-normal text-slate-400">kg/ha</span>
              </p>
              <p className="text-[11px] text-slate-400 mt-1">Empirical Status Quo</p>
            </div>

            <div className="bg-slate-800/40 p-4 rounded-xl border border-slate-800">
              <p className="text-xs text-slate-400 mb-1">Simulated Yield</p>
              <p className="text-xl font-bold text-white">
                {data.simulated_yield_kg_ha.toLocaleString()} <span className="text-xs font-normal text-slate-400">kg/ha</span>
              </p>
              <p className="text-[11px] text-slate-400 mt-1">Modified Input Space</p>
            </div>

            <div className="bg-slate-800/40 p-4 rounded-xl border border-slate-800">
              <p className="text-xs text-slate-400 mb-1">Attributed Response</p>
              <p
                className={`text-xl font-bold ${
                  data.simulated_delta_kg_ha >= 0 ? 'text-emerald-400' : 'text-rose-400'
                }`}
              >
                {data.simulated_delta_kg_ha >= 0 ? '+' : ''}
                {data.simulated_delta_kg_ha.toFixed(1)} kg/ha ({data.simulated_delta_pct >= 0 ? '+' : ''}
                {data.simulated_delta_pct}%)
              </p>
              <p className="text-[11px] text-slate-400 mt-1">Net Model Differential</p>
            </div>
          </div>

          {/* Changed Inputs Grid */}
          <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3">
            Modified Input Features
          </h4>
          <div className="space-y-2 mb-6">
            {data.changed_inputs.map((inp) => (
              <div
                key={inp.feature}
                className="p-3 bg-slate-800/40 rounded-lg border border-slate-800 flex items-center justify-between"
              >
                <div>
                  <span className="text-xs font-bold text-white">{inp.feature_label}</span>
                  <span className="text-[10px] font-mono text-slate-400 ml-2">({inp.feature})</span>
                </div>
                <div className="text-xs font-mono text-emerald-400 font-semibold">
                  Modified Value: {inp.modified_value}
                </div>
              </div>
            ))}
          </div>

          {/* Synthesis Narrative */}
          <div className="p-4 bg-slate-800/60 border border-slate-700/80 rounded-xl mb-4">
            <h5 className="text-xs font-bold text-white mb-1 flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              Model Attribution Synthesis
            </h5>
            <p className="text-xs text-slate-300 leading-relaxed">{data.model_attribution_summary}</p>
          </div>

          <p className="text-[11px] text-slate-500 italic">{data.scientific_disclaimer}</p>
        </div>
      )}
    </div>
  )
}
