import React, { useState, useEffect } from 'react'
import {
  Layers,
  Play,
  Download,
  AlertCircle,
  FileText,
  ShieldCheck,
  TrendingUp,
  Activity,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  Database,
  ArrowRight,
  Sparkles,
  Info,
  Scale
} from 'lucide-react'
import { workspaceService, useWorkspaceTemplatesQuery } from '../services/workspaceService'
import { useStates } from '../services/api'
import type { DecisionWorkspaceResponse, ScenarioItem } from '../types/workspace'

const MULTICROP_COMMODITIES = [
  { name: 'Oilseeds', tier: 'PRODUCTION_READY', strat: 'Historical ML (RandomForestRegressor)' },
  { name: 'Sugarcane', tier: 'CONDITIONAL_PRODUCTION', strat: 'Historical ML (GradientBoostingRegressor)' },
  { name: 'Rice', tier: 'BASELINE_PRODUCTION', strat: 'Historical District Mean / Persistence' },
  { name: 'Wheat', tier: 'BASELINE_PRODUCTION', strat: 'Historical District Mean / Persistence' },
  { name: 'Maize', tier: 'BASELINE_PRODUCTION', strat: 'Historical District Mean / Persistence' },
  { name: 'Chickpea', tier: 'BASELINE_PRODUCTION', strat: 'Historical District Mean / Persistence' },
  { name: 'Kharif Sorghum', tier: 'BASELINE_PRODUCTION', strat: 'Historical District Mean / Persistence' },
  { name: 'Minor Pulses', tier: 'BASELINE_PRODUCTION', strat: 'Historical District Mean / Persistence' },
  { name: 'Pigeonpea', tier: 'BASELINE_PRODUCTION', strat: 'Historical District Mean / Persistence' },
  { name: 'Rapeseed and Mustard', tier: 'BASELINE_PRODUCTION', strat: 'Historical District Mean / Persistence' },
  { name: 'Sesamum', tier: 'BASELINE_PRODUCTION', strat: 'Historical District Mean / Persistence' },
  { name: 'Groundnut', tier: 'BASELINE_PRODUCTION', strat: 'Historical District Mean / Persistence' },
  { name: 'Sorghum', tier: 'BASELINE_PRODUCTION', strat: 'Historical District Mean / Persistence' },
  { name: 'Pearl Millet', tier: 'BASELINE_PRODUCTION', strat: 'Historical District Mean / Persistence' }
]

export const DecisionWorkspace: React.FC = () => {
  const [selectedCrop, setSelectedCrop] = useState('Oilseeds')
  const [selectedState, setSelectedState] = useState('Punjab')
  const [district, setDistrict] = useState('Ludhiana')
  const [selectedYear, setSelectedYear] = useState(2017)
  const [areaDelta, setAreaDelta] = useState(0)
  const [lagDelta, setLagDelta] = useState(0)
  const [includeCustom, setIncludeCustom] = useState(false)

  const [workspaceData, setWorkspaceData] = useState<DecisionWorkspaceResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  const { data: statesData } = useStates()
  const { data: templatesData } = useWorkspaceTemplatesQuery()

  const stateNames = statesData?.data?.map(s => s.state) || [
    'Punjab', 'Haryana', 'Andhra Pradesh', 'Uttar Pradesh', 'West Bengal', 'Tamil Nadu', 'Karnataka', 'Odisha', 'Maharashtra', 'Gujarat'
  ]

  const handleRunAnalysis = async () => {
    setIsLoading(true)
    setErrorMsg(null)
    try {
      const customMods = includeCustom && (areaDelta !== 0 || lagDelta !== 0)
        ? { rice_area_pct: areaDelta, historical_yield_lag_pct: lagDelta }
        : undefined

      const res = await workspaceService.analyzeWorkspace({
        crop: selectedCrop,
        state: selectedState,
        district: district || undefined,
        forecast_year: selectedYear,
        selected_scenarios: ['conservative_improvement', 'moderate_improvement', 'stress_scenario'],
        custom_modifications: customMods
      })
      setWorkspaceData(res)
    } catch (err: any) {
      setErrorMsg(err.message || 'Workspace analysis failed')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    handleRunAnalysis()
  }, [])

  const exportJSON = () => {
    if (!workspaceData) return
    const blob = new Blob([JSON.stringify(workspaceData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `decision_workspace_${workspaceData.crop}_${workspaceData.forecast_year}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const exportMarkdown = () => {
    if (!workspaceData) return
    const md = `# Decision Workspace Report: ${workspaceData.crop} (${workspaceData.state}, ${workspaceData.district || 'All'})\n\n` +
      `**Forecast Year:** ${workspaceData.forecast_year} | **Workspace ID:** \`${workspaceData.workspace_id}\`\n\n` +
      `## Governed Baseline Forecast\n` +
      `- **Forecasted Yield:** ${workspaceData.baseline_forecast.forecast_yield_kg_ha} kg/ha\n` +
      `- **Strategy:** ${workspaceData.baseline_forecast.strategy} (${workspaceData.baseline_forecast.certification_status})\n` +
      `- **Provenance SHA-256:** \`${workspaceData.baseline_forecast.provenance_hash}\`\n\n` +
      `## Scenario Comparison Matrix\n\n` +
      `| Metric | Baseline | ` + workspaceData.scenarios.map(s => s.scenario_name).join(' | ') + ` |\n` +
      `|---|---|` + workspaceData.scenarios.map(() => '---').join('|') + `|\n` +
      `| Projected Yield (kg/ha) | ${workspaceData.baseline_forecast.forecast_yield_kg_ha} | ` + workspaceData.scenarios.map(s => `${s.scenario_output_kg_ha}`).join(' | ') + ` |\n` +
      `| Yield Delta (kg/ha) | 0.0 | ` + workspaceData.scenarios.map(s => `${s.yield_delta_kg_ha > 0 ? '+' : ''}${s.yield_delta_kg_ha}`).join(' | ') + ` |\n` +
      `| Relative Change (%) | 0.0% | ` + workspaceData.scenarios.map(s => `${s.yield_percent_change > 0 ? '+' : ''}${s.yield_percent_change}%`).join(' | ') + ` |\n\n` +
      `> ${workspaceData.decision_support_statement}\n`

    const blob = new Blob([md], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `decision_workspace_${workspaceData.crop}_${workspaceData.forecast_year}.md`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="min-h-screen bg-[#FBFBF9] text-[#1E293B] py-8 px-4 sm:px-6 lg:px-8 font-sans antialiased">
      <div className="max-w-7xl mx-auto space-y-8">

        {/* 1. Header & Governance Invariant */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 pb-6 border-b border-emerald-900/10">
          <div>
            <div className="flex items-center gap-2.5">
              <span className="p-2 rounded-lg bg-emerald-950 text-emerald-400">
                <Layers className="w-5 h-5" />
              </span>
              <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900">
                Production Decision Workspace
              </h1>
              <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 border border-emerald-300">
                Day 32 Production
              </span>
            </div>
            <p className="mt-1.5 text-sm text-slate-600 max-w-3xl">
              Inspect governed pre-season forecasts, explore what-if scenario simulations, evaluate out-of-time validation benchmarks,
              and compare quantitative trade-offs without autonomous prescriptive ranking.
            </p>
          </div>

          <div className="flex items-center gap-2.5">
            <button
              onClick={exportJSON}
              disabled={!workspaceData}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold rounded-md border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 transition shadow-sm disabled:opacity-50"
            >
              <Download className="w-3.5 h-3.5" />
              JSON
            </button>
            <button
              onClick={exportMarkdown}
              disabled={!workspaceData}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold rounded-md border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 transition shadow-sm disabled:opacity-50"
            >
              <FileText className="w-3.5 h-3.5" />
              Markdown
            </button>
          </div>
        </div>

        {/* Non-Autonomous Decision Banner */}
        <div className="p-3.5 rounded-lg bg-emerald-50/70 border border-emerald-200/80 text-xs text-emerald-900 flex items-start gap-2.5">
          <Info className="w-4 h-4 text-emerald-700 shrink-0 mt-0.5" />
          <div className="space-y-0.5">
            <span className="font-semibold">Decision-Support Governance:</span>
            <p className="text-emerald-800/90">
              The Decision Workspace synthesizes analytical evidence to assist agricultural planners. It does not rank options as "BEST" or "RECOMMENDED".
              Scenario projections represent hypothetical mathematical simulations, not physical causal certainties.
            </p>
          </div>
        </div>

        {/* 2. Control Panel & Commodity Selector */}
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-6">
          <div className="space-y-3">
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Commodity Selection (14 Certified Agricultural Crops)
            </label>
            <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-2">
              {MULTICROP_COMMODITIES.map(c => {
                const isSel = selectedCrop === c.name
                const isML = c.tier.includes('PRODUCTION') && !c.tier.includes('BASELINE')
                return (
                  <button
                    key={c.name}
                    onClick={() => setSelectedCrop(c.name)}
                    className={`px-3 py-2 text-xs rounded-lg text-left transition font-medium border ${
                      isSel
                        ? 'bg-emerald-950 text-white border-emerald-950 shadow-sm'
                        : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
                    }`}
                  >
                    <div className="font-semibold truncate">{c.name}</div>
                    <div className="text-[10px] opacity-75 truncate mt-0.5">
                      {isML ? 'Certified ML' : 'Baseline'}
                    </div>
                  </button>
                )
              })}
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 pt-2">
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">State</label>
              <select
                value={selectedState}
                onChange={e => setSelectedState(e.target.value)}
                className="w-full text-xs rounded-md border border-slate-300 py-2 px-3 bg-white focus:outline-none focus:ring-1 focus:ring-emerald-500 text-slate-800"
              >
                {stateNames.map(s => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">District</label>
              <input
                type="text"
                value={district}
                onChange={e => setDistrict(e.target.value)}
                placeholder="District (e.g. Ludhiana, Meerut)"
                className="w-full text-xs rounded-md border border-slate-300 py-2 px-3 bg-white focus:outline-none focus:ring-1 focus:ring-emerald-500 text-slate-800"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">Forecast Target Year</label>
              <input
                type="number"
                min={1966}
                max={2026}
                value={selectedYear}
                onChange={e => setSelectedYear(parseInt(e.target.value) || 2017)}
                className="w-full text-xs rounded-md border border-slate-300 py-2 px-3 bg-white focus:outline-none focus:ring-1 focus:ring-emerald-500 text-slate-800"
              />
            </div>

            <div className="flex items-end">
              <button
                onClick={handleRunAnalysis}
                disabled={isLoading}
                className="w-full inline-flex items-center justify-center gap-2 py-2 px-4 rounded-md text-xs font-semibold text-white bg-emerald-800 hover:bg-emerald-900 transition shadow disabled:opacity-50"
              >
                {isLoading ? (
                  <>
                    <Activity className="w-3.5 h-3.5 animate-spin" />
                    Synthesizing...
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 fill-current" />
                    Synthesize Workspace
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {errorMsg && (
          <div className="p-4 rounded-lg bg-rose-50 border border-rose-200 text-xs text-rose-800 flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}

        {/* 3. Main Workspace Dashboard */}
        {workspaceData && (
          <div className="space-y-8">

            {/* Baseline Overview Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

              {/* Card 1: Governed Baseline Forecast */}
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                    Governed Baseline Forecast
                  </span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-100 text-emerald-800">
                    {workspaceData.baseline_forecast.semantic_classification}
                  </span>
                </div>

                <div className="flex items-baseline gap-2">
                  <span className="text-3xl font-bold tracking-tight text-slate-900">
                    {workspaceData.baseline_forecast.forecast_yield_kg_ha.toLocaleString()}
                  </span>
                  <span className="text-xs text-slate-500 font-medium">kg/ha</span>
                </div>

                <div className="space-y-1.5 text-xs text-slate-600 pt-2 border-t border-slate-100">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Strategy:</span>
                    <span className="font-medium text-slate-800 truncate max-w-[180px]" title={workspaceData.baseline_forecast.strategy}>
                      {workspaceData.baseline_forecast.strategy}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Certification:</span>
                    <span className="font-semibold text-emerald-800">
                      {workspaceData.baseline_forecast.certification_status}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Model Artifact:</span>
                    <span className="font-medium text-slate-700">
                      {workspaceData.baseline_forecast.model_name}
                    </span>
                  </div>
                </div>

                <div className="pt-2 border-t border-slate-100 text-[11px] text-slate-400 font-mono truncate">
                  SHA: {workspaceData.baseline_forecast.provenance_hash}
                </div>
              </div>

              {/* Card 2: Historical Context & Trajectory */}
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                    Historical Benchmark
                  </span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-100 text-amber-800">
                    {workspaceData.historical_context.semantic_classification}
                  </span>
                </div>

                <div className="flex items-baseline gap-2">
                  <span className="text-3xl font-bold tracking-tight text-slate-900">
                    {workspaceData.historical_context.historical_mean_yield_kg_ha.toLocaleString()}
                  </span>
                  <span className="text-xs text-slate-500 font-medium">kg/ha (Mean)</span>
                </div>

                <div className="space-y-1.5 text-xs text-slate-600 pt-2 border-t border-slate-100">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Historical Window:</span>
                    <span className="font-medium text-slate-800">
                      {workspaceData.historical_context.historical_period} ({workspaceData.historical_context.sample_count} pts)
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Observed Spread:</span>
                    <span className="font-medium text-slate-800">
                      {workspaceData.historical_context.historical_min_yield_kg_ha} – {workspaceData.historical_context.historical_max_yield_kg_ha} kg/ha
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Trajectory Slope:</span>
                    <span className={`font-semibold ${workspaceData.historical_context.trend_slope_kg_ha_yr >= 0 ? 'text-emerald-700' : 'text-rose-700'}`}>
                      {workspaceData.historical_context.trend_slope_kg_ha_yr > 0 ? '+' : ''}{workspaceData.historical_context.trend_slope_kg_ha_yr} kg/ha/yr
                    </span>
                  </div>
                </div>

                <div className="pt-2 border-t border-slate-100 text-[11px] text-slate-500">
                  Temporal boundary isolated strictly to observations &lt; {workspaceData.forecast_year}
                </div>
              </div>

              {/* Card 3: Validation & Empirical Uncertainty */}
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                    Validation & Uncertainty
                  </span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-teal-100 text-teal-800">
                    {workspaceData.validation.semantic_classification}
                  </span>
                </div>

                <div className="flex items-baseline gap-2">
                  <span className="text-3xl font-bold tracking-tight text-slate-900">
                    {workspaceData.validation.mae_kg_ha}
                  </span>
                  <span className="text-xs text-slate-500 font-medium">kg/ha (Test MAE)</span>
                </div>

                <div className="space-y-1.5 text-xs text-slate-600 pt-2 border-t border-slate-100">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Fold Win Rate:</span>
                    <span className="font-semibold text-emerald-800">
                      {workspaceData.validation.fold_win_rate_pct}% vs baseline
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Ensemble Dispersion:</span>
                    <span className="font-medium text-slate-800">
                      {workspaceData.uncertainty.is_available
                        ? `±${((workspaceData.uncertainty.ensemble_spread_kg_ha || 0) / 2).toFixed(1)} kg/ha (P10–P90)`
                        : 'Unavailable (Deterministic Baseline)'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Validation Protocol:</span>
                    <span className="font-medium text-slate-700">
                      {workspaceData.validation.validation_protocol}
                    </span>
                  </div>
                </div>

                <div className="pt-2 border-t border-slate-100 text-[11px] text-slate-500 italic">
                  {workspaceData.uncertainty.disclaimer}
                </div>
              </div>
            </div>

            {/* 4. What-If Scenario Simulator & Interactive Controls */}
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-5">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                    <Sliders className="w-4 h-4 text-emerald-700" />
                    What-If Scenario Simulation
                  </h3>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Test counterfactual assumptions by perturbing input acreage allocation or historical yield anchors.
                  </p>
                </div>
                <div className="flex items-center gap-2 text-xs">
                  <input
                    type="checkbox"
                    id="enable-custom"
                    checked={includeCustom}
                    onChange={e => setIncludeCustom(e.target.checked)}
                    className="rounded text-emerald-700 focus:ring-emerald-500"
                  />
                  <label htmlFor="enable-custom" className="font-medium text-slate-700 cursor-pointer">
                    Enable Custom Perturbation
                  </label>
                </div>
              </div>

              {includeCustom && (
                <div className="p-4 rounded-lg bg-slate-50 border border-slate-200 grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-medium">
                      <span className="text-slate-700">Acreage Allocation Perturbation:</span>
                      <span className="text-emerald-800 font-bold">{areaDelta > 0 ? `+${areaDelta}%` : `${areaDelta}%`}</span>
                    </div>
                    <input
                      type="range"
                      min={-30}
                      max={30}
                      step={1}
                      value={areaDelta}
                      onChange={e => setAreaDelta(parseInt(e.target.value))}
                      className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-800"
                    />
                    <div className="flex justify-between text-[10px] text-slate-400">
                      <span>-30%</span>
                      <span>0% (Baseline)</span>
                      <span>+30%</span>
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-medium">
                      <span className="text-slate-700">Historical Yield Lag Perturbation:</span>
                      <span className="text-emerald-800 font-bold">{lagDelta > 0 ? `+${lagDelta}%` : `${lagDelta}%`}</span>
                    </div>
                    <input
                      type="range"
                      min={-30}
                      max={30}
                      step={1}
                      value={lagDelta}
                      onChange={e => setLagDelta(parseInt(e.target.value))}
                      className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-800"
                    />
                    <div className="flex justify-between text-[10px] text-slate-400">
                      <span>-30%</span>
                      <span>0% (Baseline)</span>
                      <span>+30%</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* 5. Side-by-Side Scenario Comparison Matrix */}
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
              <div className="p-5 border-b border-slate-100 bg-slate-50/50 flex items-center justify-between">
                <div>
                  <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                    <Scale className="w-4 h-4 text-emerald-700" />
                    Scenario Comparison Matrix
                  </h3>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Quantitative differences across simulated archetypes. Scenarios are not ranked or prescribed.
                  </p>
                </div>
                <span className="px-2.5 py-1 rounded text-xs font-semibold bg-slate-200 text-slate-700">
                  {workspaceData.scenarios.length + 1} Columns
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-100/75 text-slate-700 border-b border-slate-200 font-semibold">
                    <tr>
                      <th className="py-3 px-4 w-1/4">Evaluation Dimension</th>
                      <th className="py-3 px-4 bg-emerald-50/50 border-r border-slate-200 text-emerald-950">
                        Baseline (Governed)
                      </th>
                      {workspaceData.scenarios.map(s => (
                        <th key={s.scenario_id} className="py-3 px-4 text-slate-800">
                          {s.scenario_name}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 text-slate-600">

                    {/* Row 1: Output Yield */}
                    <tr className="hover:bg-slate-50/80">
                      <td className="py-3 px-4 font-semibold text-slate-800">Projected Yield (kg/ha)</td>
                      <td className="py-3 px-4 font-bold text-emerald-950 bg-emerald-50/30 border-r border-slate-200">
                        {workspaceData.baseline_forecast.forecast_yield_kg_ha.toFixed(1)}
                      </td>
                      {workspaceData.scenarios.map(s => (
                        <td key={s.scenario_id} className="py-3 px-4 font-semibold text-slate-900">
                          {s.scenario_output_kg_ha.toFixed(1)}
                        </td>
                      ))}
                    </tr>

                    {/* Row 2: Delta vs Baseline */}
                    <tr className="hover:bg-slate-50/80">
                      <td className="py-3 px-4 font-semibold text-slate-800">Yield Delta vs Baseline (kg/ha)</td>
                      <td className="py-3 px-4 font-mono text-slate-500 bg-emerald-50/30 border-r border-slate-200">
                        0.0
                      </td>
                      {workspaceData.scenarios.map(s => (
                        <td key={s.scenario_id} className="py-3 px-4 font-mono font-semibold">
                          <span className={s.yield_delta_kg_ha > 0 ? 'text-emerald-700' : s.yield_delta_kg_ha < 0 ? 'text-rose-700' : 'text-slate-600'}>
                            {s.yield_delta_kg_ha > 0 ? `+${s.yield_delta_kg_ha.toFixed(1)}` : s.yield_delta_kg_ha.toFixed(1)}
                          </span>
                        </td>
                      ))}
                    </tr>

                    {/* Row 3: Relative Change */}
                    <tr className="hover:bg-slate-50/80">
                      <td className="py-3 px-4 font-semibold text-slate-800">Relative Change (%)</td>
                      <td className="py-3 px-4 font-mono text-slate-500 bg-emerald-50/30 border-r border-slate-200">
                        0.0%
                      </td>
                      {workspaceData.scenarios.map(s => (
                        <td key={s.scenario_id} className="py-3 px-4 font-mono font-semibold">
                          <span className={s.yield_percent_change > 0 ? 'text-emerald-700' : s.yield_percent_change < 0 ? 'text-rose-700' : 'text-slate-600'}>
                            {s.yield_percent_change > 0 ? `+${s.yield_percent_change.toFixed(2)}%` : `${s.yield_percent_change.toFixed(2)}%`}
                          </span>
                        </td>
                      ))}
                    </tr>

                    {/* Row 4: Uncertainty */}
                    <tr className="hover:bg-slate-50/80">
                      <td className="py-3 px-4 font-semibold text-slate-800">Uncertainty Bounds</td>
                      <td className="py-3 px-4 text-slate-700 bg-emerald-50/30 border-r border-slate-200">
                        {workspaceData.uncertainty.is_available
                          ? `±${((workspaceData.uncertainty.ensemble_spread_kg_ha || 0) / 2).toFixed(1)} kg/ha (Tree P10–P90)`
                          : 'Not Available (Deterministic)'}
                      </td>
                      {workspaceData.scenarios.map(s => (
                        <td key={s.scenario_id} className="py-3 px-4 text-slate-600">
                          {s.uncertainty_note}
                        </td>
                      ))}
                    </tr>

                    {/* Row 5: Assumptions */}
                    <tr className="hover:bg-slate-50/80">
                      <td className="py-3 px-4 font-semibold text-slate-800">Simulation Assumption</td>
                      <td className="py-3 px-4 text-slate-600 bg-emerald-50/30 border-r border-slate-200">
                        Baseline empirical conditions (no intervention)
                      </td>
                      {workspaceData.scenarios.map(s => (
                        <td key={s.scenario_id} className="py-3 px-4 text-slate-600">
                          {s.scenario_assumption}
                        </td>
                      ))}
                    </tr>

                    {/* Row 6: Evidence Type */}
                    <tr className="hover:bg-slate-50/80">
                      <td className="py-3 px-4 font-semibold text-slate-800">Semantic Classification</td>
                      <td className="py-3 px-4 bg-emerald-50/30 border-r border-slate-200">
                        <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-100 text-emerald-800">
                          PREDICTED
                        </span>
                      </td>
                      {workspaceData.scenarios.map(s => (
                        <td key={s.scenario_id} className="py-3 px-4">
                          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-purple-100 text-purple-800">
                            {s.evidence_type}
                          </span>
                        </td>
                      ))}
                    </tr>

                    {/* Row 7: Status */}
                    <tr className="hover:bg-slate-50/80">
                      <td className="py-3 px-4 font-semibold text-slate-800">Operational Status</td>
                      <td className="py-3 px-4 text-emerald-800 font-semibold bg-emerald-50/30 border-r border-slate-200">
                        {workspaceData.baseline_forecast.certification_status}
                      </td>
                      {workspaceData.scenarios.map(s => (
                        <td key={s.scenario_id} className="py-3 px-4">
                          <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-100 text-slate-700">
                            {s.status}
                          </span>
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* 6. Feature Attribution & Monitoring Health (Dual Cards) */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

              {/* Attribution (Tree SHAP) */}
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                    Model Feature Attribution
                  </span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-100 text-indigo-800">
                    {workspaceData.attribution.semantic_classification}
                  </span>
                </div>

                <div className="space-y-3 pt-2">
                  {workspaceData.attribution.top_features.map((feat, idx) => (
                    <div key={idx} className="space-y-1 text-xs">
                      <div className="flex justify-between font-medium text-slate-800">
                        <span>{feat.feature_label}</span>
                        <span className="font-mono text-indigo-700">
                          {feat.importance_or_shap.toFixed(2)}
                        </span>
                      </div>
                      <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-indigo-600 rounded-full"
                          style={{ width: `${Math.min(100, Math.abs(feat.importance_or_shap) * 100)}%` }}
                        />
                      </div>
                      <p className="text-[11px] text-slate-500">{feat.interpretation}</p>
                    </div>
                  ))}
                </div>

                <p className="text-[11px] text-slate-500 pt-2 border-t border-slate-100 italic">
                  {workspaceData.attribution.methodology}
                </p>
              </div>

              {/* Monitoring & Outcome Evaluation */}
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                    Operational Monitoring & Outcomes
                  </span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-rose-100 text-rose-800">
                    {workspaceData.monitoring.semantic_classification}
                  </span>
                </div>

                <div className="space-y-2 text-xs pt-2">
                  <div className="flex justify-between py-1 border-b border-slate-50">
                    <span className="text-slate-600">Prediction PSI Drift:</span>
                    <span className={`font-mono font-semibold ${workspaceData.monitoring.overall_psi < 0.1 ? 'text-emerald-700' : 'text-amber-700'}`}>
                      {workspaceData.monitoring.overall_psi.toFixed(4)} ({workspaceData.monitoring.drift_status})
                    </span>
                  </div>

                  <div className="flex justify-between py-1 border-b border-slate-50">
                    <span className="text-slate-600">Outcome Evaluation:</span>
                    <span className={`font-semibold ${workspaceData.monitoring.outcome_evaluation_status === 'EVALUATION_AVAILABLE' ? 'text-emerald-700' : 'text-slate-500'}`}>
                      {workspaceData.monitoring.outcome_evaluation_status}
                    </span>
                  </div>

                  {workspaceData.monitoring.observed_outcome_kg_ha !== null && (
                    <div className="flex justify-between py-1 border-b border-slate-50">
                      <span className="text-slate-600">Observed Harvest Outcome:</span>
                      <span className="font-semibold text-slate-900">
                        {workspaceData.monitoring.observed_outcome_kg_ha} kg/ha (Error: {workspaceData.monitoring.forecast_error_kg_ha} kg/ha)
                      </span>
                    </div>
                  )}

                  <div className="flex justify-between py-1">
                    <span className="text-slate-600">Active Operational Alerts:</span>
                    <span className="font-medium text-slate-800">
                      {workspaceData.monitoring.active_alerts.length === 0 ? 'None (Clean)' : `${workspaceData.monitoring.active_alerts.length} active`}
                    </span>
                  </div>
                </div>

                <div className="p-3 rounded bg-slate-50 border border-slate-200/80 text-[11px] text-slate-600">
                  <span className="font-semibold text-slate-800">Cryptographic Lineage:</span>
                  <div className="font-mono text-[10px] text-slate-500 truncate mt-0.5">
                    Ref: {workspaceData.provenance.audit_reference} | SHA: {workspaceData.provenance.prediction_fingerprint}
                  </div>
                </div>
              </div>

            </div>

            {/* 7. Explicit Limitations Panel */}
            <div className="bg-amber-50/50 rounded-xl border border-amber-200/80 p-5 space-y-3">
              <h4 className="text-xs font-bold uppercase tracking-wider text-amber-900 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-700" />
                Scientific Operating Limitations
              </h4>
              <ul className="space-y-1.5 text-xs text-amber-950/90 list-disc list-inside">
                {workspaceData.limitations.map((lim, idx) => (
                  <li key={idx}>{lim}</li>
                ))}
              </ul>
            </div>

          </div>
        )}

      </div>
    </div>
  )
}
