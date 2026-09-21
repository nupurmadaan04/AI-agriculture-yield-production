import React, { useState, useEffect } from 'react'
import { useDecisionAnalysis } from '../services/decisionService'
import { useStates } from '../services/api'
import {
  Compass,
  Play,
  Download,
  AlertCircle,
  FileText,
  ShieldCheck,
  RefreshCw,
  Clock,
  TrendingUp,
  Activity,
  Layers,
  HelpCircle,
  CheckCircle2,
  AlertTriangle,
  Database,
  Search,
  ExternalLink,
  Code
} from 'lucide-react'

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

export const DecisionIntelligence: React.FC = () => {
  const [selectedCrop, setSelectedCrop] = useState('Oilseeds')
  const [selectedState, setSelectedState] = useState('Punjab')
  const [district, setDistrict] = useState('Ludhiana')
  const [selectedYear, setSelectedYear] = useState(2017)
  const [horizon, setHorizon] = useState('next_season')
  const [activeEvidenceFilter, setActiveEvidenceFilter] = useState<string>('ALL')

  const { data: statesData } = useStates()
  const stateNames = statesData?.data?.map(s => s.state) || [
    'Punjab', 'Haryana', 'Andhra Pradesh', 'Uttar Pradesh', 'West Bengal', 'Tamil Nadu', 'Karnataka', 'Odisha', 'Maharashtra', 'Gujarat'
  ]

  const { mutate: analyze, data: decisionRes, isPending, error } = useDecisionAnalysis()

  // Execute initial analysis on load
  useEffect(() => {
    analyze({
      crop: selectedCrop,
      state: selectedState,
      district: district || undefined,
      year: selectedYear,
      decision_horizon: horizon
    })
  }, [])

  const handleAnalyze = () => {
    analyze({
      crop: selectedCrop,
      state: selectedState,
      district: district || undefined,
      year: selectedYear,
      decision_horizon: horizon
    })
  }

  const brief = decisionRes?.brief
  const forecast = brief?.forecast_summary
  const hist = brief?.historical_context
  const val = brief?.validation_evidence
  const unc = brief?.uncertainty_evidence
  const mon = brief?.monitoring_evidence
  const completeness = brief?.evidence_status?.completeness_level || 'PARTIAL_EVIDENCE'

  const exportMarkdown = () => {
    if (!brief) return
    const md = `# Agricultural Decision Evidence Report (Day 31)

Decision ID: ${brief.decision_id}
Generated: ${brief.generated_at}
Target: ${brief.context.crop} • ${brief.context.state}${brief.context.district ? `, ${brief.context.district}` : ''} (${brief.context.year})
Evidence Completeness: ${completeness}

## Executive Summary
${brief.executive_summary.current_status}
${brief.executive_summary.outlook}
${brief.executive_summary.major_risk_signal}

## Forecast & Strategy
- Forecast: ${forecast?.forecast_yield_kg_ha ?? 'N/A'} kg/ha
- Strategy: ${forecast?.strategy ?? 'N/A'} (${forecast?.certification_status ?? 'N/A'})
- Model: ${forecast?.model_name ?? 'N/A'}
- Provenance: ${forecast?.provenance_hash ?? 'N/A'}

## Walk-Forward Validation
- Validation Protocol: ${val?.validation_protocol ?? 'N/A'} (${val?.validation_period ?? 'N/A'})
- Strategy MAE: ${val?.mae_kg_ha ?? 'N/A'} kg/ha
- Baseline MAE: ${val?.baseline_mae_kg_ha ?? 'N/A'} kg/ha
- Fold Win Rate: ${val?.fold_win_rate_pct ?? 'N/A'}%

## Historical Context (${hist?.start_year ?? 1966}–${hist?.end_year ?? 2016})
- Historical Mean Yield: ${hist?.historical_mean_yield_kg_ha ?? 'N/A'} kg/ha
- Historical Range: ${hist?.historical_min_yield_kg_ha ?? 'N/A'} to ${hist?.historical_max_yield_kg_ha ?? 'N/A'} kg/ha
- Trend Trajectory Slope: ${hist?.trend_slope_kg_ha_yr ?? '0.00'} kg/ha/year

## Assumptions
${(brief.assumptions || []).map(a => `* ${a}`).join('\n')}

## Limitations
${(brief.limitations || []).map(l => `* ${l}`).join('\n')}

---
${brief.footer_disclaimer}
`
    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.setAttribute('download', `DECISION_BRIEF_${brief.decision_id}.md`)
    document.body.appendChild(link)
    link.click()
    link.remove()
  }

  const exportJSON = () => {
    if (!decisionRes) return
    const blob = new Blob([JSON.stringify(decisionRes, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.setAttribute('download', `DECISION_BRIEF_${brief?.decision_id || 'REPORT'}.json`)
    document.body.appendChild(link)
    link.click()
    link.remove()
  }

  const filteredEvidence = (brief?.evidence_items || []).filter(item => {
    if (activeEvidenceFilter === 'ALL') return true
    return item.evidence_type === activeEvidenceFilter
  })

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 lg:p-8 space-y-8 font-sans">
      {/* 1. Header & Metadata */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div className="space-y-1.5">
          <div className="flex items-center gap-3">
            <span className="p-2 rounded-lg bg-emerald-950/80 border border-emerald-800/60 text-emerald-400">
              <Compass className="w-5 h-5" />
            </span>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl lg:text-2xl font-bold tracking-tight text-slate-100">
                  Decision Intelligence & Evidence-Based Forecast Brief
                </h1>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-900/40 text-emerald-300 border border-emerald-700/50">
                  DAY 31 GOVERNED
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Multi-layer synthesis • Walk-forward validation • Tree SHAP • Day 30 monitoring signals • Strictly non-causal decision support
              </p>
            </div>
          </div>
        </div>

        {brief && (
          <div className="flex items-center gap-2.5">
            <button
              onClick={exportMarkdown}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 text-slate-200 rounded-lg text-xs font-medium border border-slate-700 transition"
              title="Download report in Markdown format"
            >
              <Download className="w-3.5 h-3.5 text-emerald-400" />
              <span>Export Brief (.md)</span>
            </button>
            <button
              onClick={exportJSON}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 text-slate-200 rounded-lg text-xs font-medium border border-slate-700 transition"
              title="Download raw JSON evidence payload"
            >
              <Code className="w-3.5 h-3.5 text-blue-400" />
              <span>Raw JSON</span>
            </button>
          </div>
        )}
      </div>

      {/* 2. Decision Context Selector Bar */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-xl flex flex-wrap items-end gap-3.5">
        <div className="space-y-1.5 min-w-[170px]">
          <label className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1">
            <span>Crop Commodity</span>
            <span className="text-slate-500 font-normal">({MULTICROP_COMMODITIES.length})</span>
          </label>
          <select
            value={selectedCrop}
            onChange={e => setSelectedCrop(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-emerald-500"
          >
            {MULTICROP_COMMODITIES.map(c => (
              <option key={c.name} value={c.name}>
                {c.name} ({c.tier === 'PRODUCTION_READY' ? 'ML Certified' : c.tier === 'CONDITIONAL_PRODUCTION' ? 'Conditional ML' : 'Baseline'})
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-1.5 min-w-[160px]">
          <label className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider">State</label>
          <select
            value={selectedState}
            onChange={e => setSelectedState(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-emerald-500"
          >
            {stateNames.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>

        <div className="space-y-1.5 min-w-[150px]">
          <label className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider">District</label>
          <input
            type="text"
            placeholder="e.g. Ludhiana"
            value={district}
            onChange={e => setDistrict(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-emerald-500"
          />
        </div>

        <div className="space-y-1.5 min-w-[120px]">
          <label className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider">Forecast Year</label>
          <select
            value={selectedYear}
            onChange={e => setSelectedYear(Number(e.target.value))}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-emerald-500"
          >
            {[2026, 2025, 2018, 2017, 2016, 2015, 2014].map(y => (
              <option key={y} value={y}>
                {y} {y > 2017 ? '(Future Horizon)' : '(Ground Truth Available)'}
              </option>
            ))}
          </select>
        </div>

        <button
          onClick={handleAnalyze}
          disabled={isPending}
          className="flex items-center gap-2 px-5 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold shadow transition-colors"
        >
          {isPending ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-white" />}
          <span>{isPending ? 'Synthesizing...' : 'Synthesize Brief'}</span>
        </button>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-xs text-red-400 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>Failed to synthesize decision brief: {(error as Error).message}</span>
        </div>
      )}

      {brief && (
        <div className="space-y-8">
          {/* SECTION 1: Executive Decision Brief & Status */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
              <div className="flex items-center gap-3">
                <span className="text-xs font-mono px-2.5 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700">
                  {brief.decision_id}
                </span>
                <span className="text-sm font-semibold text-slate-200">
                  {brief.context.crop} • {brief.context.state}{brief.context.district ? `, ${brief.context.district}` : ''} • Year {brief.context.year}
                </span>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-400">Evidence Completeness:</span>
                <span className={`text-xs font-bold px-2.5 py-0.5 rounded border ${
                  completeness === 'STRONG_EVIDENCE'
                    ? 'bg-emerald-950/80 text-emerald-300 border-emerald-700/60'
                    : completeness === 'PARTIAL_EVIDENCE'
                    ? 'bg-blue-950/80 text-blue-300 border-blue-700/60'
                    : 'bg-amber-950/80 text-amber-300 border-amber-700/60'
                }`}>
                  {completeness}
                </span>
              </div>
            </div>

            {/* Quick Metrics Banner */}
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
              <div className="bg-slate-950/80 border border-slate-800/80 rounded-lg p-3 space-y-1">
                <span className="text-[10px] uppercase font-bold text-slate-400">Governed Forecast</span>
                <div className="text-lg font-bold text-emerald-400">
                  {forecast?.forecast_yield_kg_ha?.toFixed(1) ?? 'N/A'} <span className="text-xs text-slate-500 font-normal">kg/ha</span>
                </div>
                <span className="text-[10px] text-slate-500 font-mono">PREDICTED</span>
              </div>

              <div className="bg-slate-950/80 border border-slate-800/80 rounded-lg p-3 space-y-1">
                <span className="text-[10px] uppercase font-bold text-slate-400">Certified Strategy</span>
                <div className="text-xs font-semibold text-slate-200 truncate" title={forecast?.strategy}>
                  {forecast?.strategy ?? 'Historical Baseline'}
                </div>
                <span className="text-[10px] text-emerald-400 font-mono">
                  {forecast?.certification_status ?? 'BASELINE_PRODUCTION'}
                </span>
              </div>

              <div className="bg-slate-950/80 border border-slate-800/80 rounded-lg p-3 space-y-1">
                <span className="text-[10px] uppercase font-bold text-slate-400">Historical Mean</span>
                <div className="text-lg font-bold text-slate-200">
                  {hist?.historical_mean_yield_kg_ha?.toFixed(1) ?? 'N/A'} <span className="text-xs text-slate-500 font-normal">kg/ha</span>
                </div>
                <span className="text-[10px] text-slate-500 font-mono">HISTORICAL_REF</span>
              </div>

              <div className="bg-slate-950/80 border border-slate-800/80 rounded-lg p-3 space-y-1">
                <span className="text-[10px] uppercase font-bold text-slate-400">Validation MAE</span>
                <div className="text-lg font-bold text-blue-400">
                  {val?.mae_kg_ha?.toFixed(1) ?? 'N/A'} <span className="text-xs text-slate-500 font-normal">kg/ha</span>
                </div>
                <span className="text-[10px] text-slate-500 font-mono">VALIDATION</span>
              </div>

              <div className="bg-slate-950/80 border border-slate-800/80 rounded-lg p-3 space-y-1">
                <span className="text-[10px] uppercase font-bold text-slate-400">Uncertainty Spread</span>
                <div className="text-xs font-semibold text-slate-200">
                  {unc?.is_available ? `±${((unc.ensemble_spread_kg_ha || 0) / 2).toFixed(1)} kg/ha` : 'NOT_AVAILABLE'}
                </div>
                <span className="text-[10px] text-slate-500 font-mono">{unc?.is_available ? 'DERIVED (P10-P90)' : 'BASELINE'}</span>
              </div>

              <div className="bg-slate-950/80 border border-slate-800/80 rounded-lg p-3 space-y-1">
                <span className="text-[10px] uppercase font-bold text-slate-400">Monitoring Drift</span>
                <div className="text-xs font-semibold text-slate-200">
                  PSI: {mon?.prediction_drift_psi?.toFixed(4) ?? '0.0000'}
                </div>
                <span className="text-[10px] text-emerald-400 font-mono">{mon?.monitoring_status ?? 'HEALTHY'}</span>
              </div>
            </div>

            {/* Executive Summary Prose */}
            <div className="bg-slate-950/50 border border-slate-800/60 rounded-lg p-4 space-y-2 text-xs leading-relaxed text-slate-300">
              <p><strong className="text-slate-100">Baseline Context:</strong> {brief.executive_summary.current_status}</p>
              <p><strong className="text-slate-100">Forecast Outlook:</strong> {brief.executive_summary.outlook}</p>
              <p><strong className="text-slate-100">Monitoring & Signals:</strong> {brief.executive_summary.major_risk_signal}</p>
              <p><strong className="text-slate-100">Reliability Context:</strong> {brief.executive_summary.reliability_note}</p>
              <p className="text-slate-400 text-[11px] italic border-t border-slate-800/60 pt-2">
                <strong className="text-slate-300 not-italic">Scientific Guard:</strong> {brief.executive_summary.limitation_note}
              </p>
            </div>
          </div>

          {/* SECTION 2 & 3: Why This Forecast? and Historical Context */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Why This Forecast? */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-emerald-400" />
                  <h2 className="text-sm font-bold text-slate-100 uppercase tracking-wider">Why This Forecast?</h2>
                </div>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                  {forecast?.strategy ? 'GOVERNED ROUTER' : 'BASELINE PERSISTENCE'}
                </span>
              </div>

              <div className="space-y-3 text-xs">
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80 space-y-1.5">
                  <div className="flex justify-between text-slate-300">
                    <span className="font-semibold">Strategy Selected:</span>
                    <span className="text-emerald-400 font-mono font-bold">{forecast?.strategy}</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Certification Tier:</span>
                    <span className="text-slate-200">{forecast?.certification_status}</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Underlying Model:</span>
                    <span className="text-slate-200">{forecast?.model_name} (v{forecast?.model_version})</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Lineage Fingerprint:</span>
                    <span className="font-mono text-[10px] text-slate-300 truncate max-w-[200px]" title={forecast?.provenance_hash}>
                      {forecast?.provenance_hash}
                    </span>
                  </div>
                </div>

                {/* Model Attribution / Tree SHAP */}
                <div className="space-y-2">
                  <span className="text-[11px] font-bold text-slate-300 uppercase tracking-wider">
                    Feature Influence & Attributions
                  </span>
                  {brief.attribution_evidence && brief.attribution_evidence.length > 0 ? (
                    <div className="space-y-2">
                      {brief.attribution_evidence.map((att, idx) => (
                        <div key={idx} className="p-2.5 rounded bg-slate-950/70 border border-slate-800/60 space-y-1">
                          <div className="flex items-center justify-between">
                            <span className="font-medium text-slate-200">{att.feature_label}</span>
                            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
                              {att.attribution_type}
                            </span>
                          </div>
                          <p className="text-[11px] text-slate-400">{att.interpretation}</p>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-slate-500 text-[11px]">
                      Model feature importance not applicable for baseline persistence strategies.
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Historical Context */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
                <div className="flex items-center gap-2">
                  <Clock className="w-4 h-4 text-blue-400" />
                  <h2 className="text-sm font-bold text-slate-100 uppercase tracking-wider">Historical Context</h2>
                </div>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                  {hist?.start_year ?? 1966}–{hist?.end_year ?? 2016}
                </span>
              </div>

              <div className="space-y-3 text-xs">
                <div className="grid grid-cols-2 gap-2 text-center">
                  <div className="p-2.5 bg-slate-950 rounded-lg border border-slate-800/80">
                    <span className="text-[10px] text-slate-400 uppercase font-semibold">Empirical Range</span>
                    <div className="font-bold text-slate-200 mt-0.5">
                      {hist?.historical_min_yield_kg_ha?.toFixed(0) ?? '—'} – {hist?.historical_max_yield_kg_ha?.toFixed(0) ?? '—'} <span className="text-[10px] text-slate-500 font-normal">kg/ha</span>
                    </div>
                  </div>
                  <div className="p-2.5 bg-slate-950 rounded-lg border border-slate-800/80">
                    <span className="text-[10px] text-slate-400 uppercase font-semibold">Trajectory Slope</span>
                    <div className="font-bold text-slate-200 mt-0.5">
                      {(hist?.trend_slope_kg_ha_yr || 0) > 0 ? `+${hist?.trend_slope_kg_ha_yr?.toFixed(2)}` : hist?.trend_slope_kg_ha_yr?.toFixed(2)} <span className="text-[10px] text-slate-500 font-normal">kg/ha/yr</span>
                    </div>
                  </div>
                </div>

                {/* Recent Pre-Origin Observations Table */}
                <div className="space-y-1.5">
                  <span className="text-[11px] font-bold text-slate-300 uppercase tracking-wider">
                    Recent Panel Observations strictly preceding {selectedYear}
                  </span>
                  <div className="overflow-x-auto border border-slate-800 rounded-lg">
                    <table className="w-full text-left text-[11px]">
                      <thead className="bg-slate-950 text-slate-400 border-b border-slate-800">
                        <tr>
                          <th className="py-1.5 px-2.5">Year</th>
                          <th className="py-1.5 px-2.5">Observed Yield</th>
                          <th className="py-1.5 px-2.5">Harvest Area</th>
                          <th className="py-1.5 px-2.5">Classification</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 text-slate-300">
                        {hist?.recent_observations && hist.recent_observations.length > 0 ? (
                          hist.recent_observations.map((pt, i) => (
                            <tr key={i} className="hover:bg-slate-800/30">
                              <td className="py-1.5 px-2.5 font-mono">{pt.year}</td>
                              <td className="py-1.5 px-2.5 font-bold text-emerald-400">{pt.observed_yield_kg_ha} kg/ha</td>
                              <td className="py-1.5 px-2.5">{pt.observed_area_ha ? `${pt.observed_area_ha.toLocaleString()} ha` : '—'}</td>
                              <td className="py-1.5 px-2.5 font-mono text-[10px] text-slate-400">{pt.semantic_type}</td>
                            </tr>
                          ))
                        ) : (
                          <tr>
                            <td colSpan={4} className="py-3 text-center text-slate-500">
                              No prior historical observations found in regional panel.
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* SECTION 4, 5, 6: Forecast Reliability, Uncertainty, and Monitoring Signals */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Section 4: Forecast Reliability */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-3 text-xs">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="font-bold text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
                  <Activity className="w-4 h-4 text-blue-400" />
                  Reliability & Validation
                </span>
                <span className="text-[10px] font-mono text-slate-500">VALIDATION</span>
              </div>
              <div className="space-y-2 text-slate-300">
                <div className="flex justify-between">
                  <span className="text-slate-400">Protocol:</span>
                  <span className="font-medium text-slate-200">{val?.validation_protocol}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Validation Period:</span>
                  <span className="font-mono text-slate-200">{val?.validation_period}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Strategy MAE:</span>
                  <span className="font-bold text-emerald-400">{val?.mae_kg_ha?.toFixed(1)} kg/ha</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Baseline MAE:</span>
                  <span className="text-slate-300">{val?.baseline_mae_kg_ha?.toFixed(1)} kg/ha</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Fold Win Rate:</span>
                  <span className="font-bold text-slate-200">{val?.fold_win_rate_pct?.toFixed(0)}%</span>
                </div>
                {val?.legacy_benchmark_note && (
                  <div className="p-2 rounded bg-amber-950/30 border border-amber-800/40 text-[10px] text-amber-300 mt-2">
                    {val.legacy_benchmark_note}
                  </div>
                )}
              </div>
            </div>

            {/* Section 5: Uncertainty */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-3 text-xs">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="font-bold text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
                  <Layers className="w-4 h-4 text-purple-400" />
                  Uncertainty Evidence
                </span>
                <span className="text-[10px] font-mono text-slate-500">P10-P90 SPREAD</span>
              </div>
              {unc?.is_available ? (
                <div className="space-y-2.5 text-slate-300">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Empirical P10:</span>
                    <span className="font-mono font-bold text-slate-200">{unc.empirical_p10_kg_ha?.toFixed(1)} kg/ha</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Empirical P90:</span>
                    <span className="font-mono font-bold text-slate-200">{unc.empirical_p90_kg_ha?.toFixed(1)} kg/ha</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Ensemble Spread:</span>
                    <span className="font-bold text-purple-300">±{((unc.ensemble_spread_kg_ha || 0) / 2).toFixed(1)} kg/ha</span>
                  </div>
                  <div className="p-2 rounded bg-slate-950 border border-slate-800 text-[11px] text-slate-400 italic">
                    {unc.disclaimer}
                  </div>
                </div>
              ) : (
                <div className="space-y-2 text-slate-400">
                  <div className="p-3 rounded bg-slate-950 border border-slate-800/80 text-center">
                    <span className="font-semibold text-slate-300">NOT_AVAILABLE</span>
                    <p className="text-[11px] text-slate-500 mt-1">
                      Uncertainty estimates are not available for statistical baseline strategies.
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Section 6: Monitoring Signals */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-3 text-xs">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="font-bold text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
                  <TrendingUp className="w-4 h-4 text-emerald-400" />
                  Day 30 Monitoring Signals
                </span>
                <span className="text-[10px] font-mono text-slate-500">MONITORING</span>
              </div>
              <div className="space-y-2 text-slate-300">
                <div className="flex justify-between">
                  <span className="text-slate-400">Prediction Drift (PSI):</span>
                  <span className="font-mono font-bold text-slate-200">{mon?.prediction_drift_psi?.toFixed(4) ?? '0.0000'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Post-Outcome Status:</span>
                  <span className={`font-mono text-[10px] px-1.5 py-0.5 rounded ${
                    mon?.post_outcome_evaluation_status === 'EVALUATION_AVAILABLE'
                      ? 'bg-emerald-950 text-emerald-300'
                      : 'bg-amber-950 text-amber-300'
                  }`}>
                    {mon?.post_outcome_evaluation_status}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Active Alerts:</span>
                  <span className="font-bold text-slate-200">{mon?.active_alerts_count ?? 0}</span>
                </div>
                {selectedYear > 2017 && (
                  <div className="p-2 rounded bg-amber-950/30 border border-amber-800/40 text-[10px] text-amber-300">
                    Post-outcome evaluation is unavailable because harvest outcomes for year {selectedYear} are not yet observed in historical ground truth.
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* SECTION 7: Normalized Evidence Matrix */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-emerald-400" />
                <h2 className="text-sm font-bold text-slate-100 uppercase tracking-wider">
                  Auditable Evidence Matrix ({filteredEvidence.length} items)
                </h2>
              </div>

              {/* Category Filter Tabs */}
              <div className="flex flex-wrap gap-1.5 text-[10px]">
                {['ALL', 'OBSERVED', 'PREDICTED', 'VALIDATION', 'DERIVED', 'MONITORING', 'MODEL_ATTRIBUTION'].map(f => (
                  <button
                    key={f}
                    onClick={() => setActiveEvidenceFilter(f)}
                    className={`px-2 py-1 rounded transition font-medium ${
                      activeEvidenceFilter === f
                        ? 'bg-emerald-600 text-white font-bold'
                        : 'bg-slate-800 hover:bg-slate-700 text-slate-300'
                    }`}
                  >
                    {f}
                  </button>
                ))}
              </div>
            </div>

            <div className="overflow-x-auto border border-slate-800 rounded-lg">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950 text-slate-400 border-b border-slate-800 font-semibold text-[11px]">
                  <tr>
                    <th className="py-2 px-3">Evidence ID</th>
                    <th className="py-2 px-3">Classification</th>
                    <th className="py-2 px-3">Statement & Metric</th>
                    <th className="py-2 px-3">Value</th>
                    <th className="py-2 px-3">Period</th>
                    <th className="py-2 px-3">Source Module</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  {filteredEvidence.map((ev, i) => (
                    <tr key={i} className="hover:bg-slate-800/30">
                      <td className="py-2 px-3 font-mono text-[11px] text-slate-400">{ev.evidence_id}</td>
                      <td className="py-2 px-3">
                        <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                          ev.evidence_type === 'OBSERVED'
                            ? 'bg-blue-950 text-blue-300 border-blue-800/50'
                            : ev.evidence_type === 'PREDICTED'
                            ? 'bg-emerald-950 text-emerald-300 border-emerald-800/50'
                            : ev.evidence_type === 'VALIDATION'
                            ? 'bg-purple-950 text-purple-300 border-purple-800/50'
                            : ev.evidence_type === 'MONITORING'
                            ? 'bg-teal-950 text-teal-300 border-teal-800/50'
                            : 'bg-slate-800 text-slate-300 border-slate-700'
                        }`}>
                          {ev.evidence_type}
                        </span>
                      </td>
                      <td className="py-2 px-3">
                        <p className="font-medium text-slate-200">{ev.statement}</p>
                        {ev.limitation && <p className="text-[10px] text-slate-500 mt-0.5">Limitation: {ev.limitation}</p>}
                      </td>
                      <td className="py-2 px-3 font-mono font-bold text-slate-100 whitespace-nowrap">
                        {typeof ev.value === 'number' ? ev.value : String(ev.value)} <span className="text-[10px] font-normal text-slate-400">{ev.unit}</span>
                      </td>
                      <td className="py-2 px-3 text-[11px] text-slate-400 whitespace-nowrap">{ev.period || '—'}</td>
                      <td className="py-2 px-3 text-[11px] text-slate-400 whitespace-nowrap">{ev.source_module}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* SECTION 8: Simulated Scenario Options (Non-Causal) */}
          {brief.decision_options && brief.decision_options.length > 0 && (
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center gap-2">
                  <Layers className="w-4 h-4 text-purple-400" />
                  <h2 className="text-sm font-bold text-slate-100 uppercase tracking-wider">
                    Simulated Scenario Comparisons (Day 10 Engine)
                  </h2>
                </div>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800/50">
                  SIMULATED ONLY • NON-PRESCRIPTIVE
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {brief.decision_options.map((opt, idx) => (
                  <div key={idx} className="bg-slate-950/80 border border-slate-800 rounded-lg p-4 space-y-2.5 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-slate-200">{opt.title}</span>
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
                        {opt.scenario_type}
                      </span>
                    </div>

                    <div className="text-lg font-bold text-slate-100">
                      {opt.projected_yield_kg_ha?.toFixed(1)} <span className="text-xs text-slate-500 font-normal">kg/ha projected</span>
                    </div>

                    <div className="space-y-1 text-[11px] text-slate-400">
                      <div className="flex justify-between">
                        <span>Projected Delta:</span>
                        <span className={(opt.projected_yield_delta_kg_ha || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                          {(opt.projected_yield_delta_kg_ha || 0) >= 0 ? `+${opt.projected_yield_delta_kg_ha?.toFixed(1)}` : opt.projected_yield_delta_kg_ha?.toFixed(1)} kg/ha
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Tradeoffs:</span>
                        <span className="text-slate-300">{opt.tradeoffs}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Reliability:</span>
                        <span className="text-slate-300">{opt.model_reliability}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* SECTION 9: Assumptions & Limitations (Always Visible) */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-3 text-xs">
              <div className="flex items-center gap-2 border-b border-slate-800 pb-2.5">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <h3 className="font-bold text-slate-100 uppercase tracking-wider">Operating Assumptions</h3>
              </div>
              <ul className="space-y-2 text-slate-300 list-disc list-inside">
                {(brief.assumptions || []).map((asmp, i) => (
                  <li key={i} className="text-slate-300 leading-relaxed">{asmp}</li>
                ))}
              </ul>
            </div>

            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-3 text-xs">
              <div className="flex items-center gap-2 border-b border-slate-800 pb-2.5">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                <h3 className="font-bold text-slate-100 uppercase tracking-wider">Methodological Limitations</h3>
              </div>
              <ul className="space-y-2 text-slate-300 list-disc list-inside">
                {(brief.limitations || []).map((lmt, i) => (
                  <li key={i} className="text-slate-300 leading-relaxed">{lmt}</li>
                ))}
              </ul>
            </div>
          </div>

          {/* Footer Disclaimer */}
          <div className="text-center text-[11px] text-slate-500 py-4 border-t border-slate-800/80 leading-relaxed">
            {brief.footer_disclaimer}
          </div>
        </div>
      )}
    </div>
  )
}
