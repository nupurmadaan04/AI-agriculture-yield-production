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
    const md = `# Agricultural Decision Evidence Brief

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
    <div className="min-h-screen bg-background text-foreground p-6 lg:p-8 space-y-8 font-sans">
      {/* 1. Header & Metadata */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border pb-5">
        <div className="space-y-1.5">
          <div className="flex items-center gap-3">
            <span className="p-2 rounded-lg bg-primary/10 border border-primary/20 text-primary">
              <Compass className="w-5 h-5" />
            </span>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl lg:text-2xl font-bold tracking-tight text-foreground">
                  Decision Intelligence & Evidence-Based Forecast Brief
                </h1>
              </div>
              <p className="text-xs text-muted-foreground">
                Multi-layer synthesis • Walk-forward validation • Tree SHAP • Monitoring signals • Strictly non-causal decision support
              </p>
            </div>
          </div>
        </div>

        {brief && (
          <div className="flex items-center gap-2.5">
            <button
              onClick={exportMarkdown}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-muted hover:bg-muted/80 text-foreground rounded-lg text-xs font-medium border border-border transition"
              title="Download report in Markdown format"
            >
              <Download className="w-3.5 h-3.5 text-primary" />
              <span>Export Brief (.md)</span>
            </button>
            <button
              onClick={exportJSON}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-muted hover:bg-muted/80 text-foreground rounded-lg text-xs font-medium border border-border transition"
              title="Download raw JSON evidence payload"
            >
              <Code className="w-3.5 h-3.5 text-blue-400" />
              <span>Raw JSON</span>
            </button>
          </div>
        )}
      </div>

      {/* 2. Decision Context Selector Bar */}
      <div className="bg-card border border-border rounded-xl p-4 shadow-sm flex flex-wrap items-end gap-3.5">
        <div className="space-y-1.5 min-w-[170px]">
          <label className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-1">
            <span>Crop Commodity</span>
            <span className="text-muted-foreground/70 font-normal">({MULTICROP_COMMODITIES.length})</span>
          </label>
          <select
            value={selectedCrop}
            onChange={e => setSelectedCrop(e.target.value)}
            className="w-full bg-background border border-border rounded-lg px-3 py-1.5 text-xs text-foreground focus:outline-none focus:border-primary"
          >
            {MULTICROP_COMMODITIES.map(c => (
              <option key={c.name} value={c.name}>
                {c.name} ({c.tier === 'PRODUCTION_READY' ? 'ML Certified' : c.tier === 'CONDITIONAL_PRODUCTION' ? 'Conditional ML' : 'Baseline'})
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-1.5 min-w-[160px]">
          <label className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">State</label>
          <select
            value={selectedState}
            onChange={e => setSelectedState(e.target.value)}
            className="w-full bg-background border border-border rounded-lg px-3 py-1.5 text-xs text-foreground focus:outline-none focus:border-primary"
          >
            {stateNames.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>

        <div className="space-y-1.5 min-w-[150px]">
          <label className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">District</label>
          <input
            type="text"
            placeholder="e.g. Ludhiana"
            value={district}
            onChange={e => setDistrict(e.target.value)}
            className="w-full bg-background border border-border rounded-lg px-3 py-1.5 text-xs text-foreground placeholder-muted-foreground focus:outline-none focus:border-primary"
          />
        </div>

        <div className="space-y-1.5 min-w-[120px]">
          <label className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">Forecast Year</label>
          <select
            value={selectedYear}
            onChange={e => setSelectedYear(Number(e.target.value))}
            className="w-full bg-background border border-border rounded-lg px-3 py-1.5 text-xs text-foreground focus:outline-none focus:border-primary"
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
          className="flex items-center gap-2 px-5 py-2 bg-primary hover:bg-primary/90 disabled:opacity-50 text-primary-foreground rounded-lg text-xs font-semibold shadow transition-colors"
        >
          {isPending ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-white" />}
          <span>{isPending ? 'Synthesizing...' : 'Synthesize Brief'}</span>
        </button>
      </div>

      {error && (
        <div className="bg-destructive/5 border border-destructive/20 rounded-xl p-4 text-xs text-destructive flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>Failed to synthesize decision brief: {(error as Error).message}</span>
        </div>
      )}

      {brief && (
        <div className="space-y-8">
          {/* SECTION 1: Executive Decision Brief & Status */}
          <div className="bg-card border border-border rounded-xl p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border pb-3">
              <div className="flex items-center gap-3">
                <span className="text-xs font-mono px-2.5 py-1 rounded bg-muted text-muted-foreground border border-border">
                  {brief.decision_id}
                </span>
                <span className="text-sm font-semibold text-foreground">
                  {brief.context.crop} • {brief.context.state}{brief.context.district ? `, ${brief.context.district}` : ''} • Year {brief.context.year}
                </span>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">Evidence Completeness:</span>
                <span className={`text-xs font-bold px-2.5 py-0.5 rounded border ${
                  completeness === 'STRONG_EVIDENCE'
                    ? 'bg-primary/10 text-primary border-primary/30'
                    : completeness === 'PARTIAL_EVIDENCE'
                    ? 'bg-blue-500/10 text-blue-600 dark:text-blue-300 border-blue-500/30'
                    : 'bg-amber-500/10 text-amber-700 dark:text-amber-300 border-amber-500/30'
                }`}>
                  {completeness}
                </span>
              </div>
            </div>

            {/* Quick Metrics Banner */}
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
              <div className="bg-muted/30 border border-border/60 rounded-lg p-3 space-y-1">
                <span className="text-[10px] uppercase font-bold text-muted-foreground">Governed Forecast</span>
                <div className="text-lg font-bold text-primary">
                  {forecast?.forecast_yield_kg_ha?.toFixed(1) ?? 'N/A'} <span className="text-xs text-muted-foreground/70 font-normal">kg/ha</span>
                </div>
                <span className="text-[10px] text-muted-foreground font-mono">PREDICTED</span>
              </div>

              <div className="bg-muted/30 border border-border/60 rounded-lg p-3 space-y-1">
                <span className="text-[10px] uppercase font-bold text-muted-foreground">Certified Strategy</span>
                <div className="text-xs font-semibold text-foreground truncate" title={forecast?.strategy}>
                  {forecast?.strategy ?? 'Historical Baseline'}
                </div>
                <span className="text-[10px] text-primary font-mono">
                  {forecast?.certification_status ?? 'BASELINE_PRODUCTION'}
                </span>
              </div>

              <div className="bg-muted/30 border border-border/60 rounded-lg p-3 space-y-1">
                <span className="text-[10px] uppercase font-bold text-muted-foreground">Historical Mean</span>
                <div className="text-lg font-bold text-foreground">
                  {hist?.historical_mean_yield_kg_ha?.toFixed(1) ?? 'N/A'} <span className="text-xs text-muted-foreground/70 font-normal">kg/ha</span>
                </div>
                <span className="text-[10px] text-muted-foreground font-mono">HISTORICAL_REF</span>
              </div>

              <div className="bg-muted/30 border border-border/60 rounded-lg p-3 space-y-1">
                <span className="text-[10px] uppercase font-bold text-muted-foreground">Validation MAE</span>
                <div className="text-lg font-bold text-sky-600 dark:text-blue-400">
                  {val?.mae_kg_ha?.toFixed(1) ?? 'N/A'} <span className="text-xs text-muted-foreground/70 font-normal">kg/ha</span>
                </div>
                <span className="text-[10px] text-muted-foreground font-mono">VALIDATION</span>
              </div>

              <div className="bg-muted/30 border border-border/60 rounded-lg p-3 space-y-1">
                <span className="text-[10px] uppercase font-bold text-muted-foreground">Uncertainty Spread</span>
                <div className="text-xs font-semibold text-foreground">
                  {unc?.is_available ? `±${((unc.ensemble_spread_kg_ha || 0) / 2).toFixed(1)} kg/ha` : 'NOT_AVAILABLE'}
                </div>
                <span className="text-[10px] text-muted-foreground font-mono">{unc?.is_available ? 'DERIVED (P10-P90)' : 'BASELINE'}</span>
              </div>

              <div className="bg-muted/30 border border-border/60 rounded-lg p-3 space-y-1">
                <span className="text-[10px] uppercase font-bold text-muted-foreground">Monitoring Drift</span>
                <div className="text-xs font-semibold text-foreground">
                  PSI: {mon?.prediction_drift_psi?.toFixed(4) ?? '0.0000'}
                </div>
                <span className="text-[10px] text-primary font-mono">{mon?.monitoring_status ?? 'HEALTHY'}</span>
              </div>
            </div>

            {/* Executive Summary Prose */}
            <div className="bg-muted/20 border border-border/50 rounded-lg p-4 space-y-2 text-xs leading-relaxed text-foreground/80">
              <p><strong className="text-foreground">Baseline Context:</strong> {brief.executive_summary.current_status}</p>
              <p><strong className="text-foreground">Forecast Outlook:</strong> {brief.executive_summary.outlook}</p>
              <p><strong className="text-foreground">Monitoring & Signals:</strong> {brief.executive_summary.major_risk_signal}</p>
              <p><strong className="text-foreground">Reliability Context:</strong> {brief.executive_summary.reliability_note}</p>
              <p className="text-muted-foreground text-[11px] italic border-t border-border/50 pt-2">
                <strong className="text-foreground/90 not-italic">Scientific Guard:</strong> {brief.executive_summary.limitation_note}
              </p>
            </div>
          </div>

          {/* SECTION 2 & 3: Why This Forecast? and Historical Context */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Why This Forecast? */}
            <div className="bg-card border border-border rounded-xl p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-border pb-2.5">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-primary" />
                  <h2 className="text-sm font-bold text-foreground uppercase tracking-wider">Why This Forecast?</h2>
                </div>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-muted text-muted-foreground">
                  {forecast?.strategy ? 'GOVERNED ROUTER' : 'BASELINE PERSISTENCE'}
                </span>
              </div>

              <div className="space-y-3 text-xs">
                <div className="p-3 rounded-lg bg-muted/30 border border-border/60 space-y-1.5">
                  <div className="flex justify-between text-foreground/80">
                    <span className="font-semibold">Strategy Selected:</span>
                    <span className="text-primary font-mono font-bold">{forecast?.strategy}</span>
                  </div>
                  <div className="flex justify-between text-muted-foreground">
                    <span>Certification Tier:</span>
                    <span className="text-foreground">{forecast?.certification_status}</span>
                  </div>
                  <div className="flex justify-between text-muted-foreground">
                    <span>Underlying Model:</span>
                    <span className="text-foreground">{forecast?.model_name} (v{forecast?.model_version})</span>
                  </div>
                  <div className="flex justify-between text-muted-foreground">
                    <span>Lineage Fingerprint:</span>
                    <span className="font-mono text-[10px] text-foreground/80 truncate max-w-[200px]" title={forecast?.provenance_hash}>
                      {forecast?.provenance_hash}
                    </span>
                  </div>
                </div>

                {/* Model Attribution / Tree SHAP */}
                <div className="space-y-2">
                  <span className="text-[11px] font-bold text-foreground/80 uppercase tracking-wider">
                    Feature Influence & Attributions
                  </span>
                  {brief.attribution_evidence && brief.attribution_evidence.length > 0 ? (
                    <div className="space-y-2">
                      {brief.attribution_evidence.map((att, idx) => (
                        <div key={idx} className="p-2.5 rounded bg-muted/20 border border-border/50 space-y-1">
                          <div className="flex items-center justify-between">
                            <span className="font-medium text-foreground">{att.feature_label}</span>
                            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-muted text-foreground/80">
                              {att.attribution_type}
                            </span>
                          </div>
                          <p className="text-[11px] text-muted-foreground">{att.interpretation}</p>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-muted-foreground text-[11px]">
                      Model feature importance not applicable for baseline persistence strategies.
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Historical Context */}
            <div className="bg-card border border-border rounded-xl p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-border pb-2.5">
                <div className="flex items-center gap-2">
                  <Clock className="w-4 h-4 text-blue-400" />
                  <h2 className="text-sm font-bold text-foreground uppercase tracking-wider">Historical Context</h2>
                </div>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-muted text-muted-foreground">
                  {hist?.start_year ?? 1966}–{hist?.end_year ?? 2016}
                </span>
              </div>

              <div className="space-y-3 text-xs">
                <div className="grid grid-cols-2 gap-2 text-center">
                  <div className="p-2.5 bg-muted/30 rounded-lg border border-border/60">
                    <span className="text-[10px] text-muted-foreground uppercase font-semibold">Empirical Range</span>
                    <div className="font-bold text-foreground mt-0.5">
                      {hist?.historical_min_yield_kg_ha?.toFixed(0) ?? '—'} – {hist?.historical_max_yield_kg_ha?.toFixed(0) ?? '—'} <span className="text-[10px] text-muted-foreground/70 font-normal">kg/ha</span>
                    </div>
                  </div>
                  <div className="p-2.5 bg-muted/30 rounded-lg border border-border/60">
                    <span className="text-[10px] text-muted-foreground uppercase font-semibold">Trajectory Slope</span>
                    <div className="font-bold text-foreground mt-0.5">
                      {(hist?.trend_slope_kg_ha_yr || 0) > 0 ? `+${hist?.trend_slope_kg_ha_yr?.toFixed(2)}` : hist?.trend_slope_kg_ha_yr?.toFixed(2)} <span className="text-[10px] text-muted-foreground/70 font-normal">kg/ha/yr</span>
                    </div>
                  </div>
                </div>

                {/* Recent Pre-Origin Observations Table */}
                <div className="space-y-1.5">
                  <span className="text-[11px] font-bold text-foreground/80 uppercase tracking-wider">
                    Recent Panel Observations strictly preceding {selectedYear}
                  </span>
                  <div className="overflow-x-auto border border-border rounded-lg">
                    <table className="w-full text-left text-[11px]">
                      <thead className="bg-background/80 text-muted-foreground border-b border-border">
                        <tr>
                          <th className="py-1.5 px-2.5">Year</th>
                          <th className="py-1.5 px-2.5">Observed Yield</th>
                          <th className="py-1.5 px-2.5">Harvest Area</th>
                          <th className="py-1.5 px-2.5">Classification</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border/50 text-foreground/80">
                        {hist?.recent_observations && hist.recent_observations.length > 0 ? (
                          hist.recent_observations.map((pt, i) => (
                            <tr key={i} className="hover:bg-muted/30">
                              <td className="py-1.5 px-2.5 font-mono">{pt.year}</td>
                              <td className="py-1.5 px-2.5 font-bold text-primary">{pt.observed_yield_kg_ha} kg/ha</td>
                              <td className="py-1.5 px-2.5">{pt.observed_area_ha ? `${pt.observed_area_ha.toLocaleString()} ha` : '—'}</td>
                              <td className="py-1.5 px-2.5 font-mono text-[10px] text-muted-foreground">{pt.semantic_type}</td>
                            </tr>
                          ))
                        ) : (
                          <tr>
                            <td colSpan={4} className="py-3 text-center text-muted-foreground">
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
            <div className="bg-card border border-border rounded-xl p-5 space-y-3 text-xs">
              <div className="flex items-center justify-between border-b border-border pb-2">
                <span className="font-bold text-foreground uppercase tracking-wider flex items-center gap-1.5">
                  <Activity className="w-4 h-4 text-blue-400" />
                  Reliability & Validation
                </span>
                <span className="text-[10px] font-mono text-muted-foreground">VALIDATION</span>
              </div>
              <div className="space-y-2 text-foreground/80">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Protocol:</span>
                  <span className="font-medium text-foreground">{val?.validation_protocol}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Validation Period:</span>
                  <span className="font-mono text-foreground">{val?.validation_period}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Strategy MAE:</span>
                  <span className="font-bold text-primary">{val?.mae_kg_ha?.toFixed(1)} kg/ha</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Baseline MAE:</span>
                  <span className="text-foreground/80">{val?.baseline_mae_kg_ha?.toFixed(1)} kg/ha</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Fold Win Rate:</span>
                  <span className="font-bold text-foreground">{val?.fold_win_rate_pct?.toFixed(0)}%</span>
                </div>
                {val?.legacy_benchmark_note && (
                  <div className="p-2 rounded bg-amber-500/10 border border-amber-500/20 text-[10px] text-amber-700 dark:text-amber-300 mt-2">
                    {val.legacy_benchmark_note}
                  </div>
                )}
              </div>
            </div>

            {/* Section 5: Uncertainty */}
            <div className="bg-card border border-border rounded-xl p-5 space-y-3 text-xs">
              <div className="flex items-center justify-between border-b border-border pb-2">
                <span className="font-bold text-foreground uppercase tracking-wider flex items-center gap-1.5">
                  <Layers className="w-4 h-4 text-purple-400" />
                  Uncertainty Evidence
                </span>
                <span className="text-[10px] font-mono text-muted-foreground">P10-P90 SPREAD</span>
              </div>
              {unc?.is_available ? (
                <div className="space-y-2.5 text-foreground/80">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Empirical P10:</span>
                    <span className="font-mono font-bold text-foreground">{unc.empirical_p10_kg_ha?.toFixed(1)} kg/ha</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Empirical P90:</span>
                    <span className="font-mono font-bold text-foreground">{unc.empirical_p90_kg_ha?.toFixed(1)} kg/ha</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Ensemble Spread:</span>
                    <span className="font-bold text-purple-600 dark:text-purple-300">±{((unc.ensemble_spread_kg_ha || 0) / 2).toFixed(1)} kg/ha</span>
                  </div>
                  <div className="p-2 rounded bg-background/80 border border-border text-[11px] text-muted-foreground italic">
                    {unc.disclaimer}
                  </div>
                </div>
              ) : (
                <div className="space-y-2 text-muted-foreground">
                  <div className="p-3 rounded bg-muted/30 border border-border/60 text-center">
                    <span className="font-semibold text-foreground/80">NOT_AVAILABLE</span>
                    <p className="text-[11px] text-muted-foreground mt-1">
                      Uncertainty estimates are not available for statistical baseline strategies.
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Section 6: Monitoring Signals */}
            <div className="bg-card border border-border rounded-xl p-5 space-y-3 text-xs">
              <div className="flex items-center justify-between border-b border-border pb-2">
                <span className="font-bold text-foreground uppercase tracking-wider flex items-center gap-1.5">
                  <TrendingUp className="w-4 h-4 text-primary" />
                  Active Monitoring Signals
                </span>
                <span className="text-[10px] font-mono text-muted-foreground">MONITORING</span>
              </div>
              <div className="space-y-2 text-foreground/80">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Prediction Drift (PSI):</span>
                  <span className="font-mono font-bold text-foreground">{mon?.prediction_drift_psi?.toFixed(4) ?? '0.0000'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Post-Outcome Status:</span>
                  <span className={`font-mono text-[10px] px-1.5 py-0.5 rounded ${
                    mon?.post_outcome_evaluation_status === 'EVALUATION_AVAILABLE'
                      ? 'bg-primary/10 text-primary'
                      : 'bg-amber-500/10 text-amber-700 dark:text-amber-300'
                  }`}>
                    {mon?.post_outcome_evaluation_status}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Active Alerts:</span>
                  <span className="font-bold text-foreground">{mon?.active_alerts_count ?? 0}</span>
                </div>
                {selectedYear > 2017 && (
                  <div className="p-2 rounded bg-amber-500/10 border border-amber-500/20 text-[10px] text-amber-700 dark:text-amber-300">
                    Post-outcome evaluation is unavailable because harvest outcomes for year {selectedYear} are not yet observed in historical ground truth.
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* SECTION 7: Normalized Evidence Matrix */}
          <div className="bg-card border border-border rounded-xl p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border pb-3">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-primary" />
                <h2 className="text-sm font-bold text-foreground uppercase tracking-wider">
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
                        ? 'bg-primary text-primary-foreground font-bold'
                        : 'bg-muted hover:bg-muted/80 text-muted-foreground'
                    }`}
                  >
                    {f}
                  </button>
                ))}
              </div>
            </div>

            <div className="overflow-x-auto border border-border rounded-lg">
              <table className="w-full text-left text-xs">
                <thead className="bg-background/80 text-muted-foreground border-b border-border font-semibold text-[11px]">
                  <tr>
                    <th className="py-2 px-3">Evidence ID</th>
                    <th className="py-2 px-3">Classification</th>
                    <th className="py-2 px-3">Statement & Metric</th>
                    <th className="py-2 px-3">Value</th>
                    <th className="py-2 px-3">Period</th>
                    <th className="py-2 px-3">Source Module</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50 text-foreground/80">
                  {filteredEvidence.map((ev, i) => (
                    <tr key={i} className="hover:bg-muted/30">
                      <td className="py-2 px-3 font-mono text-[11px] text-muted-foreground">{ev.evidence_id}</td>
                      <td className="py-2 px-3">
                        <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                          ev.evidence_type === 'OBSERVED'
                            ? 'bg-blue-500/10 text-blue-600 dark:text-blue-300 border-blue-500/20'
                            : ev.evidence_type === 'PREDICTED'
                            ? 'bg-primary/10 text-primary border-primary/20'
                            : ev.evidence_type === 'VALIDATION'
                            ? 'bg-purple-500/10 text-purple-700 dark:text-purple-300 border-purple-500/20'
                            : ev.evidence_type === 'MONITORING'
                            ? 'bg-teal-500/10 text-teal-600 dark:text-teal-300 border-teal-500/20'
                            : 'bg-muted text-muted-foreground border-border'
                        }`}>
                          {ev.evidence_type}
                        </span>
                      </td>
                      <td className="py-2 px-3">
                        <p className="font-medium text-foreground">{ev.statement}</p>
                        {ev.limitation && <p className="text-[10px] text-muted-foreground mt-0.5">Limitation: {ev.limitation}</p>}
                      </td>
                      <td className="py-2 px-3 font-mono font-bold text-foreground whitespace-nowrap">
                        {typeof ev.value === 'number' ? ev.value : String(ev.value)} <span className="text-[10px] font-normal text-muted-foreground">{ev.unit}</span>
                      </td>
                      <td className="py-2 px-3 text-[11px] text-muted-foreground whitespace-nowrap">{ev.period || '—'}</td>
                      <td className="py-2 px-3 text-[11px] text-muted-foreground whitespace-nowrap">{ev.source_module}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* SECTION 8: Simulated Scenario Options (Non-Causal) */}
          {brief.decision_options && brief.decision_options.length > 0 && (
            <div className="bg-card border border-border rounded-xl p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-border pb-3">
                <div className="flex items-center gap-2">
                  <Layers className="w-4 h-4 text-purple-400" />
                  <h2 className="text-sm font-bold text-foreground uppercase tracking-wider">
                    Simulated Scenario Comparisons
                  </h2>
                </div>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-500/10 text-purple-700 dark:text-purple-300 border border-purple-500/20">
                  SIMULATED ONLY • NON-PRESCRIPTIVE
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {brief.decision_options.map((opt, idx) => (
                  <div key={idx} className="bg-muted/20 border border-border rounded-lg p-4 space-y-2.5 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-foreground">{opt.title}</span>
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                        {opt.scenario_type}
                      </span>
                    </div>

                    <div className="text-lg font-bold text-foreground">
                      {opt.projected_yield_kg_ha?.toFixed(1)} <span className="text-xs text-muted-foreground/70 font-normal">kg/ha projected</span>
                    </div>

                    <div className="space-y-1 text-[11px] text-muted-foreground">
                      <div className="flex justify-between">
                        <span>Projected Delta:</span>
                        <span className={(opt.projected_yield_delta_kg_ha || 0) >= 0 ? 'text-primary' : 'text-red-400'}>
                          {(opt.projected_yield_delta_kg_ha || 0) >= 0 ? `+${opt.projected_yield_delta_kg_ha?.toFixed(1)}` : opt.projected_yield_delta_kg_ha?.toFixed(1)} kg/ha
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Tradeoffs:</span>
                        <span className="text-foreground/80">{opt.tradeoffs}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Reliability:</span>
                        <span className="text-foreground/80">{opt.model_reliability}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* SECTION 9: Assumptions & Limitations (Always Visible) */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-card border border-border rounded-xl p-5 space-y-3 text-xs">
              <div className="flex items-center gap-2 border-b border-border pb-2.5">
                <CheckCircle2 className="w-4 h-4 text-primary" />
                <h3 className="font-bold text-foreground uppercase tracking-wider">Operating Assumptions</h3>
              </div>
              <ul className="space-y-2 text-foreground/80 list-disc list-inside">
                {(brief.assumptions || []).map((asmp, i) => (
                  <li key={i} className="text-foreground/80 leading-relaxed">{asmp}</li>
                ))}
              </ul>
            </div>

            <div className="bg-card border border-border rounded-xl p-5 space-y-3 text-xs">
              <div className="flex items-center gap-2 border-b border-border pb-2.5">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                <h3 className="font-bold text-foreground uppercase tracking-wider">Methodological Limitations</h3>
              </div>
              <ul className="space-y-2 text-foreground/80 list-disc list-inside">
                {(brief.limitations || []).map((lmt, i) => (
                  <li key={i} className="text-foreground/80 leading-relaxed">{lmt}</li>
                ))}
              </ul>
            </div>
          </div>

          {/* Footer Disclaimer */}
          <div className="text-center text-[11px] text-muted-foreground py-4 border-t border-border/50 leading-relaxed">
            {brief.footer_disclaimer}
          </div>
        </div>
      )}
    </div>
  )
}
