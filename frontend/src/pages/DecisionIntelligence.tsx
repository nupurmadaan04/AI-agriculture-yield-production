import React, { useState, useEffect } from 'react'
import { useDecisionAnalysis } from '../services/decisionService'
import { useStates } from '../services/api'
import { DecisionConfidenceMatrix } from '../components/decision/DecisionConfidenceMatrix'
import { ExecutiveDecisionBrief } from '../components/decision/ExecutiveDecisionBrief'
import { DecisionEvidenceTable } from '../components/decision/DecisionEvidenceTable'
import { DecisionSignalFusionPanel } from '../components/decision/DecisionSignalFusionPanel'
import { DecisionPriorityCard } from '../components/decision/DecisionPriorityCard'
import { DecisionOptionsPanel } from '../components/decision/DecisionOptionsPanel'
import { EvidenceProvenance } from '../components/decision/EvidenceProvenance'
import { DecisionAuditPanel } from '../components/decision/DecisionAuditPanel'
import {
  Compass,
  Play,
  Download,
  AlertCircle,
  FileText,
  ShieldCheck,
  RefreshCw,
  Sparkles
} from 'lucide-react'

export const DecisionIntelligence: React.FC = () => {
  const [selectedState, setSelectedState] = useState('Punjab')
  const [district, setDistrict] = useState('')
  const [selectedYear, setSelectedYear] = useState(2017)
  const [crop, setCrop] = useState('Rice')
  const [horizon, setHorizon] = useState('next_season')

  const { data: statesData } = useStates()
  const stateNames = statesData?.data?.map(s => s.state) || [
    'Punjab', 'Haryana', 'Andhra Pradesh', 'Uttar Pradesh', 'West Bengal', 'Tamil Nadu', 'Karnataka', 'Odisha'
  ]

  const { mutate: analyze, data: decisionRes, isPending, error } = useDecisionAnalysis()

  // Execute initial analysis on load
  useEffect(() => {
    analyze({
      crop,
      state: selectedState,
      district: district || undefined,
      year: selectedYear,
      decision_horizon: horizon
    })
  }, [])

  const handleAnalyze = () => {
    analyze({
      crop,
      state: selectedState,
      district: district || undefined,
      year: selectedYear,
      decision_horizon: horizon
    })
  }

  const exportMarkdown = () => {
    if (!decisionRes?.brief) return
    const md = `# Agricultural Decision Evidence Report\n\nDecision ID: ${decisionRes.decision_id}\nTarget: ${selectedState} (${district || 'Statewide'})\n\n${decisionRes.brief.executive_summary.current_status}\n${decisionRes.brief.executive_summary.outlook}\n\n## Priorities\n${decisionRes.brief.analytical_priorities.map(p => `* ${p.priority_rank}. ${p.issue} (${p.priority_level})`).join('\n')}\n\n---\n${decisionRes.brief.footer_disclaimer}`
    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.setAttribute('download', `DECISION_REPORT_${decisionRes.decision_id}.md`)
    document.body.appendChild(link)
    link.click()
    link.remove()
  }

  const brief = decisionRes?.brief

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 lg:p-8 space-y-6">
      {/* Top Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div className="space-y-1">
          <div className="flex items-center gap-2.5">
            <Compass className="w-6 h-6 text-emerald-400" />
            <h1 className="text-xl lg:text-2xl font-bold tracking-tight text-slate-100">
              Agricultural Decision Intelligence & Evidence Synthesis
            </h1>
          </div>
          <p className="text-xs text-slate-400">
            Multi-Layer Orchestration • Forecast • Geospatial Risk • Temporal Monitoring • Reliability • XAI • Scenario Optimization
          </p>
        </div>

        <div className="flex items-center gap-2">
          {brief && (
            <button
              onClick={exportMarkdown}
              className="flex items-center gap-1.5 px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-medium border border-slate-700 transition-colors"
            >
              <Download className="w-4 h-4 text-emerald-400" />
              <span>Export Decision Report</span>
            </button>
          )}
        </div>
      </div>

      {/* Decision Context Control Bar */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 shadow-lg flex flex-wrap items-end gap-3.5">
        <div className="space-y-1.5 min-w-[140px]">
          <label className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Crop</label>
          <select
            value={crop}
            onChange={e => setCrop(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-emerald-500"
          >
            <option value="Rice">Rice (Kharif)</option>
            <option value="Wheat">Wheat (Rabi Benchmark)</option>
          </select>
        </div>

        <div className="space-y-1.5 min-w-[160px]">
          <label className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">State</label>
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

        <div className="space-y-1.5 min-w-[140px]">
          <label className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">District (Optional)</label>
          <input
            type="text"
            placeholder="e.g. Ludhiana"
            value={district}
            onChange={e => setDistrict(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-emerald-500"
          />
        </div>

        <div className="space-y-1.5 min-w-[110px]">
          <label className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Year</label>
          <select
            value={selectedYear}
            onChange={e => setSelectedYear(Number(e.target.value))}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-emerald-500"
          >
            {[2018, 2017, 2016, 2015].map(y => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </div>

        <div className="space-y-1.5 min-w-[130px]">
          <label className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Horizon</label>
          <select
            value={horizon}
            onChange={e => setHorizon(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-emerald-500"
          >
            <option value="next_season">Next Season</option>
            <option value="multi_year">Multi-Year</option>
          </select>
        </div>

        <button
          onClick={handleAnalyze}
          disabled={isPending}
          className="flex items-center gap-2 px-5 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold shadow transition-colors"
        >
          {isPending ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-white" />}
          <span>{isPending ? 'Synthesizing...' : 'Analyze Decision'}</span>
        </button>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-xs text-red-400 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>Failed to synthesize decision intelligence: {(error as Error).message}</span>
        </div>
      )}

      {brief && (
        <div className="space-y-6">
          {/* 5-Dimension Confidence Matrix */}
          <DecisionConfidenceMatrix status={brief.evidence_status} />

          {/* Executive Decision Brief */}
          <ExecutiveDecisionBrief brief={brief} />

          {/* Signals & Priorities */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <DecisionSignalFusionPanel signals={brief.signals} />
            <DecisionPriorityCard priorities={brief.analytical_priorities} />
          </div>

          {/* Scenario Decision Options & Robustness */}
          <DecisionOptionsPanel
            options={brief.decision_options}
            robustness={brief.robustness}
          />

          {/* Normalized Evidence Table */}
          <DecisionEvidenceTable items={brief.evidence_items} />

          {/* Lineage Provenance DAG */}
          <EvidenceProvenance provenance={brief.provenance} />

          {/* Cryptographic Audit Certificate */}
          <DecisionAuditPanel audit={brief.audit_record} />

          {/* Footer Disclaimer */}
          <div className="text-center text-[11px] text-slate-500 py-4 border-t border-slate-800/80 leading-relaxed">
            {brief.footer_disclaimer}
          </div>
        </div>
      )}
    </div>
  )
}
