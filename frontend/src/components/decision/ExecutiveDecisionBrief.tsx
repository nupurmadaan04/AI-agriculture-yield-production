import React, { useState } from 'react'
import type { DecisionBrief } from '../../types/decision'
import { FileText, ChevronDown, ChevronUp, AlertTriangle, ShieldCheck, Target, Sparkles, BookOpen } from 'lucide-react'

interface Props {
  brief: DecisionBrief
}

const CLASSIFICATION_BADGES: Record<string, string> = {
  FACT: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
  'MODEL OUTPUT': 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30',
  SIMULATION: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/30',
  INTERPRETATION: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
  DERIVED: 'bg-purple-500/10 text-purple-400 border-purple-500/30',
  VALIDATION: 'bg-blue-500/10 text-blue-400 border-blue-500/30'
}

export const ExecutiveDecisionBrief: React.FC<Props> = ({ brief }) => {
  const [expandedSections, setExpandedSections] = useState<Record<number, boolean>>({
    1: true,
    2: true,
    5: true,
    6: true,
    12: true
  })

  const toggleSection = (num: number) => {
    setExpandedSections(prev => ({
      ...prev,
      [num]: !prev[num]
    }))
  }

  const exec = brief.executive_summary

  return (
    <div className="space-y-6">
      {/* Executive KPI Banner */}
      <div className="bg-gradient-to-br from-slate-900 via-slate-900/90 to-slate-950 border border-slate-800 rounded-xl p-6 shadow-xl space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-4">
          <div className="flex items-center gap-2.5">
            <Sparkles className="w-5 h-5 text-emerald-400" />
            <div>
              <h2 className="text-base font-bold text-slate-100">Executive Decision Synthesis</h2>
              <div className="text-xs text-slate-400">
                Target: {brief.context.state} {brief.context.district ? `(${brief.context.district})` : ''} • Crop: {brief.context.crop} • Year: {brief.context.year}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono px-3 py-1 bg-emerald-500/10 border border-emerald-500/30 rounded-lg text-emerald-400">
              {brief.decision_id}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="space-y-3">
            <div className="bg-slate-950/60 rounded-lg p-3 border border-slate-800/80">
              <div className="text-slate-400 font-medium mb-1">Current Status</div>
              <div className="text-slate-200 leading-relaxed">{exec.current_status}</div>
            </div>
            <div className="bg-slate-950/60 rounded-lg p-3 border border-slate-800/80">
              <div className="text-slate-400 font-medium mb-1">Forecast Outlook</div>
              <div className="text-cyan-300 leading-relaxed font-medium">{exec.outlook}</div>
            </div>
          </div>

          <div className="space-y-3">
            <div className="bg-slate-950/60 rounded-lg p-3 border border-slate-800/80">
              <div className="text-slate-400 font-medium mb-1">Major Risk Signal</div>
              <div className="text-amber-300 leading-relaxed">{exec.major_risk_signal}</div>
            </div>
            <div className="bg-slate-950/60 rounded-lg p-3 border border-slate-800/80">
              <div className="text-slate-400 font-medium mb-1">Top Analytical Priority</div>
              <div className="text-emerald-300 font-semibold leading-relaxed">{exec.highest_priority_issue}</div>
            </div>
          </div>
        </div>
      </div>

      {/* 16-Section Accordion */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-3">
        <div className="flex items-center gap-2 mb-2">
          <BookOpen className="w-4 h-4 text-cyan-400" />
          <h3 className="text-sm font-semibold text-slate-100">16-Section Structured Agricultural Decision Brief</h3>
        </div>

        <div className="space-y-2">
          {brief.sections.map(sec => {
            const isExpanded = !!expandedSections[sec.section_number]
            return (
              <div
                key={sec.section_number}
                className="border border-slate-800/80 rounded-lg bg-slate-950/50 overflow-hidden transition-colors"
              >
                <button
                  onClick={() => toggleSection(sec.section_number)}
                  className="w-full flex items-center justify-between p-3 text-left hover:bg-slate-900/50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-xs font-mono text-slate-500 w-6">#{sec.section_number}</span>
                    <span className="text-xs font-medium text-slate-200">{sec.title}</span>
                    <span className={`px-2 py-0.5 rounded text-[10px] font-semibold border ${CLASSIFICATION_BADGES[sec.classification] || 'bg-slate-800 text-slate-300'}`}>
                      {sec.classification}
                    </span>
                  </div>
                  {isExpanded ? (
                    <ChevronUp className="w-4 h-4 text-slate-400" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-slate-400" />
                  )}
                </button>

                {isExpanded && (
                  <div className="px-4 pb-3 pt-1 text-xs text-slate-300 border-t border-slate-800/40 leading-relaxed">
                    {sec.content}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
