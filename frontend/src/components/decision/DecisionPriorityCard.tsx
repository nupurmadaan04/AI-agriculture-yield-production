import React from 'react'
import type { DecisionPriority } from '../../types/decision'
import { Target, CheckCircle2 } from 'lucide-react'

interface Props {
  priorities: DecisionPriority[]
}

const PRIORITY_BADGES: Record<string, string> = {
  HIGH: 'bg-red-500/10 text-red-400 border-red-500/30',
  MODERATE: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
  LOW: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
}

export const DecisionPriorityCard: React.FC<Props> = ({ priorities }) => {
  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
      <div className="flex items-center gap-2">
        <Target className="w-4 h-4 text-emerald-400" />
        <h3 className="text-sm font-semibold text-slate-100">Recommended Analytical Priorities</h3>
      </div>

      <div className="space-y-3">
        {priorities.map(p => (
          <div
            key={p.priority_rank}
            className="bg-slate-950/60 border border-slate-800 rounded-lg p-4 space-y-2.5 text-xs"
          >
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="w-5 h-5 rounded-full bg-slate-800 flex items-center justify-center font-bold text-slate-300 text-[11px]">
                  {p.priority_rank}
                </span>
                <span className="font-semibold text-slate-100 text-sm">{p.issue}</span>
              </div>
              <span className={`px-2.5 py-0.5 rounded text-[10px] font-bold border ${PRIORITY_BADGES[p.priority_level] || 'bg-slate-800 text-slate-400'}`}>
                {p.priority_level} PRIORITY
              </span>
            </div>

            <div className="space-y-1 pl-7">
              {p.reasoning.map((r, i) => (
                <div key={i} className="flex items-start gap-1.5 text-slate-300 text-xs">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400/80 mt-0.5 shrink-0" />
                  <span>{r}</span>
                </div>
              ))}
            </div>

            <div className="pl-7 pt-1 text-[11px] text-slate-500">
              Supporting Evidence: {p.supporting_evidence.join(', ') || 'Global baseline'}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
