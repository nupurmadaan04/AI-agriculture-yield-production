import React from 'react'
import type { DecisionSignal } from '../../types/decision'
import { Activity, AlertTriangle, CheckCircle, Info } from 'lucide-react'

interface Props {
  signals: DecisionSignal[]
}

const STRENGTH_BADGES: Record<string, string> = {
  HIGH: 'bg-red-500/10 text-red-400 border-red-500/30',
  MODERATE: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
  LOW: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
  NEGLIGIBLE: 'bg-slate-500/10 text-slate-400 border-slate-500/30'
}

export const DecisionSignalFusionPanel: React.FC<Props> = ({ signals }) => {
  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
      <div className="flex items-center gap-2">
        <Activity className="w-4 h-4 text-amber-400" />
        <h3 className="text-sm font-semibold text-slate-100">Decision Signal Fusion ({signals.length} Fused Signals)</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {signals.map(sig => (
          <div
            key={sig.signal_name}
            className="bg-slate-950/60 border border-slate-800 rounded-lg p-3.5 space-y-2 text-xs"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="font-semibold text-slate-200">{sig.signal_label}</span>
              <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${STRENGTH_BADGES[sig.strength] || 'bg-slate-800 text-slate-400'}`}>
                {sig.strength} STRENGTH
              </span>
            </div>

            <div className="text-slate-300 leading-relaxed">
              {sig.interpretation}
            </div>

            <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-slate-800/60 text-[11px] text-slate-400">
              <span>Severity: <strong className="text-slate-300">{sig.severity}</strong></span>
              <span>•</span>
              <span>Persistence: <strong className="text-slate-300">{sig.persistence}</strong></span>
              <span>•</span>
              <span>Evidence: <strong className="text-emerald-400">{sig.supporting_evidence.length} IDs</strong></span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
