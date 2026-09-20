import React from 'react'
import type { EvidenceStatus } from '../../types/decision'
import { ShieldCheck, Database, CheckCircle2, TrendingUp, BarChart2 } from 'lucide-react'

interface Props {
  status: EvidenceStatus
}

export const DecisionConfidenceMatrix: React.FC<Props> = ({ status }) => {
  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 space-y-1">
        <div className="flex items-center gap-1.5 text-xs text-slate-400">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
          <span>Evidence Agreement</span>
        </div>
        <div className="text-sm font-semibold text-slate-100">{status.evidence_agreement}</div>
        <div className="text-[11px] text-slate-500">Multi-source alignment</div>
      </div>

      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 space-y-1">
        <div className="flex items-center gap-1.5 text-xs text-slate-400">
          <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
          <span>Model Reliability</span>
        </div>
        <div className="text-sm font-semibold text-cyan-300">{status.model_reliability}</div>
        <div className="text-[11px] text-slate-500">Chronological out-of-time</div>
      </div>

      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 space-y-1">
        <div className="flex items-center gap-1.5 text-xs text-slate-400">
          <Database className="w-3.5 h-3.5 text-blue-400" />
          <span>Data Quality</span>
        </div>
        <div className="text-sm font-semibold text-slate-100">{status.data_quality_score}</div>
        <div className="text-[11px] text-slate-500">4-Pillar SAIF audit</div>
      </div>

      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 space-y-1">
        <div className="flex items-center gap-1.5 text-xs text-slate-400">
          <BarChart2 className="w-3.5 h-3.5 text-amber-400" />
          <span>Prediction Spread</span>
        </div>
        <div className="text-sm font-semibold text-slate-100">{status.prediction_spread}</div>
        <div className="text-[11px] text-slate-500">RF tree ensemble spread</div>
      </div>

      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 space-y-1">
        <div className="flex items-center gap-1.5 text-xs text-slate-400">
          <TrendingUp className="w-3.5 h-3.5 text-purple-400" />
          <span>Signal Persistence</span>
        </div>
        <div className="text-sm font-semibold text-slate-100">{status.signal_persistence}</div>
        <div className="text-[11px] text-slate-500">Multi-year trajectory</div>
      </div>
    </div>
  )
}
