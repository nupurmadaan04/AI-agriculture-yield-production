import React from 'react'
import type { DecisionOption, DecisionRobustness } from '../../types/decision'
import { Sliders, ShieldCheck, AlertCircle, ArrowUpRight, ArrowDownRight, Scale } from 'lucide-react'

interface Props {
  options: DecisionOption[]
  robustness: DecisionRobustness[]
}

const ROBUSTNESS_BADGES: Record<string, string> = {
  ROBUST: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
  'MODERATELY ROBUST': 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30',
  SENSITIVE: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
  UNSUPPORTED: 'bg-red-500/10 text-red-400 border-red-500/30'
}

export const DecisionOptionsPanel: React.FC<Props> = ({ options, robustness }) => {
  const robustnessMap = new Map(robustness.map(r => [r.option_id, r]))

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sliders className="w-4 h-4 text-cyan-400" />
          <h3 className="text-sm font-semibold text-slate-100">Scenario Decision Options & Robustness</h3>
        </div>
        <span className="text-xs text-slate-400">Day 10 Archetypes & Pareto Optimization</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {options.map(opt => {
          const rob = robustnessMap.get(opt.option_id)
          const isPos = opt.projected_yield_delta_kg_ha >= 0

          return (
            <div
              key={opt.option_id}
              className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 flex flex-col justify-between space-y-3"
            >
              <div className="space-y-2">
                <div className="flex items-start justify-between gap-2">
                  <span className="text-xs font-semibold text-slate-100 leading-snug">{opt.title}</span>
                  {rob && (
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold border shrink-0 ${ROBUSTNESS_BADGES[rob.classification] || 'bg-slate-800'}`}>
                      {rob.classification}
                    </span>
                  )}
                </div>

                <div className="flex items-baseline gap-2 pt-1">
                  <span className="text-xl font-bold font-mono text-slate-100">{opt.projected_yield_kg_ha}</span>
                  <span className="text-xs text-slate-400">kg/ha</span>
                  <span className={`text-xs font-semibold flex items-center ${isPos ? 'text-emerald-400' : 'text-red-400'}`}>
                    {isPos ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                    {opt.projected_yield_delta_kg_ha > 0 ? `+${opt.projected_yield_delta_kg_ha}` : opt.projected_yield_delta_kg_ha} kg/ha
                  </span>
                </div>

                <div className="text-[11px] text-slate-300 leading-relaxed pt-1">
                  <strong>Trade-offs:</strong> {opt.tradeoffs}
                </div>
              </div>

              <div className="border-t border-slate-800/80 pt-2.5 space-y-1 text-[11px] text-slate-400">
                <div className="flex justify-between">
                  <span>Production Impact:</span>
                  <strong className={opt.projected_production_delta_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                    {opt.projected_production_delta_pct > 0 ? `+${opt.projected_production_delta_pct}` : opt.projected_production_delta_pct}%
                  </strong>
                </div>
                <div className="flex justify-between">
                  <span>Risk Shift:</span>
                  <strong className="text-slate-200">{opt.risk_change}</strong>
                </div>
                {rob && (
                  <div className="text-[10px] text-slate-500 pt-1">
                    Sensitivity bounds: ±20% perturbation (Max dev: {rob.max_tested_deviation_kg_ha} kg/ha)
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
