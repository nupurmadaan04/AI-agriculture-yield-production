import React from 'react'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { FeatureContribution } from '../../types/explainability'

interface FeatureContributionChartProps {
  contributions: FeatureContribution[]
}

export const FeatureContributionChart: React.FC<FeatureContributionChartProps> = ({ contributions }) => {
  if (!contributions || contributions.length === 0) return null

  const maxAbs = Math.max(...contributions.map((c) => Math.abs(c.contribution_kg_ha)), 1.0)

  return (
    <div className="space-y-3">
      {contributions.map((c) => {
        const isPositive = c.contribution_direction === 'POSITIVE'
        const barWidth = Math.min(100, Math.round((Math.abs(c.contribution_kg_ha) / maxAbs) * 100))

        return (
          <div
            key={c.feature}
            className="p-3 bg-slate-800/40 rounded-lg border border-slate-800 hover:border-slate-700 transition-colors"
          >
            <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
              <div className="flex items-center gap-2">
                <div
                  className={`p-1 rounded ${
                    isPositive ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'
                  }`}
                >
                  {isPositive ? <TrendingUp className="w-3.5 h-3.5" /> : <TrendingDown className="w-3.5 h-3.5" />}
                </div>
                <div>
                  <span className="text-xs font-semibold text-white">{c.feature_label}</span>
                  <span className="text-[10px] font-mono text-slate-400 ml-2">({c.feature})</span>
                </div>
              </div>

              <div className="flex items-center gap-3 text-xs">
                <div className="text-slate-400">
                  Val: <span className="text-slate-200 font-medium">{c.feature_value.toLocaleString()}</span>
                  <span className="text-slate-500 mx-1">/</span>
                  Base: <span className="text-slate-400">{c.baseline_value.toLocaleString()}</span>
                </div>
                <div className="text-right min-w-[90px]">
                  <span
                    className={`font-mono font-bold ${
                      isPositive ? 'text-emerald-400' : 'text-rose-400'
                    }`}
                  >
                    {isPositive ? '+' : ''}
                    {c.contribution_kg_ha.toFixed(1)} kg/ha
                  </span>
                  <span className="text-[10px] text-slate-400 ml-1">({c.relative_influence_pct}%)</span>
                </div>
              </div>
            </div>

            {/* Directional Waterfall Bar */}
            <div className="flex items-center gap-2 h-2.5 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
              <div className="flex-1 flex justify-end">
                {!isPositive && (
                  <div
                    className="bg-rose-500 h-full rounded-l-full transition-all duration-500"
                    style={{ width: `${barWidth}%` }}
                  />
                )}
              </div>
              <div className="w-0.5 h-full bg-slate-700" />
              <div className="flex-1 flex justify-start">
                {isPositive && (
                  <div
                    className="bg-emerald-500 h-full rounded-r-full transition-all duration-500"
                    style={{ width: `${barWidth}%` }}
                  />
                )}
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
