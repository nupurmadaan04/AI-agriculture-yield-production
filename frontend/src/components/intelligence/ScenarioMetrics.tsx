import React from 'react'
import { TrendingUp, TrendingDown, ShieldAlert, Sparkles, AlertTriangle } from 'lucide-react'
import { ScenarioSimulationDelta } from '../../types/intelligence'
import { formatYield, formatNumber } from '../../lib/utils'
import { Badge } from '../ui/Badge'

interface ScenarioMetricsProps {
  delta: ScenarioSimulationDelta
  baselineYield: number
  scenarioYield: number
  baselineRisk: number
  scenarioRisk: number
}

export const ScenarioMetrics: React.FC<ScenarioMetricsProps> = ({
  delta,
  baselineYield,
  scenarioYield,
  baselineRisk,
  scenarioRisk
}) => {
  const isYieldPositive = delta.yield_delta_kg_ha >= 0
  const isRiskReduced = delta.risk_delta <= 0

  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
      {/* Yield Shift Card */}
      <div className="p-4 rounded-xl border border-border bg-card/80 space-y-2">
        <div className="flex items-center justify-between text-xs text-muted-foreground font-semibold uppercase">
          <span>Modeled Yield Shift</span>
          {isYieldPositive ? (
            <TrendingUp className="w-4 h-4 text-emerald-500" />
          ) : (
            <TrendingDown className="w-4 h-4 text-amber-500" />
          )}
        </div>
        <div className="text-2xl font-black font-mono text-foreground flex items-baseline gap-1.5">
          <span className={isYieldPositive ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400'}>
            {isYieldPositive ? '+' : ''}{delta.yield_delta_kg_ha.toFixed(1)}
          </span>
          <span className="text-xs font-normal text-muted-foreground">kg/ha</span>
        </div>
        <div className="text-[11px] text-muted-foreground font-mono">
          {isYieldPositive ? '+' : ''}{delta.yield_percent_change.toFixed(2)}% vs baseline ({formatYield(baselineYield)})
        </div>
      </div>

      {/* Risk Shift Card */}
      <div className="p-4 rounded-xl border border-border bg-card/80 space-y-2">
        <div className="flex items-center justify-between text-xs text-muted-foreground font-semibold uppercase">
          <span>Risk Score Shift</span>
          <ShieldAlert className={`w-4 h-4 ${isRiskReduced ? 'text-emerald-500' : 'text-amber-500'}`} />
        </div>
        <div className="text-2xl font-black font-mono text-foreground flex items-baseline gap-1.5">
          <span className={isRiskReduced ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400'}>
            {delta.risk_delta > 0 ? '+' : ''}{delta.risk_delta.toFixed(1)}
          </span>
          <span className="text-xs font-normal text-muted-foreground">pts</span>
        </div>
        <div className="text-[11px] text-muted-foreground font-mono">
          Shift: {baselineRisk.toFixed(1)} → {scenarioRisk.toFixed(1)} / 100 ({delta.risk_direction})
        </div>
      </div>

      {/* Uncertainty Delta Card */}
      <div className="p-4 rounded-xl border border-border bg-card/80 space-y-2">
        <div className="flex items-center justify-between text-xs text-muted-foreground font-semibold uppercase">
          <span>Prediction Spread Delta</span>
          <Sparkles className="w-4 h-4 text-sky-500" />
        </div>
        <div className="text-2xl font-black font-mono text-foreground flex items-baseline gap-1.5">
          <span>{delta.spread_delta_kg_ha > 0 ? '+' : ''}{delta.spread_delta_kg_ha.toFixed(1)}</span>
          <span className="text-xs font-normal text-muted-foreground">kg/ha</span>
        </div>
        <div className="text-[11px] text-muted-foreground font-mono">
          Ensemble tree interval dispersion (P10–P90)
        </div>
      </div>
    </div>
  )
}
