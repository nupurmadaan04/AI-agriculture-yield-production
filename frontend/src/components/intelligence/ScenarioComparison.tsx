import React from 'react'
import { ScenarioProfile } from '../../types/intelligence'
import { formatYield, formatNumber } from '../../lib/utils'
import { Badge } from '../ui/Badge'
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card'
import { CheckCircle2, AlertTriangle, ShieldCheck, Sparkles } from 'lucide-react'

interface ScenarioComparisonProps {
  baseline: ScenarioProfile
  scenario: ScenarioProfile
  stateName: string
  districtName: string
}

export const ScenarioComparison: React.FC<ScenarioComparisonProps> = ({
  baseline,
  scenario,
  stateName,
  districtName
}) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* Baseline Column */}
      <Card className="border-border/80 bg-card/60">
        <CardHeader className="pb-3 border-b border-border/40">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-bold flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-slate-400" />
              <span>1. Baseline District Profile</span>
            </CardTitle>
            <Badge variant="outline" className="text-[10px]">ICRISAT Historical</Badge>
          </div>
          <p className="text-[11px] text-muted-foreground">{stateName} ({districtName})</p>
        </CardHeader>

        <CardContent className="space-y-4 pt-4 text-xs">
          <div className="p-3.5 rounded-xl bg-muted/40 text-center space-y-0.5">
            <span className="text-[10px] text-muted-foreground font-semibold uppercase">Estimated Baseline Yield</span>
            <div className="text-3xl font-black text-foreground font-mono">
              {formatYield(baseline.predicted_yield)}
            </div>
            <p className="text-[11px] text-muted-foreground font-mono">
              Spread (P10–P90): {formatYield(baseline.lower_bound)} – {formatYield(baseline.upper_bound)}
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="p-2.5 rounded-lg border border-border bg-card">
              <span className="text-[10px] text-muted-foreground block">Risk Score</span>
              <div className="font-mono font-bold text-foreground flex items-center justify-between mt-0.5">
                <span>{baseline.risk_score.toFixed(1)}/100</span>
                <Badge
                  variant={baseline.risk_level === 'LOW' ? 'success' : baseline.risk_level === 'MODERATE' ? 'blue' : 'warning'}
                  className="text-[9px]"
                >
                  {baseline.risk_level}
                </Badge>
              </div>
            </div>

            <div className="p-2.5 rounded-lg border border-border bg-card">
              <span className="text-[10px] text-muted-foreground block">Anomaly Status</span>
              <div className="font-semibold flex items-center gap-1.5 mt-1 text-[11px]">
                {baseline.is_anomaly ? (
                  <span className="text-amber-500 flex items-center gap-1 font-bold">
                    <AlertTriangle className="w-3 h-3" /> Anomaly
                  </span>
                ) : (
                  <span className="text-emerald-500 flex items-center gap-1 font-bold">
                    <CheckCircle2 className="w-3 h-3" /> Normal
                  </span>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Scenario Column */}
      <Card className="border-sky-500/40 bg-sky-500/5 shadow-lg shadow-sky-500/5">
        <CardHeader className="pb-3 border-b border-sky-500/20">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-bold flex items-center gap-2 text-sky-600 dark:text-sky-400">
              <span className="w-2.5 h-2.5 rounded-full bg-sky-500 animate-pulse" />
              <span>2. Simulated Scenario Projection</span>
            </CardTitle>
            <Badge variant="blue" className="text-[10px]">What-If Simulation</Badge>
          </div>
          <p className="text-[11px] text-muted-foreground">Modified Input Vector</p>
        </CardHeader>

        <CardContent className="space-y-4 pt-4 text-xs">
          <div className="p-3.5 rounded-xl bg-sky-500/10 border border-sky-500/20 text-center space-y-0.5">
            <span className="text-[10px] text-sky-700 dark:text-sky-300 font-semibold uppercase">Projected Scenario Yield</span>
            <div className="text-3xl font-black text-foreground font-mono">
              {formatYield(scenario.predicted_yield)}
            </div>
            <p className="text-[11px] text-muted-foreground font-mono">
              Spread (P10–P90): {formatYield(scenario.lower_bound)} – {formatYield(scenario.upper_bound)}
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="p-2.5 rounded-lg border border-sky-500/20 bg-card">
              <span className="text-[10px] text-muted-foreground block">Scenario Risk</span>
              <div className="font-mono font-bold text-foreground flex items-center justify-between mt-0.5">
                <span>{scenario.risk_score.toFixed(1)}/100</span>
                <Badge
                  variant={scenario.risk_level === 'LOW' ? 'success' : scenario.risk_level === 'MODERATE' ? 'blue' : 'warning'}
                  className="text-[9px]"
                >
                  {scenario.risk_level}
                </Badge>
              </div>
            </div>

            <div className="p-2.5 rounded-lg border border-sky-500/20 bg-card">
              <span className="text-[10px] text-muted-foreground block">Anomaly Flag</span>
              <div className="font-semibold flex items-center gap-1.5 mt-1 text-[11px]">
                {scenario.is_anomaly ? (
                  <span className="text-amber-500 flex items-center gap-1 font-bold">
                    <AlertTriangle className="w-3 h-3" /> Outlier Signal
                  </span>
                ) : (
                  <span className="text-emerald-500 flex items-center gap-1 font-bold">
                    <CheckCircle2 className="w-3 h-3" /> Valid Bounds
                  </span>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
