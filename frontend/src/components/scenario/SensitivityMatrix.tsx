import React from 'react'
import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { Activity, ArrowUp, ArrowDown, HelpCircle } from 'lucide-react'
import { SensitivityResult } from '../../types/scenario'

interface SensitivityMatrixProps {
  sensitivityData?: SensitivityResult | null
  isLoading?: boolean
}

export const SensitivityMatrix: React.FC<SensitivityMatrixProps> = ({
  sensitivityData,
  isLoading = false
}) => {
  if (isLoading) {
    return (
      <Card className="p-6 text-center text-xs text-muted-foreground">
        Evaluating feature perturbation matrix (-20% to +20%)...
      </Card>
    )
  }

  if (!sensitivityData || !sensitivityData.sensitivity_matrix || sensitivityData.sensitivity_matrix.length === 0) {
    return null
  }

  return (
    <Card className="p-6 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border/40 pb-4">
        <div>
          <h3 className="text-base font-bold text-foreground flex items-center gap-2">
            <Activity className="w-5 h-5 text-cyan-400" />
            <span>Feature Sensitivity & Output Elasticity Matrix ({sensitivityData.location})</span>
          </h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Systematic ±20% perturbation response across supported agricultural features.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="blue">Baseline: {sensitivityData.baseline_prediction.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} kg/ha</Badge>
          <Badge variant="green">Top Driver: {sensitivityData.most_sensitive_feature}</Badge>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs text-left border-collapse">
          <thead>
            <tr className="border-b border-border text-muted-foreground font-mono text-[11px]">
              <th className="py-2.5 px-3">Rank</th>
              <th className="py-2.5 px-3">Feature Key</th>
              <th className="py-2.5 px-3 text-right">Baseline</th>
              <th className="py-2.5 px-3 text-center">-20% Shock</th>
              <th className="py-2.5 px-3 text-center">-10% Shock</th>
              <th className="py-2.5 px-3 text-center">0% Baseline</th>
              <th className="py-2.5 px-3 text-center">+10% Gain</th>
              <th className="py-2.5 px-3 text-center">+20% Gain</th>
              <th className="py-2.5 px-3 text-right">Elasticity Index</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/40 font-mono">
            {sensitivityData.sensitivity_matrix.map((row) => {
              const p_m20 = row.perturbation_responses.find((p) => p.perturbation_pct === -20)
              const p_m10 = row.perturbation_responses.find((p) => p.perturbation_pct === -10)
              const p_0 = row.perturbation_responses.find((p) => p.perturbation_pct === 0)
              const p_p10 = row.perturbation_responses.find((p) => p.perturbation_pct === 10)
              const p_p20 = row.perturbation_responses.find((p) => p.perturbation_pct === 20)

              return (
                <tr key={row.feature_key} className="hover:bg-muted/30 transition-colors">
                  <td className="py-2.5 px-3 font-bold text-cyan-400">
                    #{row.sensitivity_rank}
                  </td>

                  <td className="py-2.5 px-3 font-sans">
                    <div className="font-medium text-foreground">{row.feature_name}</div>
                    <div className="text-[10px] text-muted-foreground font-mono">{row.feature_key}</div>
                  </td>

                  <td className="py-2.5 px-3 text-right font-medium text-foreground">
                    {row.baseline_value.toFixed(1)}
                  </td>

                  {/* -20% Column */}
                  <td className="py-2.5 px-3 text-center">
                    <span className={`px-2 py-1 rounded text-[11px] font-bold ${
                      (p_m20?.yield_delta || 0) < 0 ? 'bg-red-500/10 text-red-400' : 'bg-muted/40 text-muted-foreground'
                    }`}>
                      {p_m20 ? Math.round(p_m20.predicted_yield).toLocaleString() : '—'} ({p_m20 && p_m20.yield_percent_change > 0 ? '+' : ''}{p_m20 ? p_m20.yield_percent_change.toFixed(1) : 0}%)
                    </span>
                  </td>

                  {/* -10% Column */}
                  <td className="py-2.5 px-3 text-center">
                    <span className={`px-2 py-1 rounded text-[11px] font-bold ${
                      (p_m10?.yield_delta || 0) < 0 ? 'bg-red-500/10 text-red-400' : 'bg-muted/40 text-muted-foreground'
                    }`}>
                      {p_m10 ? Math.round(p_m10.predicted_yield).toLocaleString() : '—'} ({p_m10 && p_m10.yield_percent_change > 0 ? '+' : ''}{p_m10 ? p_m10.yield_percent_change.toFixed(1) : 0}%)
                    </span>
                  </td>

                  {/* 0% Column */}
                  <td className="py-2.5 px-3 text-center font-bold text-foreground bg-blue-500/5">
                    {p_0 ? Math.round(p_0.predicted_yield).toLocaleString() : '—'}
                  </td>

                  {/* +10% Column */}
                  <td className="py-2.5 px-3 text-center">
                    <span className={`px-2 py-1 rounded text-[11px] font-bold ${
                      (p_p10?.yield_delta || 0) > 0 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-muted/40 text-muted-foreground'
                    }`}>
                      {p_p10 ? Math.round(p_p10.predicted_yield).toLocaleString() : '—'} ({p_p10 && p_p10.yield_percent_change > 0 ? '+' : ''}{p_p10 ? p_p10.yield_percent_change.toFixed(1) : 0}%)
                    </span>
                  </td>

                  {/* +20% Column */}
                  <td className="py-2.5 px-3 text-center">
                    <span className={`px-2 py-1 rounded text-[11px] font-bold ${
                      (p_p20?.yield_delta || 0) > 0 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-muted/40 text-muted-foreground'
                    }`}>
                      {p_p20 ? Math.round(p_p20.predicted_yield).toLocaleString() : '—'} ({p_p20 && p_p20.yield_percent_change > 0 ? '+' : ''}{p_p20 ? p_p20.yield_percent_change.toFixed(1) : 0}%)
                    </span>
                  </td>

                  {/* Elasticity Index */}
                  <td className="py-2.5 px-3 text-right font-bold text-emerald-400">
                    {row.elasticity_index.toFixed(2)}%
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div className="p-3 rounded bg-muted/40 text-[11px] text-muted-foreground font-sans flex items-center gap-2">
        <HelpCircle className="w-4 h-4 text-slate-400 shrink-0" />
        <span>{sensitivityData.scientific_disclaimer}</span>
      </div>
    </Card>
  )
}
