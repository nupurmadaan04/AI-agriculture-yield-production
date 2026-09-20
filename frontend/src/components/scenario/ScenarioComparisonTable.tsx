import React from 'react'
import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { Layers, ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react'
import { ScenarioComparisonResult } from '../../types/scenario'

interface ScenarioComparisonTableProps {
  comparison?: ScenarioComparisonResult | null
}

export const ScenarioComparisonTable: React.FC<ScenarioComparisonTableProps> = ({
  comparison
}) => {
  if (!comparison || !comparison.comparison_matrix || comparison.comparison_matrix.length === 0) {
    return null
  }

  return (
    <Card className="p-6 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border/40 pb-4">
        <div>
          <h3 className="text-base font-bold text-foreground flex items-center gap-2">
            <Layers className="w-5 h-5 text-purple-400" />
            <span>Multi-Scenario Comparative Evaluation ({comparison.location})</span>
          </h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Evaluates hypothetical candidate scenarios against the immutable baseline for {comparison.horizon}-year forecast horizon.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="blue">Baseline: {comparison.baseline_yield.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} kg/ha</Badge>
          <Badge variant="purple">Δ Range: {comparison.yield_range_kg_ha.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} kg/ha</Badge>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs text-left border-collapse">
          <thead>
            <tr className="border-b border-border text-muted-foreground font-mono text-[11px]">
              <th className="py-2.5 px-3">Scenario Archetype</th>
              <th className="py-2.5 px-3 text-right">Projected Yield</th>
              <th className="py-2.5 px-3 text-right">Δ Yield (kg/ha)</th>
              <th className="py-2.5 px-3 text-right">% Change</th>
              <th className="py-2.5 px-3 text-right">Risk Score</th>
              <th className="py-2.5 px-3 text-right">Warning Score</th>
              <th className="py-2.5 px-3 text-right">Spread (P10–P90)</th>
              <th className="py-2.5 px-3">Interpretation</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/40 font-mono">
            {comparison.comparison_matrix.map((row) => {
              const isPositive = row.yield_delta > 0
              const isNegative = row.yield_delta < 0

              return (
                <tr
                  key={row.scenario_id}
                  className={`hover:bg-muted/30 transition-colors ${row.is_baseline ? 'bg-blue-500/5 font-semibold' : ''}`}
                >
                  <td className="py-3 px-3 font-sans">
                    <div className="font-medium text-foreground flex items-center gap-1.5">
                      {row.scenario_name}
                      {row.is_baseline && <Badge variant="blue">Baseline</Badge>}
                    </div>
                    <div className="text-[10px] text-muted-foreground font-mono">{row.scenario_id}</div>
                  </td>

                  <td className="py-3 px-3 text-right font-bold text-foreground">
                    {row.projected_yield.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} kg/ha
                  </td>

                  <td className="py-3 px-3 text-right">
                    {row.is_baseline ? (
                      <span className="text-muted-foreground">—</span>
                    ) : (
                      <span className={`inline-flex items-center gap-0.5 font-bold ${isPositive ? 'text-emerald-400' : 'text-red-400'}`}>
                        {isPositive ? <ArrowUpRight className="w-3.5 h-3.5" /> : <ArrowDownRight className="w-3.5 h-3.5" />}
                        {isPositive ? `+${row.yield_delta.toFixed(1)}` : row.yield_delta.toFixed(1)}
                      </span>
                    )}
                  </td>

                  <td className="py-3 px-3 text-right">
                    {row.is_baseline ? (
                      <span className="text-muted-foreground">0.0%</span>
                    ) : (
                      <span className={`font-bold ${isPositive ? 'text-emerald-400' : 'text-red-400'}`}>
                        {isPositive ? `+${row.yield_percent_change.toFixed(2)}%` : `${row.yield_percent_change.toFixed(2)}%`}
                      </span>
                    )}
                  </td>

                  <td className="py-3 px-3 text-right">
                    <span className={row.risk_score > 60 ? 'text-red-400 font-bold' : (row.risk_score < 30 ? 'text-emerald-400 font-bold' : 'text-foreground')}>
                      {row.risk_score.toFixed(1)}
                    </span>
                  </td>

                  <td className="py-3 px-3 text-right">
                    <span className={row.warning_score > 50 ? 'text-amber-400 font-bold' : 'text-foreground'}>
                      {row.warning_score.toFixed(1)}
                    </span>
                  </td>

                  <td className="py-3 px-3 text-right text-muted-foreground">
                    ±{row.prediction_spread.toFixed(1)}
                  </td>

                  <td className="py-3 px-3 font-sans text-muted-foreground text-[11px] max-w-xs">
                    {row.interpretation}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </Card>
  )
}
