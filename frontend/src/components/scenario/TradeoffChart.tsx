import React from 'react'
import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip,
  CartesianGrid,
  Cell
} from 'recharts'
import { Scale } from 'lucide-react'
import { OptimizationResult } from '../../types/scenario'

interface TradeoffChartProps {
  optimizationResult?: OptimizationResult | null
}

export const TradeoffChart: React.FC<TradeoffChartProps> = ({
  optimizationResult
}) => {
  if (!optimizationResult || !optimizationResult.all_ranked_candidates || optimizationResult.all_ranked_candidates.length === 0) {
    return null
  }

  const data = optimizationResult.all_ranked_candidates.map((c) => ({
    name: c.scenario_name,
    yieldDelta: c.yield_delta,
    resourceShift: c.resource_change_pct,
    risk: c.risk_score,
    decisionScore: c.decision_score,
    isPareto: c.is_pareto_optimal,
    isFeasible: c.is_feasible,
    rank: c.rank
  }))

  return (
    <Card className="p-6 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border/40 pb-4">
        <div>
          <h3 className="text-base font-bold text-foreground flex items-center gap-2">
            <Scale className="w-5 h-5 text-indigo-400" />
            <span>Decision Trade-off Space & Pareto Frontier</span>
          </h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Plots projected yield delta vs input resource shift. Green markers represent non-dominated Pareto alternatives.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="green">Pareto Optimal</Badge>
          <Badge variant="blue">Dominated Candidates</Badge>
        </div>
      </div>

      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis
              type="number"
              dataKey="resourceShift"
              name="Resource Shift (%)"
              unit="%"
              stroke="#94a3b8"
              fontSize={11}
              label={{ value: 'Resource Allocation Shift (%)', position: 'insideBottom', offset: -10, fill: '#94a3b8', fontSize: 11 }}
            />
            <YAxis
              type="number"
              dataKey="yieldDelta"
              name="Yield Delta (kg/ha)"
              unit=" kg/ha"
              stroke="#94a3b8"
              fontSize={11}
              label={{ value: 'Projected Δ Yield (kg/ha)', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 11 }}
            />
            <ZAxis type="number" dataKey="decisionScore" range={[100, 400]} name="Decision Score" />
            <Tooltip
              cursor={{ strokeDasharray: '3 3' }}
              contentStyle={{
                backgroundColor: 'rgba(15, 23, 42, 0.95)',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '8px',
                fontSize: '12px',
                fontFamily: 'monospace'
              }}
              formatter={(val: any, name: any) => {
                if (name === 'Resource Shift (%)') return [`${val}%`, name || '']
                if (name === 'Yield Delta (kg/ha)') return [`${val > 0 ? '+' : ''}${val} kg/ha`, name || '']
                return [val, name || '']
              }}
            />
            <Scatter name="Candidate Scenarios" data={data}>
              {data.map((entry, index) => {
                let color = '#3b82f6'
                if (!entry.isFeasible) color = '#ef4444'
                else if (entry.isPareto) color = '#10b981'

                return (
                  <Cell
                    key={`cell-${index}`}
                    fill={color}
                    stroke={entry.isPareto ? '#34d399' : '#1e293b'}
                    strokeWidth={entry.isPareto ? 2 : 1}
                  />
                )
              })}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <div className="flex flex-wrap items-center justify-between text-[11px] text-muted-foreground pt-2 border-t border-border/40 font-mono">
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 inline-block"></span>
          Pareto Optimal (Non-Dominated)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-blue-400 inline-block"></span>
          Feasible Dominated
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-red-400 inline-block"></span>
          Constraint Infeasible
        </span>
      </div>
    </Card>
  )
}
