import React, { useState } from 'react'
import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { Target, CheckCircle, XCircle, Award, Scale, Play } from 'lucide-react'
import { OptimizationResult } from '../../types/scenario'

interface OptimizationPanelProps {
  optimizationResult?: OptimizationResult | null
  onRunOptimization: (params: {
    weights: {
      yield_improvement: number
      risk_reduction: number
      resource_efficiency: number
      model_reliability: number
    }
    constraints: {
      min_yield?: number
      max_resource_change_pct?: number
      max_risk_score?: number
      min_reliability_score?: number
    }
  }) => void
  isLoading?: boolean
}

export const OptimizationPanel: React.FC<OptimizationPanelProps> = ({
  optimizationResult,
  onRunOptimization,
  isLoading = false
}) => {
  // Objective Weights
  const [wYield, setWYield] = useState(40)
  const [wRisk, setWRisk] = useState(25)
  const [wResource, setWResource] = useState(20)
  const [wReliability, setWReliability] = useState(15)

  // Constraints
  const [minYield, setMinYield] = useState<string>('')
  const [maxResource, setMaxResource] = useState<string>('20')
  const [maxRisk, setMaxRisk] = useState<string>('60')
  const [minReliability, setMinReliability] = useState<string>('0.70')

  const handleOptimize = (e: React.FormEvent) => {
    e.preventDefault()
    onRunOptimization({
      weights: {
        yield_improvement: wYield / 100.0,
        risk_reduction: wRisk / 100.0,
        resource_efficiency: wResource / 100.0,
        model_reliability: wReliability / 100.0
      },
      constraints: {
        min_yield: minYield ? Number(minYield) : undefined,
        max_resource_change_pct: maxResource ? Number(maxResource) : undefined,
        max_risk_score: maxRisk ? Number(maxRisk) : undefined,
        min_reliability_score: minReliability ? Number(minReliability) : undefined
      }
    })
  }

  const rec = optimizationResult?.recommended_scenario

  return (
    <Card className="p-6 space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border/40 pb-4">
        <div>
          <h3 className="text-base font-bold text-foreground flex items-center gap-2">
            <Target className="w-5 h-5 text-amber-400" />
            <span>Multi-Objective Decision Optimizer & Pareto Frontier</span>
          </h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Evaluates candidate strategies across user-configured objective weights and feasibility bounds.
          </p>
        </div>
        <Badge variant="amber">Transparent Linear Scalarization</Badge>
      </div>

      <form onSubmit={handleOptimize} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Objective Weights */}
          <div className="p-4 rounded-xl bg-card/60 border border-border/50 space-y-3">
            <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground font-mono">
              Objective Function Weights (Total: {wYield + wRisk + wResource + wReliability}%)
            </span>

            <div className="space-y-3 pt-1">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-foreground font-medium">Yield Improvement</span>
                  <span className="font-mono text-emerald-400 font-bold">{wYield}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={wYield}
                  onChange={(e) => setWYield(Number(e.target.value))}
                  className="w-full h-1.5 bg-muted rounded appearance-none cursor-pointer accent-emerald-500"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-foreground font-medium">Risk Reduction</span>
                  <span className="font-mono text-blue-400 font-bold">{wRisk}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={wRisk}
                  onChange={(e) => setWRisk(Number(e.target.value))}
                  className="w-full h-1.5 bg-muted rounded appearance-none cursor-pointer accent-blue-500"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-foreground font-medium">Resource Efficiency</span>
                  <span className="font-mono text-cyan-400 font-bold">{wResource}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={wResource}
                  onChange={(e) => setWResource(Number(e.target.value))}
                  className="w-full h-1.5 bg-muted rounded appearance-none cursor-pointer accent-cyan-500"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-foreground font-medium">Model Reliability</span>
                  <span className="font-mono text-purple-400 font-bold">{wReliability}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={wReliability}
                  onChange={(e) => setWReliability(Number(e.target.value))}
                  className="w-full h-1.5 bg-muted rounded appearance-none cursor-pointer accent-purple-500"
                />
              </div>
            </div>
          </div>

          {/* Feasibility Constraints */}
          <div className="p-4 rounded-xl bg-card/60 border border-border/50 space-y-3">
            <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground font-mono">
              Explicit Feasibility Constraints
            </span>

            <div className="grid grid-cols-2 gap-3 pt-1">
              <div>
                <label className="block text-[11px] text-muted-foreground mb-1">Min Expected Yield (kg/ha)</label>
                <input
                  type="number"
                  placeholder="e.g. 3500"
                  value={minYield}
                  onChange={(e) => setMinYield(e.target.value)}
                  className="w-full text-xs rounded bg-background border border-border px-3 py-1.5 text-foreground font-mono focus:outline-none focus:ring-1 focus:ring-amber-500"
                />
              </div>

              <div>
                <label className="block text-[11px] text-muted-foreground mb-1">Max Resource Shift (%)</label>
                <input
                  type="number"
                  placeholder="e.g. 20"
                  value={maxResource}
                  onChange={(e) => setMaxResource(e.target.value)}
                  className="w-full text-xs rounded bg-background border border-border px-3 py-1.5 text-foreground font-mono focus:outline-none focus:ring-1 focus:ring-amber-500"
                />
              </div>

              <div>
                <label className="block text-[11px] text-muted-foreground mb-1">Max Acceptable Risk (0–100)</label>
                <input
                  type="number"
                  placeholder="e.g. 60"
                  value={maxRisk}
                  onChange={(e) => setMaxRisk(e.target.value)}
                  className="w-full text-xs rounded bg-background border border-border px-3 py-1.5 text-foreground font-mono focus:outline-none focus:ring-1 focus:ring-amber-500"
                />
              </div>

              <div>
                <label className="block text-[11px] text-muted-foreground mb-1">Min Model Test R²</label>
                <input
                  type="number"
                  step="0.05"
                  placeholder="e.g. 0.70"
                  value={minReliability}
                  onChange={(e) => setMinReliability(e.target.value)}
                  className="w-full text-xs rounded bg-background border border-border px-3 py-1.5 text-foreground font-mono focus:outline-none focus:ring-1 focus:ring-amber-500"
                />
              </div>
            </div>

            <div className="pt-3">
              <button
                type="submit"
                disabled={isLoading}
                className="w-full py-2.5 rounded-lg bg-amber-600 hover:bg-amber-500 text-white font-bold text-xs flex items-center justify-center gap-2 shadow-lg shadow-amber-900/30 transition-all disabled:opacity-50"
              >
                {isLoading ? 'Solving Decision Problem...' : (
                  <>
                    <Play className="w-4 h-4 fill-current" />
                    Solve Multi-Objective Optimization
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </form>

      {/* Recommended Candidate Output */}
      {rec && (
        <div className="p-5 rounded-xl border border-emerald-500/30 bg-emerald-500/5 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border/40 pb-3">
            <div className="flex items-center gap-2">
              <Award className="w-6 h-6 text-emerald-400" />
              <div>
                <div className="text-xs font-mono uppercase text-emerald-400 font-bold">Recommended Pareto Strategy</div>
                <h4 className="text-base font-bold text-foreground">{rec.scenario_name}</h4>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="green">Score: {rec.decision_score}/100</Badge>
              <Badge variant="blue">Rank #{rec.rank}</Badge>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
            <div className="p-2.5 rounded bg-card/60 border border-border/50">
              <div className="text-muted-foreground text-[10px]">Projected Yield</div>
              <div className="text-sm font-bold text-emerald-400">{rec.projected_yield.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} kg/ha</div>
              <div className="text-[10px] text-muted-foreground">{rec.yield_delta > 0 ? `+${rec.yield_delta.toFixed(1)}` : rec.yield_delta.toFixed(1)} kg/ha</div>
            </div>

            <div className="p-2.5 rounded bg-card/60 border border-border/50">
              <div className="text-muted-foreground text-[10px]">Risk Score</div>
              <div className="text-sm font-bold text-foreground">{rec.risk_score.toFixed(1)}/100</div>
              <div className="text-[10px] text-emerald-400">Within Threshold</div>
            </div>

            <div className="p-2.5 rounded bg-card/60 border border-border/50">
              <div className="text-muted-foreground text-[10px]">Resource Shift</div>
              <div className="text-sm font-bold text-cyan-400">{rec.resource_change_pct.toFixed(1)}%</div>
              <div className="text-[10px] text-muted-foreground">Cropland Allocation</div>
            </div>

            <div className="p-2.5 rounded bg-card/60 border border-border/50">
              <div className="text-muted-foreground text-[10px]">Pareto Status</div>
              <div className="text-sm font-bold text-purple-400">Non-Dominated</div>
              <div className="text-[10px] text-muted-foreground">Optimal Frontier</div>
            </div>
          </div>

          {/* Tradeoff Summary */}
          <div className="space-y-2 text-xs font-sans">
            <div className="text-muted-foreground">
              <span className="font-semibold text-foreground">Decision Trade-off: </span>
              {rec.tradeoff_summary}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
              <div className="space-y-1">
                <span className="text-[11px] font-bold text-emerald-400 uppercase font-mono">Key Strengths</span>
                <ul className="list-disc list-inside text-muted-foreground text-xs space-y-0.5">
                  {rec.strengths.map((s, idx) => (
                    <li key={idx}>{s}</li>
                  ))}
                </ul>
              </div>

              <div className="space-y-1">
                <span className="text-[11px] font-bold text-amber-400 uppercase font-mono">Trade-off Considerations</span>
                <ul className="list-disc list-inside text-muted-foreground text-xs space-y-0.5">
                  {rec.limitations.map((l, idx) => (
                    <li key={idx}>{l}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </Card>
  )
}
