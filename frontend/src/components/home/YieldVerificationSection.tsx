import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import { Calculator, AlertCircle, ArrowRight, CheckCircle2, ChevronRight } from 'lucide-react'
import { formatNumber, formatYield } from '../../lib/utils'
import { Button } from '../ui/Button'
import { Badge } from '../ui/Badge'
import { useStates } from '../../services/api'

export const YieldVerificationSection: React.FC = () => {
  const { data: statesData } = useStates()
  const states = statesData?.data || []

  const [calcState, setCalcState] = useState('Punjab')
  const [calcArea, setCalcArea] = useState('250.0')
  const [calcProduction, setCalcProduction] = useState('1000.0')

  const areaNum = parseFloat(calcArea) || 1
  const prodNum = parseFloat(calcProduction) || 0
  const liveYieldKgHa = (prodNum / areaNum) * 1000
  const liveYieldQtlAcre = (liveYieldKgHa / 100) * 0.404686

  const currentState = states.find(s => s.state === calcState) || states[0] || {
    state: 'Punjab',
    rank: 1,
    average_yield: 3991.08,
  }
  const diffFromStateAvg = liveYieldKgHa - currentState.average_yield

  return (
    <section id="post-harvest-verification" className="py-16 md:py-24 border-b border-border/80 bg-background scroll-mt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-10">
        
        {/* Header */}
        <div className="max-w-3xl space-y-3">
          <div className="inline-flex items-center gap-2">
            <Badge variant="outline" className="text-[10px] font-mono font-bold tracking-wider uppercase">
              MODE: POST-HARVEST / DERIVED
            </Badge>
          </div>
          <h2 className="text-2xl sm:text-4xl font-extrabold text-foreground tracking-tight">
            Post-Harvest Yield Verification
          </h2>
          <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
            Check the mathematical consistency of reported production and cultivated area using the standard agronomic formula.
          </p>
        </div>

        {/* Verification Card & Context Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* Left: Interactive Calculation Module */}
          <div className="lg:col-span-7 bg-card border border-border rounded-xl p-6 sm:p-8 shadow-subtle space-y-6">
            
            <div className="flex items-center justify-between border-b border-border/60 pb-3">
              <div className="flex items-center gap-2">
                <Calculator className="w-4 h-4 text-primary" />
                <h3 className="text-sm font-bold text-foreground">Agronomic Ratio Calculator</h3>
              </div>
              <span className="text-[11px] font-mono text-muted-foreground">Formula: (Prod / Area) × 1000</span>
            </div>

            {/* Inputs Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              
              {/* Target State */}
              <div className="space-y-1.5 sm:col-span-3">
                <label className="text-xs font-semibold text-foreground flex justify-between">
                  <span>State Benchmark Reference</span>
                  <span className="text-[11px] text-muted-foreground font-mono">
                    Avg: {formatYield(currentState.average_yield)}
                  </span>
                </label>
                <select
                  value={calcState}
                  onChange={(e) => setCalcState(e.target.value)}
                  className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs font-medium focus:ring-2 focus:ring-primary focus:outline-none cursor-pointer"
                >
                  {states.length > 0 ? (
                    states.map(s => (
                      <option key={s.state} value={s.state}>{s.state} (Rank #{s.rank})</option>
                    ))
                  ) : (
                    <option value="Punjab">Punjab (Rank #1)</option>
                  )}
                </select>
              </div>

              {/* Area */}
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-foreground">
                  Cultivated Area (ha)
                </label>
                <input
                  type="number"
                  step="10"
                  value={calcArea}
                  onChange={(e) => setCalcArea(e.target.value)}
                  className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs font-mono"
                  placeholder="e.g. 250"
                />
              </div>

              {/* Production */}
              <div className="space-y-1.5 sm:col-span-2">
                <label className="text-xs font-semibold text-foreground">
                  Reported Production (tonnes)
                </label>
                <input
                  type="number"
                  step="10"
                  value={calcProduction}
                  onChange={(e) => setCalcProduction(e.target.value)}
                  className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs font-mono"
                  placeholder="e.g. 1000"
                />
              </div>
            </div>

            {/* Derived Output Display */}
            <div className="p-5 rounded-xl bg-muted/40 border border-border/80 space-y-3">
              <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-2">
                <div>
                  <span className="text-[10px] uppercase font-bold text-muted-foreground tracking-wider">
                    Derived Agronomic Yield
                  </span>
                  <div className="text-2xl sm:text-3xl font-black text-foreground font-mono mt-0.5">
                    {formatYield(liveYieldKgHa)}
                  </div>
                </div>
                <div className="text-left sm:text-right">
                  <span className="text-[10px] uppercase font-bold text-muted-foreground tracking-wider">
                    Imperial / Indian Conversion
                  </span>
                  <div className="text-sm font-bold text-foreground font-mono mt-0.5">
                    {formatNumber(liveYieldQtlAcre, 2)} Qtl / Acre
                  </div>
                </div>
              </div>

              {/* Comparison against state historical mean */}
              <div className="pt-2 border-t border-border/50 flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Variance vs {calcState} Historical Mean:</span>
                <span className={`font-bold font-mono ${diffFromStateAvg >= 0 ? 'text-emerald-700 dark:text-emerald-400' : 'text-amber-700 dark:text-amber-400'}`}>
                  {diffFromStateAvg >= 0 ? `+${formatNumber(diffFromStateAvg, 1)}` : formatNumber(diffFromStateAvg, 1)} kg/ha
                </span>
              </div>
            </div>

            {/* Mandatory Scientific Guard Note */}
            <div className="p-3.5 rounded-lg border border-border bg-muted/30 text-xs text-muted-foreground flex items-start gap-2.5">
              <AlertCircle className="w-4 h-4 text-muted-foreground shrink-0 mt-0.5" />
              <p className="leading-relaxed">
                <strong className="text-foreground font-semibold">Scientific Notice:</strong> This calculation derives yield from reported agricultural quantities. It is not an out-of-sample forecast.
              </p>
            </div>

          </div>

          {/* Right: Methodological Explanation */}
          <div className="lg:col-span-5 space-y-5">
            <div className="rounded-xl border border-border bg-card p-6 space-y-4 shadow-subtle">
              <h3 className="text-sm font-bold text-foreground">
                Why Post-Harvest Verification Matters
              </h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Statistical reporting agencies and crop auditors require an auditable check on historical data submissions. Post-harvest verification guarantees data integrity by flagging mathematical discrepancies before records enter analytical pipelines.
              </p>
              
              <div className="space-y-2.5 text-xs text-muted-foreground pt-1">
                <div className="flex items-start gap-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-primary shrink-0 mt-0.5" />
                  <span>Validates transcription consistency across district statistical returns.</span>
                </div>
                <div className="flex items-start gap-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-primary shrink-0 mt-0.5" />
                  <span>Identifies impossible acreage/tonnage reporting entries.</span>
                </div>
                <div className="flex items-start gap-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-primary shrink-0 mt-0.5" />
                  <span>Establishes historical ground-truth benchmarks for pre-season models.</span>
                </div>
              </div>

              <div className="pt-3 border-t border-border/60">
                <Link
                  to="/calculator"
                  className="text-xs font-bold text-primary hover:underline flex items-center gap-1.5"
                >
                  <span>Open Full Multi-District Verification Tool</span>
                  <ChevronRight className="w-3.5 h-3.5" />
                </Link>
              </div>
            </div>

            <div className="p-5 rounded-xl border border-primary/20 bg-primary/5 space-y-2 text-xs">
              <span className="font-bold text-foreground">Looking for Pre-Season Estimates?</span>
              <p className="text-muted-foreground leading-relaxed">
                If you need genuine pre-season yield estimates generated before harvest without reported production numbers, use our governed Pre-Season Forecasting Engine.
              </p>
              <Link
                to="/forecast"
                className="inline-flex items-center gap-1 font-bold text-primary hover:underline pt-1"
              >
                <span>Launch Pre-Season Forecaster</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>
          </div>

        </div>

      </div>
    </section>
  )
}
