import React, { useState, useEffect } from 'react'
import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { Sliders, Play, RotateCcw, AlertTriangle, Sparkles } from 'lucide-react'
import { useStates, useDistricts } from '../../services/api'

interface ScenarioBuilderProps {
  onRunSimulation: (params: {
    state: string
    district?: string
    horizon: number
    scenario_type: string
    modifications: Record<string, number>
  }) => void
  isLoading?: boolean
}

export const ScenarioBuilder: React.FC<ScenarioBuilderProps> = ({
  onRunSimulation,
  isLoading = false
}) => {
  const { data: statesData } = useStates()
  const [selectedState, setSelectedState] = useState('Punjab')
  const [selectedDistrict, setSelectedDistrict] = useState('')
  const [horizon, setHorizon] = useState(1)
  const [scenarioType, setScenarioType] = useState('moderate_improvement')

  // Sliders for supported agricultural features
  const [riceAreaPct, setRiceAreaPct] = useState(12.0)
  const [yieldLagPct, setYieldLagPct] = useState(10.0)
  const [rollingYieldPct, setRollingYieldPct] = useState(7.0)
  const [totalCroppedAreaPct, setTotalCroppedAreaPct] = useState(5.0)

  const { data: districtsData } = useDistricts({ state: selectedState })
  const districts = districtsData?.data ? Array.from(new Set(districtsData.data.map((d: any) => d.district))) : []
  const availableStates = statesData?.data ? Array.from(new Set(statesData.data.map((s: any) => s.state))) : ['Punjab', 'Haryana', 'Tamil Nadu', 'Uttar Pradesh']

  // Auto-fill archetype presets
  useEffect(() => {
    if (scenarioType === 'baseline') {
      setRiceAreaPct(0)
      setYieldLagPct(0)
      setRollingYieldPct(0)
      setTotalCroppedAreaPct(0)
    } else if (scenarioType === 'conservative_improvement') {
      setRiceAreaPct(5)
      setYieldLagPct(5)
      setRollingYieldPct(3)
      setTotalCroppedAreaPct(2)
    } else if (scenarioType === 'moderate_improvement') {
      setRiceAreaPct(12)
      setYieldLagPct(10)
      setRollingYieldPct(7)
      setTotalCroppedAreaPct(5)
    } else if (scenarioType === 'stress_scenario') {
      setRiceAreaPct(-15)
      setYieldLagPct(-15)
      setRollingYieldPct(-10)
      setTotalCroppedAreaPct(-8)
    }
  }, [scenarioType])

  const handleReset = () => {
    setScenarioType('baseline')
    setRiceAreaPct(0)
    setYieldLagPct(0)
    setRollingYieldPct(0)
    setTotalCroppedAreaPct(0)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onRunSimulation({
      state: selectedState,
      district: selectedDistrict && selectedDistrict !== 'all' ? selectedDistrict : undefined,
      horizon,
      scenario_type: scenarioType,
      modifications: {
        rice_area_pct: riceAreaPct,
        historical_yield_lag_pct: yieldLagPct,
        rolling_yield_pct: rollingYieldPct,
        total_cropped_area_pct: totalCroppedAreaPct
      }
    })
  }

  return (
    <Card className="p-6 space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border/40 pb-4">
        <div>
          <h2 className="text-lg font-bold text-foreground flex items-center gap-2">
            <Sliders className="w-5 h-5 text-emerald-400" />
            <span>Agricultural What-If Scenario Builder</span>
          </h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            Configure empirical input perturbations to project hypothetical yield responses under zero-leakage constraints.
          </p>
        </div>
        <Badge variant="blue">Exogenous RF Pipeline</Badge>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Top Controls: Location & Archetype */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-xs font-semibold text-foreground mb-1">Target State</label>
            <select
              value={selectedState}
              onChange={(e) => {
                setSelectedState(e.target.value)
                setSelectedDistrict('')
              }}
              className="w-full text-xs rounded-lg bg-background border border-border px-3 py-2 text-foreground focus:outline-none focus:ring-1 focus:ring-emerald-500"
            >
              {availableStates.map((s: string) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-foreground mb-1">Target District</label>
            <select
              value={selectedDistrict}
              onChange={(e) => setSelectedDistrict(e.target.value)}
              className="w-full text-xs rounded-lg bg-background border border-border px-3 py-2 text-foreground focus:outline-none focus:ring-1 focus:ring-emerald-500"
            >
              <option value="">All / State Representative</option>
              {districts.map((d: string) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-foreground mb-1">Forecast Horizon</label>
            <select
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
              className="w-full text-xs rounded-lg bg-background border border-border px-3 py-2 text-foreground focus:outline-none focus:ring-1 focus:ring-emerald-500 font-mono"
            >
              <option value={1}>1 Year (t+1 Horizon)</option>
              <option value={2}>2 Years (t+2 Horizon)</option>
              <option value={3}>3 Years (t+3 Horizon)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-foreground mb-1">Scenario Archetype</label>
            <select
              value={scenarioType}
              onChange={(e) => setScenarioType(e.target.value)}
              className="w-full text-xs rounded-lg bg-background border border-border px-3 py-2 text-foreground focus:outline-none focus:ring-1 focus:ring-emerald-500 font-medium"
            >
              <option value="baseline">Baseline (Δ = 0)</option>
              <option value="conservative_improvement">Conservative (+5% Area/Yield)</option>
              <option value="moderate_improvement">Moderate (+12% Area, +10% Yield)</option>
              <option value="stress_scenario">Stress Scenario (-15% Contraction)</option>
              <option value="custom">Custom Parameters</option>
            </select>
          </div>
        </div>

        {/* Feature Sliders */}
        <div className="p-4 rounded-xl bg-card/60 border border-border/50 space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground font-mono">
              Supported Agricultural Feature Perturbations
            </span>
            <button
              type="button"
              onClick={handleReset}
              className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1 transition-colors"
            >
              <RotateCcw className="w-3 h-3" /> Reset to Baseline
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Rice Area Slider */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="font-medium text-foreground">Rice Acreage Allocation Change</span>
                <span className={`font-mono font-bold ${riceAreaPct > 0 ? 'text-emerald-400' : (riceAreaPct < 0 ? 'text-red-400' : 'text-slate-400')}`}>
                  {riceAreaPct > 0 ? `+${riceAreaPct}%` : `${riceAreaPct}%`}
                </span>
              </div>
              <input
                type="range"
                min="-30"
                max="30"
                step="1"
                value={riceAreaPct}
                onChange={(e) => {
                  setRiceAreaPct(Number(e.target.value))
                  setScenarioType('custom')
                }}
                className="w-full h-1.5 bg-muted rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
              <div className="flex justify-between text-[10px] text-muted-foreground font-mono">
                <span>-30% Contraction</span>
                <span>0%</span>
                <span>+30% Expansion</span>
              </div>
            </div>

            {/* Historical Yield Lag Slider */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="font-medium text-foreground">Prior Year Yield Baseline (t-1 Lag)</span>
                <span className={`font-mono font-bold ${yieldLagPct > 0 ? 'text-emerald-400' : (yieldLagPct < 0 ? 'text-red-400' : 'text-slate-400')}`}>
                  {yieldLagPct > 0 ? `+${yieldLagPct}%` : `${yieldLagPct}%`}
                </span>
              </div>
              <input
                type="range"
                min="-30"
                max="30"
                step="1"
                value={yieldLagPct}
                onChange={(e) => {
                  setYieldLagPct(Number(e.target.value))
                  setScenarioType('custom')
                }}
                className="w-full h-1.5 bg-muted rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
              <div className="flex justify-between text-[10px] text-muted-foreground font-mono">
                <span>-30% Shock</span>
                <span>0%</span>
                <span>+30% Gain</span>
              </div>
            </div>

            {/* Rolling Yield Slider */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="font-medium text-foreground">3-Yr Rolling Mean Yield Trend</span>
                <span className={`font-mono font-bold ${rollingYieldPct > 0 ? 'text-emerald-400' : (rollingYieldPct < 0 ? 'text-red-400' : 'text-slate-400')}`}>
                  {rollingYieldPct > 0 ? `+${rollingYieldPct}%` : `${rollingYieldPct}%`}
                </span>
              </div>
              <input
                type="range"
                min="-20"
                max="20"
                step="1"
                value={rollingYieldPct}
                onChange={(e) => {
                  setRollingYieldPct(Number(e.target.value))
                  setScenarioType('custom')
                }}
                className="w-full h-1.5 bg-muted rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
              <div className="flex justify-between text-[10px] text-muted-foreground font-mono">
                <span>-20%</span>
                <span>0%</span>
                <span>+20%</span>
              </div>
            </div>

            {/* Total Cropped Area Slider */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="font-medium text-foreground">Total Cropped Area Shift</span>
                <span className={`font-mono font-bold ${totalCroppedAreaPct > 0 ? 'text-emerald-400' : (totalCroppedAreaPct < 0 ? 'text-red-400' : 'text-slate-400')}`}>
                  {totalCroppedAreaPct > 0 ? `+${totalCroppedAreaPct}%` : `${totalCroppedAreaPct}%`}
                </span>
              </div>
              <input
                type="range"
                min="-20"
                max="20"
                step="1"
                value={totalCroppedAreaPct}
                onChange={(e) => {
                  setTotalCroppedAreaPct(Number(e.target.value))
                  setScenarioType('custom')
                }}
                className="w-full h-1.5 bg-muted rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
              <div className="flex justify-between text-[10px] text-muted-foreground font-mono">
                <span>-20%</span>
                <span>0%</span>
                <span>+20%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Action Button */}
        <div className="flex items-center justify-end gap-3 pt-2">
          <button
            type="submit"
            disabled={isLoading}
            className="px-6 py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs flex items-center gap-2 shadow-lg shadow-emerald-900/30 transition-all disabled:opacity-50"
          >
            {isLoading ? (
              <>Simulating What-If Response...</>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                Run Scenario Simulation
              </>
            )}
          </button>
        </div>
      </form>
    </Card>
  )
}
