import React, { useState, useEffect } from 'react'
import {
  Sliders,
  Sparkles,
  RotateCcw,
  ShieldCheck,
  AlertTriangle,
  Info,
  CheckCircle2,
  TrendingUp,
  TrendingDown,
  Loader2,
  ArrowRight,
  Database,
  Layers,
  HelpCircle,
  FlaskConical
} from 'lucide-react'
import { useFilters, usePredictScenario } from '../services/api'
import { ScenarioSimulationResponse } from '../types/intelligence'
import { ScenarioSlider } from '../components/intelligence/ScenarioSlider'
import { ScenarioComparison } from '../components/intelligence/ScenarioComparison'
import { ScenarioMetrics } from '../components/intelligence/ScenarioMetrics'
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from '../components/ui/Card'
import { Button } from '../components/ui/Button'
import { Badge } from '../components/ui/Badge'

export const ScenarioLab: React.FC = () => {
  const { data: filtersData } = useFilters()
  const availableStates = filtersData?.states || [
    'Punjab', 'Haryana', 'Tamil Nadu', 'West Bengal', 'Uttar Pradesh',
    'Andhra Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Gujarat',
    'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh',
    'Maharashtra', 'Orissa', 'Rajasthan', 'Telangana', 'Uttarakhand'
  ]
  const availableYears = filtersData?.years || [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]

  // Baseline Selection State
  const [selectedState, setSelectedState] = useState<string>('Punjab')
  const [selectedDistrict, setSelectedDistrict] = useState<string>('Ludhiana')
  const [selectedYear, setSelectedYear] = useState<number>(2017)

  // Simulation Sliders State
  const [baseArea, setBaseArea] = useState<number>(250.0)
  const [scenArea, setScenArea] = useState<number>(275.0)
  const [scenTotArea, setScenTotArea] = useState<number>(350.0)
  const [scenWheat, setScenWheat] = useState<number>(120.0)
  const [scenCotton, setScenCotton] = useState<number>(10.0)
  const [scenSugar, setScenSugar] = useState<number>(15.0)
  const [scenLag, setScenLag] = useState<number>(3980.0)
  const [scenRoll, setScenRoll] = useState<number>(3900.0)

  const scenarioMutation = usePredictScenario()
  const [result, setResult] = useState<ScenarioSimulationResponse | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  // Run initial simulation on load
  const executeSimulation = async () => {
    setErrorMsg(null)
    try {
      const res = await scenarioMutation.mutateAsync({
        year: selectedYear,
        state: selectedState,
        district: selectedDistrict,
        baseline_rice_area: baseArea,
        scenario_rice_area: scenArea,
        scenario_total_cropped_area: scenTotArea,
        scenario_wheat_area: scenWheat,
        scenario_cotton_area: scenCotton,
        scenario_sugarcane_area: scenSugar,
        scenario_historical_yield_lag: scenLag,
        scenario_rolling_yield: scenRoll
      })
      setResult(res)
    } catch (err: any) {
      setErrorMsg(err.message || 'Scenario simulation failed.')
    }
  }

  useEffect(() => {
    executeSimulation()
  }, [selectedState, selectedYear])

  const resetAllSliders = () => {
    setScenArea(baseArea)
    setScenTotArea(350.0)
    setScenWheat(120.0)
    setScenCotton(10.0)
    setScenSugar(15.0)
    setScenLag(3980.0)
    setScenRoll(3900.0)
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
      {/* Header */}
      <div className="space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-500/10 text-sky-600 dark:text-sky-400 text-xs font-semibold">
          <FlaskConical className="w-3.5 h-3.5" />
          <span>What-If Agricultural Simulation Lab</span>
        </div>
        <h1 className="text-2xl sm:text-4xl font-extrabold text-foreground tracking-tight">
          Scenario Intelligence Lab
        </h1>
        <p className="text-xs sm:text-sm text-muted-foreground max-w-3xl leading-relaxed">
          Explore how modifications in cultivated rice acreage, land allocation shares, and historical productivity lags influence model-estimated yield and composite risk.
        </p>
      </div>

      {/* SECTION 1: BASELINE CONFIGURATION & SCENARIO CONTROLS */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Baseline & Sliders (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          <Card className="border-border">
            <CardHeader className="pb-3 border-b border-border/40">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-bold flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-sky-500" />
                  <span>1. Simulation Baseline & District Setup</span>
                </CardTitle>
                <Badge variant="outline">Input Setup</Badge>
              </div>
              <CardDescription className="text-xs">
                Select region and configure what-if variables
              </CardDescription>
            </CardHeader>

            <CardContent className="space-y-4 pt-4 text-xs">
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <label className="font-semibold text-foreground">State</label>
                  <select
                    value={selectedState}
                    onChange={(e) => setSelectedState(e.target.value)}
                    className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs font-medium focus:ring-2 focus:ring-sky-500 focus:outline-none"
                  >
                    {availableStates.map(s => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </div>

                <div className="space-y-1.5">
                  <label className="font-semibold text-foreground">Simulation Year</label>
                  <select
                    value={selectedYear}
                    onChange={(e) => setSelectedYear(Number(e.target.value))}
                    className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs font-mono focus:ring-2 focus:ring-sky-500 focus:outline-none"
                  >
                    {availableYears.map(y => (
                      <option key={y} value={y}>{y}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="font-semibold text-foreground">District Name</label>
                <input
                  type="text"
                  value={selectedDistrict}
                  onChange={(e) => setSelectedDistrict(e.target.value)}
                  placeholder="e.g. Ludhiana, Patiala, Thanjavur"
                  className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs font-medium focus:ring-2 focus:ring-sky-500 focus:outline-none"
                />
              </div>

              {/* Slider Controls */}
              <div className="space-y-3 pt-3 border-t border-border">
                <div className="flex items-center justify-between">
                  <span className="font-bold text-foreground text-xs uppercase tracking-wider">
                    2. Scenario Input Modifiers
                  </span>
                  <button
                    type="button"
                    onClick={resetAllSliders}
                    className="text-[11px] text-sky-600 dark:text-sky-400 hover:underline font-semibold cursor-pointer"
                  >
                    Reset All
                  </button>
                </div>

                <ScenarioSlider
                  label="Cultivated Rice Area"
                  unit="'000 ha"
                  baselineValue={baseArea}
                  currentValue={scenArea}
                  min={10}
                  max={600}
                  step={5}
                  onChange={setScenArea}
                  onReset={() => setScenArea(baseArea)}
                />

                <ScenarioSlider
                  label="Total Cropped Land Capacity"
                  unit="'000 ha"
                  baselineValue={350.0}
                  currentValue={scenTotArea}
                  min={50}
                  max={1000}
                  step={10}
                  onChange={setScenTotArea}
                  onReset={() => setScenTotArea(350.0)}
                />

                <ScenarioSlider
                  label="Prior-Year Rice Yield (t-1)"
                  unit="kg/ha"
                  baselineValue={3980.0}
                  currentValue={scenLag}
                  min={1000}
                  max={6000}
                  step={50}
                  onChange={setScenLag}
                  onReset={() => setScenLag(3980.0)}
                />

                <ScenarioSlider
                  label="Wheat Cropped Acreage"
                  unit="'000 ha"
                  baselineValue={120.0}
                  currentValue={scenWheat}
                  min={0}
                  max={400}
                  step={5}
                  onChange={setScenWheat}
                  onReset={() => setScenWheat(120.0)}
                />
              </div>

              {errorMsg && (
                <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-600 dark:text-red-400 text-xs flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  <span>{errorMsg}</span>
                </div>
              )}
            </CardContent>

            <CardFooter className="pt-2">
              <Button
                onClick={executeSimulation}
                disabled={scenarioMutation.isPending}
                className="w-full gap-2 text-xs font-bold shadow-md shadow-sky-500/10"
              >
                {scenarioMutation.isPending ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Projecting Model Trajectory...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    <span>Run Scenario Simulation</span>
                  </>
                )}
              </Button>
            </CardFooter>
          </Card>
        </div>

        {/* Right Column: Simulation Output & Comparative Breakdown (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {result ? (
            <>
              {/* SECTION 4: DUAL COLUMN COMPARISON */}
              <ScenarioComparison
                baseline={result.baseline}
                scenario={result.scenario}
                stateName={result.state}
                districtName={result.district}
              />

              {/* SECTION 5: CHANGE ANALYSIS METRICS */}
              <ScenarioMetrics
                delta={result.delta}
                baselineYield={result.baseline.predicted_yield}
                scenarioYield={result.scenario.predicted_yield}
                baselineRisk={result.baseline.risk_score}
                scenarioRisk={result.scenario.risk_score}
              />

              {/* SECTION 6: MODEL INTERPRETATION & CHANGED FEATURES */}
              <Card className="border-border">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-bold">Why did the estimate change? (Feature Shifts)</CardTitle>
                  <CardDescription className="text-xs">
                    Model response to modified agricultural feature inputs
                  </CardDescription>
                </CardHeader>

                <CardContent className="space-y-4 text-xs">
                  {result.changed_features.length > 0 ? (
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs border-collapse">
                        <thead>
                          <tr className="border-b border-border bg-muted/40 text-muted-foreground">
                            <th className="py-2 px-3 font-semibold">Modified Parameter</th>
                            <th className="py-2 px-3 font-semibold">Baseline</th>
                            <th className="py-2 px-3 font-semibold">Scenario</th>
                            <th className="py-2 px-3 font-semibold">Delta</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border/50 font-mono">
                          {result.changed_features.map((cf, idx) => (
                            <tr key={idx} className="hover:bg-muted/20">
                              <td className="py-2 px-3 font-sans font-medium text-foreground">{cf.feature_name}</td>
                              <td className="py-2 px-3">{cf.baseline_value}</td>
                              <td className="py-2 px-3 font-bold text-foreground">{cf.scenario_value}</td>
                              <td className="py-2 px-3">
                                <span className={cf.absolute_change >= 0 ? 'text-emerald-600 dark:text-emerald-400 font-bold' : 'text-amber-600 dark:text-amber-400 font-bold'}>
                                  {cf.absolute_change >= 0 ? '+' : ''}{cf.absolute_change} ({cf.percent_change >= 0 ? '+' : ''}{cf.percent_change}%)
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className="text-muted-foreground italic text-xs py-2">
                      No modified parameters detected. Adjust the sliders on the left to simulate agricultural interventions.
                    </p>
                  )}

                  <p className="text-[11px] text-muted-foreground leading-relaxed pt-2 border-t border-border/60">
                    {result.explanation}
                  </p>

                  {result.warnings.length > 0 && (
                    <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-800 dark:text-amber-300 text-xs space-y-1">
                      <div className="flex items-center gap-1.5 font-bold">
                        <AlertTriangle className="w-3.5 h-3.5 text-amber-500" />
                        <span>Simulation Notice:</span>
                      </div>
                      <ul className="list-disc list-inside space-y-0.5 text-[11px]">
                        {result.warnings.map((w, idx) => (
                          <li key={idx}>{w}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* SECTION 7: SCIENTIFIC DISCLAIMER */}
              <div className="p-3.5 rounded-xl border border-border/80 bg-muted/30 text-[11px] text-muted-foreground flex items-start gap-2.5">
                <Info className="w-4 h-4 text-muted-foreground shrink-0 mt-0.5" />
                <p className="leading-relaxed">
                  <strong>Scientific Notice:</strong> {result.disclaimer}
                </p>
              </div>
            </>
          ) : (
            <Card className="h-full flex items-center justify-center p-12 text-center text-muted-foreground text-xs">
              <div className="space-y-3 max-w-sm">
                <Sliders className="w-10 h-10 mx-auto text-muted-foreground/40" />
                <p className="font-semibold text-foreground">Ready for Scenario Simulation</p>
                <p className="text-[11px]">Adjust what-if parameters and click Run Scenario Simulation to generate comparative projections.</p>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
