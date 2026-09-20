import React, { useState, useEffect } from 'react'
import {
  Calculator,
  Sparkles,
  CloudRain,
  ShieldCheck,
  RotateCcw,
  Info,
  AlertTriangle,
  TrendingUp,
  CheckCircle2,
  Lock,
  Loader2,
  History,
  Trash2,
  ArrowRight,
  Database,
  Layers,
  HelpCircle,
  Sliders,
  Maximize2,
  ShieldAlert,
  Flame,
  Activity,
  Award
} from 'lucide-react'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  Cell
} from 'recharts'
import { formatNumber, formatYield } from '../lib/utils'
import { Button } from '../components/ui/Button'
import { Badge } from '../components/ui/Badge'
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from '../components/ui/Card'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/ui/Tabs'
import {
  useFilters,
  useStates,
  useModels,
  usePredictPostHarvest,
  usePredictPreSeason,
  usePredictPreSeasonAdvanced,
  usePredictRisk,
  useExplainPrediction,
  useDetectAnomaly
} from '../services/api'
import {
  PostHarvestPredictResponse,
  PreSeasonPredictResponse,
  PreSeasonAdvancedPredictResponse,
  RiskAssessmentResponse,
  ExplainabilityResponse,
  AnomalyDetectionResponse,
  ModelMetadataItem,
  PredictionHistoryEntry
} from '../types/model'

const HISTORY_STORAGE_KEY = 'agri_prediction_history_v1'

export const YieldCalculator: React.FC = () => {
  const [calcMode, setCalcMode] = useState<'post-harvest' | 'pre-season'>('post-harvest')
  const [preSeasonSubMode, setPreSeasonSubMode] = useState<'basic' | 'advanced'>('advanced')

  // API Data
  const { data: filtersData } = useFilters()
  const { data: statesData } = useStates()
  const { data: modelsData } = useModels()

  const availableStates = filtersData?.states || [
    'Punjab', 'Haryana', 'Uttar Pradesh', 'Tamil Nadu', 'West Bengal',
    'Andhra Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Gujarat',
    'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh',
    'Maharashtra', 'Orissa', 'Rajasthan', 'Telangana', 'Uttarakhand'
  ]
  const availableYears = filtersData?.years || [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]

  // Model Selection
  const allModels: ModelMetadataItem[] = modelsData?.models || []
  const compatibleModels = allModels.filter(m =>
    calcMode === 'post-harvest' ? m.is_post_harvest_only || m.mode_compatibility.includes('post-harvest') : !m.is_post_harvest_only
  )
  const [selectedModelId, setSelectedModelId] = useState<string>('rf-post-harvest')

  useEffect(() => {
    if (calcMode === 'post-harvest') {
      setSelectedModelId('rf-post-harvest')
    } else {
      setSelectedModelId(preSeasonSubMode === 'advanced' ? 'rf-pre-season-exogenous' : 'rf-pre-season')
    }
  }, [calcMode, preSeasonSubMode])

  // Form Inputs: Post-Harvest
  const [phYear, setPhYear] = useState<number>(2017)
  const [phState, setPhState] = useState<string>('Punjab')
  const [phDistrict, setPhDistrict] = useState<string>('Ludhiana')
  const [phArea, setPhArea] = useState<string>('250.0')
  const [phProduction, setPhProduction] = useState<string>('1000.0')

  // Form Inputs: Pre-Season
  const [psYear, setPsYear] = useState<number>(2017)
  const [psState, setPsState] = useState<string>('Punjab')
  const [psDistrict, setPsDistrict] = useState<string>('Ludhiana')
  const [psArea, setPsArea] = useState<string>('250.0')

  // Advanced Pre-Season Inputs
  const [psTotalArea, setPsTotalArea] = useState<string>('350.0')
  const [psWheatArea, setPsWheatArea] = useState<string>('120.0')
  const [psCottonArea, setPsCottonArea] = useState<string>('10.0')
  const [psSugarcaneArea, setPsSugarcaneArea] = useState<string>('15.0')
  const [psLagYield, setPsLagYield] = useState<string>('4200.0')
  const [showAdvancedInputs, setShowAdvancedInputs] = useState<boolean>(false)

  // Prediction Mutation States
  const postHarvestMutation = usePredictPostHarvest()
  const preSeasonMutation = usePredictPreSeason()
  const preSeasonAdvMutation = usePredictPreSeasonAdvanced()
  const riskMutation = usePredictRisk()
  const explainMutation = useExplainPrediction()
  const anomalyMutation = useDetectAnomaly()

  const [postHarvestResult, setPostHarvestResult] = useState<PostHarvestPredictResponse | null>(null)
  const [preSeasonResult, setPreSeasonResult] = useState<PreSeasonPredictResponse | null>(null)
  const [preSeasonAdvResult, setPreSeasonAdvResult] = useState<PreSeasonAdvancedPredictResponse | null>(null)

  // Intelligence Results
  const [riskResult, setRiskResult] = useState<RiskAssessmentResponse | null>(null)
  const [explainResult, setExplainResult] = useState<ExplainabilityResponse | null>(null)
  const [anomalyResult, setAnomalyResult] = useState<AnomalyDetectionResponse | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  // Client-side LocalStorage Prediction History
  const [history, setHistory] = useState<PredictionHistoryEntry[]>([])

  useEffect(() => {
    try {
      const saved = localStorage.getItem(HISTORY_STORAGE_KEY)
      if (saved) {
        setHistory(JSON.parse(saved))
      }
    } catch {
      // ignore
    }
  }, [])

  const saveToHistory = (entry: PredictionHistoryEntry) => {
    const updated = [entry, ...history.slice(0, 19)]
    setHistory(updated)
    try {
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(updated))
    } catch {
      // ignore
    }
  }

  const clearHistory = () => {
    setHistory([])
    localStorage.removeItem(HISTORY_STORAGE_KEY)
  }

  // Handle Post-Harvest Submit
  const handlePostHarvestSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setErrorMsg(null)
    const areaNum = parseFloat(phArea)
    const prodNum = parseFloat(phProduction)

    if (isNaN(areaNum) || areaNum <= 0) {
      setErrorMsg('Cultivated area must be a valid positive number greater than 0.')
      return
    }
    if (isNaN(prodNum) || prodNum < 0) {
      setErrorMsg('Production must be a non-negative number.')
      return
    }

    try {
      const res = await postHarvestMutation.mutateAsync({
        year: Number(phYear),
        state: phState,
        district: phDistrict || undefined,
        area: areaNum,
        production: prodNum,
      })
      setPostHarvestResult(res)

      // Trigger Anomaly and Risk Checks
      const [anomRes, riskRes] = await Promise.all([
        anomalyMutation.mutateAsync({
          year: Number(phYear),
          state: phState,
          district: phDistrict || undefined,
          area: areaNum,
          production: prodNum,
          yield: res.predicted_yield,
        }),
        riskMutation.mutateAsync({
          year: Number(phYear),
          state: phState,
          district: phDistrict || undefined,
          area: areaNum,
          predicted_yield: res.predicted_yield,
          lower_bound: res.predicted_yield * 0.95,
          upper_bound: res.predicted_yield * 1.05,
        })
      ])
      setAnomalyResult(anomRes)
      setRiskResult(riskRes)

      saveToHistory({
        id: `ph_${Date.now()}`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        mode: 'post-harvest',
        state: res.state,
        district: res.matched_district || phDistrict,
        year: res.year,
        area: res.area,
        production: res.production,
        modelName: res.model_name,
        predictedYield: res.predicted_yield,
        deterministicYield: res.deterministic_yield,
        actualYield: res.actual_yield,
        errorDelta: res.difference,
        riskLevel: riskRes.risk_level,
        riskScore: riskRes.risk_score,
      })
    } catch (err: any) {
      setErrorMsg(err.message || 'Post-harvest prediction failed.')
    }
  }

  // Handle Pre-Season Submit
  const handlePreSeasonSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setErrorMsg(null)
    const areaNum = parseFloat(psArea)

    if (isNaN(areaNum) || areaNum <= 0) {
      setErrorMsg('Cultivated area must be a valid positive number greater than 0.')
      return
    }

    try {
      if (preSeasonSubMode === 'advanced') {
        const totArea = parseFloat(psTotalArea) || undefined
        const wArea = parseFloat(psWheatArea) || undefined
        const cArea = parseFloat(psCottonArea) || undefined
        const sArea = parseFloat(psSugarcaneArea) || undefined
        const lagY = parseFloat(psLagYield) || undefined

        const res = await preSeasonAdvMutation.mutateAsync({
          year: Number(psYear),
          state: psState,
          district: psDistrict || undefined,
          area: areaNum,
          total_cropped_area: totArea,
          wheat_area: wArea,
          cotton_area: cArea,
          sugarcane_area: sArea,
          rice_yield_lag1: lagY,
        })
        setPreSeasonAdvResult(res)

        // Trigger Explainability, Risk, and Anomaly Intelligence
        const [expRes, riskRes, anomRes] = await Promise.all([
          explainMutation.mutateAsync({
            year: Number(psYear),
            state: psState,
            district: psDistrict || undefined,
            area: areaNum,
            total_cropped_area: totArea,
            wheat_area: wArea,
            cotton_area: cArea,
            sugarcane_area: sArea,
            rice_yield_lag1: lagY,
          }),
          riskMutation.mutateAsync({
            year: Number(psYear),
            state: psState,
            district: psDistrict || undefined,
            area: areaNum,
            predicted_yield: res.predicted_yield,
            lower_bound: res.uncertainty.lower_bound_10th_pct,
            upper_bound: res.uncertainty.upper_bound_90th_pct,
          }),
          anomalyMutation.mutateAsync({
            year: Number(psYear),
            state: psState,
            district: psDistrict || undefined,
            area: areaNum,
            yield: res.predicted_yield,
            total_cropped_area: totArea,
          })
        ])

        setExplainResult(expRes)
        setRiskResult(riskRes)
        setAnomalyResult(anomRes)

        saveToHistory({
          id: `ps_adv_${Date.now()}`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
          mode: 'pre-season-advanced',
          state: res.state,
          district: res.matched_district || psDistrict,
          year: res.year,
          area: res.area,
          modelName: res.model_name,
          predictedYield: res.predicted_yield,
          actualYield: res.actual_yield,
          riskLevel: riskRes.risk_level,
          riskScore: riskRes.risk_score,
        })
      } else {
        const res = await preSeasonMutation.mutateAsync({
          year: Number(psYear),
          state: psState,
          district: psDistrict || undefined,
          area: areaNum,
        })
        setPreSeasonResult(res)

        const riskRes = await riskMutation.mutateAsync({
          year: Number(psYear),
          state: psState,
          district: psDistrict || undefined,
          area: areaNum,
          predicted_yield: res.predicted_yield,
          lower_bound: res.predicted_yield * 0.85,
          upper_bound: res.predicted_yield * 1.15,
        })
        setRiskResult(riskRes)

        saveToHistory({
          id: `ps_${Date.now()}`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
          mode: 'pre-season',
          state: res.state,
          district: res.matched_district || psDistrict,
          year: res.year,
          area: res.area,
          modelName: res.model_name,
          predictedYield: res.predicted_yield,
          actualYield: res.actual_yield,
          riskLevel: riskRes.risk_level,
          riskScore: riskRes.risk_score,
        })
      }
    } catch (err: any) {
      setErrorMsg(err.message || 'Pre-season prediction failed.')
    }
  }

  // Pre-load an initial example calculation on mount
  useEffect(() => {
    if (!postHarvestResult && !postHarvestMutation.isPending) {
      postHarvestMutation.mutateAsync({
        year: 2017,
        state: 'Punjab',
        district: 'Ludhiana',
        area: 250.0,
        production: 1000.0,
      }).then(res => setPostHarvestResult(res)).catch(() => {})
    }
  }, [])

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      {/* Top Header */}
      <div className="space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-semibold">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Agricultural Prediction & Intelligence Workspace</span>
        </div>
        <h1 className="text-2xl sm:text-4xl font-extrabold text-foreground tracking-tight">
          AI Yield Intelligence Workspace
        </h1>
        <p className="text-xs sm:text-sm text-muted-foreground max-w-3xl leading-relaxed">
          Generate validated yield predictions, evaluate prediction risk scores, inspect tree-based feature contributions, and detect unusual agricultural anomalies.
        </p>
      </div>

      {/* Mode Switcher Tabs */}
      <Tabs value={calcMode} onValueChange={(val) => {
        setCalcMode(val as any)
        setErrorMsg(null)
      }}>
        <TabsList className="grid grid-cols-2 max-w-xl">
          <TabsTrigger value="post-harvest" className="gap-2 text-xs font-semibold">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-500" />
            <span>Post-Harvest Verification</span>
          </TabsTrigger>
          <TabsTrigger value="pre-season" className="gap-2 text-xs font-semibold">
            <CloudRain className="w-3.5 h-3.5 text-sky-500" />
            <span>Pre-Season Estimation</span>
          </TabsTrigger>
        </TabsList>

        {/* ========================================================================= */}
        {/* MODE A: POST-HARVEST VERIFICATION */}
        {/* ========================================================================= */}
        <TabsContent value="post-harvest" className="space-y-6">
          <div className="p-4 rounded-xl border border-emerald-500/20 bg-emerald-500/5 text-xs text-foreground flex items-start gap-3">
            <ShieldCheck className="w-5 h-5 text-emerald-600 dark:text-emerald-400 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <p className="font-bold text-emerald-700 dark:text-emerald-400">
                Mode A: Post-Harvest Verification & Algebraic Ground-Truth Audit
              </p>
              <p className="text-muted-foreground leading-relaxed text-[11px]">
                In agricultural reporting, reported Yield is defined as <code className="font-mono text-primary font-bold">(Production / Area) × 1000</code>. This mode evaluates whether machine learning models curve-fit the hyperbola accurately and compares ML predictions against the exact deterministic formula and dataset records.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Input Form Column (6 cols) */}
            <div className="lg:col-span-6 space-y-6">
              <Card>
                <CardHeader className="pb-4">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base font-bold">1. Harvest Observation Inputs</CardTitle>
                    <Badge variant="success">Post-Harvest Full Features</Badge>
                  </div>
                  <CardDescription className="text-xs">
                    Features: [Year, State Code, Cultivated Area, Harvest Production]
                  </CardDescription>
                </CardHeader>

                <form onSubmit={handlePostHarvestSubmit}>
                  <CardContent className="space-y-4 text-xs">
                    <div className="space-y-1.5">
                      <label className="font-semibold text-foreground flex items-center justify-between">
                        <span>Selected Evaluator</span>
                        <span className="text-[10px] text-muted-foreground">Persisted in Models/</span>
                      </label>
                      <select
                        value={selectedModelId}
                        onChange={(e) => setSelectedModelId(e.target.value)}
                        className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs font-medium focus:ring-2 focus:ring-primary focus:outline-none"
                      >
                        {compatibleModels.map(m => (
                          <option key={m.id} value={m.id}>
                            {m.name} ({m.badge})
                          </option>
                        ))}
                      </select>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div className="space-y-1.5">
                        <label className="font-semibold text-foreground">Agricultural Year</label>
                        <select
                          value={phYear}
                          onChange={(e) => setPhYear(Number(e.target.value))}
                          className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs font-mono focus:ring-2 focus:ring-primary focus:outline-none"
                        >
                          {availableYears.map(y => (
                            <option key={y} value={y}>{y}</option>
                          ))}
                        </select>
                      </div>

                      <div className="space-y-1.5">
                        <label className="font-semibold text-foreground">State</label>
                        <select
                          value={phState}
                          onChange={(e) => setPhState(e.target.value)}
                          className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs focus:ring-2 focus:ring-primary focus:outline-none"
                        >
                          {availableStates.map(s => (
                            <option key={s} value={s}>{s}</option>
                          ))}
                        </select>
                      </div>
                    </div>

                    <div className="space-y-1.5">
                      <label className="font-semibold text-foreground flex items-center justify-between">
                        <span>District Name</span>
                        <span className="text-[10px] text-muted-foreground">Matches historical panel data</span>
                      </label>
                      <input
                        type="text"
                        value={phDistrict}
                        onChange={(e) => setPhDistrict(e.target.value)}
                        placeholder="e.g. Ludhiana, Gurdaspur, Patiala"
                        className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs font-medium focus:ring-2 focus:ring-primary focus:outline-none"
                      />
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div className="space-y-1.5">
                        <label className="font-semibold text-foreground">Cultivated Area ('000 ha)</label>
                        <input
                          type="number"
                          step="0.1"
                          min="0.1"
                          value={phArea}
                          onChange={(e) => setPhArea(e.target.value)}
                          className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs font-mono focus:ring-2 focus:ring-primary focus:outline-none"
                          required
                        />
                        <span className="text-[10px] text-muted-foreground">
                          = {formatNumber((parseFloat(phArea) || 0) * 1000, 0)} Hectares
                        </span>
                      </div>

                      <div className="space-y-1.5">
                        <label className="font-semibold text-foreground">Reported Production ('000 Tons)</label>
                        <input
                          type="number"
                          step="0.1"
                          min="0"
                          value={phProduction}
                          onChange={(e) => setPhProduction(e.target.value)}
                          className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs font-mono focus:ring-2 focus:ring-primary focus:outline-none"
                          required
                        />
                        <span className="text-[10px] text-muted-foreground">
                          = {formatNumber((parseFloat(phProduction) || 0) * 1000, 0)} Metric Tons
                        </span>
                      </div>
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
                      type="submit"
                      disabled={postHarvestMutation.isPending}
                      className="w-full gap-2 text-xs font-bold"
                    >
                      {postHarvestMutation.isPending ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          <span>Computing Pipeline Inference...</span>
                        </>
                      ) : (
                        <>
                          <ShieldCheck className="w-4 h-4" />
                          <span>Run Post-Harvest ML Prediction & Verification</span>
                        </>
                      )}
                    </Button>
                  </CardFooter>
                </form>
              </Card>
            </div>

            {/* Live Results Column (6 cols) */}
            <div className="lg:col-span-6 space-y-6">
              {postHarvestResult ? (
                <Card className="border-emerald-500/30 shadow-xl shadow-emerald-500/5">
                  <div className="bg-gradient-to-r from-emerald-500 via-primary to-sky-500 h-2" />
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-base font-bold">Verification & Comparison Results</CardTitle>
                      <Badge variant="success">Audited via ICRISAT</Badge>
                    </div>
                    <CardDescription className="text-xs">
                      {postHarvestResult.state} {postHarvestResult.matched_district ? `(${postHarvestResult.matched_district})` : ''} • Year {postHarvestResult.year}
                    </CardDescription>
                  </CardHeader>

                  <CardContent className="space-y-5 text-xs">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      <div className="p-4 rounded-xl border border-border bg-card space-y-1">
                        <div className="flex items-center justify-between text-[11px] text-muted-foreground font-semibold uppercase">
                          <span>ML Pipeline Prediction</span>
                          <Badge variant="outline" className="text-[10px]">Scikit-Learn</Badge>
                        </div>
                        <div className="text-2xl sm:text-3xl font-black text-foreground font-mono">
                          {formatYield(postHarvestResult.predicted_yield)}
                        </div>
                        <p className="text-[11px] text-muted-foreground font-mono">
                          RandomForest Regressor
                        </p>
                      </div>

                      <div className="p-4 rounded-xl border border-emerald-500/30 bg-emerald-500/5 space-y-1">
                        <div className="flex items-center justify-between text-[11px] text-emerald-700 dark:text-emerald-400 font-semibold uppercase">
                          <span>Exact Formula (P/A × 1000)</span>
                          <Badge variant="success" className="text-[10px]">Ground Truth</Badge>
                        </div>
                        <div className="text-2xl sm:text-3xl font-black text-emerald-600 dark:text-emerald-400 font-mono">
                          {formatYield(postHarvestResult.deterministic_yield)}
                        </div>
                        <p className="text-[11px] text-muted-foreground font-mono">
                          Algebraic identity
                        </p>
                      </div>
                    </div>

                    {/* Historical Match Box */}
                    {postHarvestResult.historical_matched && postHarvestResult.actual_yield !== null && (
                      <div className="p-3.5 rounded-xl border border-primary/20 bg-primary/5 space-y-2">
                        <div className="flex items-center justify-between font-semibold">
                          <span className="flex items-center gap-1.5 text-primary">
                            <Database className="w-3.5 h-3.5" />
                            <span>Matched Historical ICRISAT Record:</span>
                          </span>
                          <span className="font-mono text-sm font-bold text-foreground">
                            {formatYield(postHarvestResult.actual_yield || 0)}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-[11px] text-muted-foreground pt-1 border-t border-border/50">
                          <div>ML Error vs Actual: <span className="font-mono font-bold text-foreground">{formatNumber(postHarvestResult.ml_error ?? 0, 1)} kg/ha</span></div>
                          <div>Deterministic Error: <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400">{formatNumber(postHarvestResult.deterministic_error ?? 0, 1)} kg/ha</span></div>
                        </div>
                      </div>
                    )}

                    {/* Anomaly Badge if checked */}
                    {anomalyResult && (
                      <div className={`p-3 rounded-xl border flex items-center justify-between ${
                        anomalyResult.is_anomaly
                          ? 'border-amber-500/30 bg-amber-500/10 text-amber-900 dark:text-amber-200'
                          : 'border-emerald-500/20 bg-emerald-500/5 text-emerald-800 dark:text-emerald-300'
                      }`}>
                        <div className="flex items-center gap-2 font-bold text-xs">
                          {anomalyResult.is_anomaly ? <AlertTriangle className="w-4 h-4 text-amber-500" /> : <CheckCircle2 className="w-4 h-4 text-emerald-500" />}
                          <span>{anomalyResult.is_anomaly ? `Anomaly Detected: ${anomalyResult.severity}` : 'Normal Agricultural Record'}</span>
                        </div>
                        <span className="text-[11px] font-mono opacity-80">
                          Score: {anomalyResult.anomaly_score.toFixed(1)}/100
                        </span>
                      </div>
                    )}
                  </CardContent>
                </Card>
              ) : (
                <Card className="h-full flex items-center justify-center p-8 text-center text-muted-foreground text-xs">
                  <div className="space-y-3 max-w-sm">
                    <ShieldCheck className="w-10 h-10 mx-auto text-muted-foreground/40" />
                    <p className="font-semibold text-foreground">Ready for Post-Harvest Verification</p>
                    <p className="text-[11px]">Enter district observation values and click calculate to execute the production model pipeline.</p>
                  </div>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>

        {/* ========================================================================= */}
        {/* MODE B: PRE-SEASON ESTIMATION */}
        {/* ========================================================================= */}
        <TabsContent value="pre-season" className="space-y-6">
          <div className="p-4 rounded-xl border border-sky-500/30 bg-sky-500/10 text-xs text-foreground flex items-start justify-between gap-4">
            <div className="flex items-start gap-3">
              <CloudRain className="w-5 h-5 text-sky-600 dark:text-sky-400 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <p className="font-bold text-sky-800 dark:text-sky-300">
                  Mode B: Pre-Season Estimation (Leak-Free Operational Forecasting)
                </p>
                <p className="text-muted-foreground leading-relaxed text-[11px]">
                  Pre-season mode <strong>strictly excludes harvest production</strong>. Choose between the minimal baseline (Area + State + Year) or the advanced exogenous model leveraging pre-season land allocation and historical baseline lags.
                </p>
              </div>
            </div>

            <div className="flex items-center gap-1 bg-background/80 p-1 rounded-lg border border-sky-500/20 shrink-0">
              <button
                type="button"
                onClick={() => setPreSeasonSubMode('advanced')}
                className={`px-3 py-1 text-xs font-bold rounded-md transition-colors ${
                  preSeasonSubMode === 'advanced'
                    ? 'bg-sky-500 text-white shadow-sm'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                Advanced Exogenous ML
              </button>
              <button
                type="button"
                onClick={() => setPreSeasonSubMode('basic')}
                className={`px-3 py-1 text-xs font-bold rounded-md transition-colors ${
                  preSeasonSubMode === 'basic'
                    ? 'bg-sky-500 text-white shadow-sm'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                Basic Baseline
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Input Form Column (6 cols) */}
            <div className="lg:col-span-6 space-y-6">
              <Card>
                <CardHeader className="pb-4">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base font-bold">
                      {preSeasonSubMode === 'advanced' ? '1. Advanced Pre-Season Sowing Parameters' : '1. Basic Planting Inputs'}
                    </CardTitle>
                    <Badge variant="blue">
                      {preSeasonSubMode === 'advanced' ? 'Exogenous Land & Lags' : 'Year + State + Area'}
                    </Badge>
                  </div>
                  <CardDescription className="text-xs">
                    {preSeasonSubMode === 'advanced'
                      ? 'Features: [Year, State, Rice Area, Cropped Land, Rice Share, Historical Yield Lags]'
                      : 'Features: [Year, State Code, Cultivated Area]'}
                  </CardDescription>
                </CardHeader>

                <form onSubmit={handlePreSeasonSubmit}>
                  <CardContent className="space-y-4 text-xs">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div className="space-y-1.5">
                        <label className="font-semibold text-foreground">Sowing Year</label>
                        <select
                          value={psYear}
                          onChange={(e) => setPsYear(Number(e.target.value))}
                          className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs font-mono focus:ring-2 focus:ring-primary focus:outline-none"
                        >
                          {availableYears.map(y => (
                            <option key={y} value={y}>{y}</option>
                          ))}
                        </select>
                      </div>

                      <div className="space-y-1.5">
                        <label className="font-semibold text-foreground">State</label>
                        <select
                          value={psState}
                          onChange={(e) => setPsState(e.target.value)}
                          className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs focus:ring-2 focus:ring-primary focus:outline-none"
                        >
                          {availableStates.map(s => (
                            <option key={s} value={s}>{s}</option>
                          ))}
                        </select>
                      </div>
                    </div>

                    <div className="space-y-1.5">
                      <label className="font-semibold text-foreground flex items-center justify-between">
                        <span>District Name</span>
                        <span className="text-[10px] text-muted-foreground">Autocompletes regional agro-climatic defaults</span>
                      </label>
                      <input
                        type="text"
                        value={psDistrict}
                        onChange={(e) => setPsDistrict(e.target.value)}
                        placeholder="e.g. Ludhiana, Patiala"
                        className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs font-medium focus:ring-2 focus:ring-primary focus:outline-none"
                      />
                    </div>

                    <div className="space-y-1.5">
                      <label className="font-semibold text-foreground">Cultivated Rice Area ('000 ha)</label>
                      <input
                        type="number"
                        step="0.1"
                        min="0.1"
                        value={psArea}
                        onChange={(e) => setPsArea(e.target.value)}
                        className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs font-mono focus:ring-2 focus:ring-primary focus:outline-none"
                        required
                      />
                    </div>

                    {/* Advanced Pre-Season Land Allocation & Lags Accordion */}
                    {preSeasonSubMode === 'advanced' && (
                      <div className="space-y-3 pt-2 border-t border-border">
                        <button
                          type="button"
                          onClick={() => setShowAdvancedInputs(!showAdvancedInputs)}
                          className="flex items-center justify-between w-full text-left font-bold text-sky-600 dark:text-sky-400 text-xs hover:underline"
                        >
                          <span className="flex items-center gap-1.5">
                            <Sliders className="w-3.5 h-3.5" />
                            <span>Customize Land Allocation & Historical Lags (Optional)</span>
                          </span>
                          <span className="text-[10px] text-muted-foreground font-normal">
                            {showAdvancedInputs ? 'Hide' : 'Auto-filled from ICRISAT defaults'}
                          </span>
                        </button>

                        {showAdvancedInputs && (
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 p-3 rounded-xl bg-muted/30 border border-border/70 text-[11px] animate-in fade-in duration-200">
                            <div className="space-y-1">
                              <label className="font-semibold">Total Cropped Area ('000 ha)</label>
                              <input
                                type="number"
                                step="1"
                                value={psTotalArea}
                                onChange={(e) => setPsTotalArea(e.target.value)}
                                className="w-full rounded-lg border border-border bg-background px-2.5 py-1.5 text-xs font-mono"
                              />
                            </div>

                            <div className="space-y-1">
                              <label className="font-semibold">Wheat Cropped Area ('000 ha)</label>
                              <input
                                type="number"
                                step="1"
                                value={psWheatArea}
                                onChange={(e) => setPsWheatArea(e.target.value)}
                                className="w-full rounded-lg border border-border bg-background px-2.5 py-1.5 text-xs font-mono"
                              />
                            </div>

                            <div className="space-y-1">
                              <label className="font-semibold">Cotton Cropped Area ('000 ha)</label>
                              <input
                                type="number"
                                step="1"
                                value={psCottonArea}
                                onChange={(e) => setPsCottonArea(e.target.value)}
                                className="w-full rounded-lg border border-border bg-background px-2.5 py-1.5 text-xs font-mono"
                              />
                            </div>

                            <div className="space-y-1">
                              <label className="font-semibold">Sugarcane Area ('000 ha)</label>
                              <input
                                type="number"
                                step="1"
                                value={psSugarcaneArea}
                                onChange={(e) => setPsSugarcaneArea(e.target.value)}
                                className="w-full rounded-lg border border-border bg-background px-2.5 py-1.5 text-xs font-mono"
                              />
                            </div>

                            <div className="space-y-1 sm:col-span-2">
                              <label className="font-semibold">Historical Previous Year Yield (t-1 kg/ha)</label>
                              <input
                                type="number"
                                step="10"
                                value={psLagYield}
                                onChange={(e) => setPsLagYield(e.target.value)}
                                className="w-full rounded-lg border border-border bg-background px-2.5 py-1.5 text-xs font-mono"
                              />
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    <div className="p-3 rounded-lg border border-border/80 bg-muted/40 space-y-1 text-[11px] text-muted-foreground">
                      <div className="flex items-center gap-1.5 font-semibold text-foreground">
                        <Lock className="w-3.5 h-3.5 text-muted-foreground" />
                        <span>Production Input Strictly Locked</span>
                      </div>
                      <p>
                        Harvest production cannot be known before harvest and is strictly excluded from pre-season models.
                      </p>
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
                      type="submit"
                      disabled={preSeasonMutation.isPending || preSeasonAdvMutation.isPending}
                      className="w-full gap-2 text-xs font-bold"
                    >
                      {preSeasonMutation.isPending || preSeasonAdvMutation.isPending ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          <span>Computing Pre-Season Forecast...</span>
                        </>
                      ) : (
                        <>
                          <CloudRain className="w-4 h-4" />
                          <span>
                            {preSeasonSubMode === 'advanced'
                              ? 'Run Advanced Exogenous Pre-Season Forecast'
                              : 'Run Basic Pre-Season Baseline'}
                          </span>
                        </>
                      )}
                    </Button>
                  </CardFooter>
                </form>
              </Card>
            </div>

            {/* Results Column (6 cols) */}
            <div className="lg:col-span-6 space-y-6">
              {preSeasonSubMode === 'advanced' && preSeasonAdvResult ? (
                <Card className="border-sky-500/30 shadow-xl shadow-sky-500/5">
                  <div className="bg-gradient-to-r from-sky-500 via-primary to-indigo-500 h-2" />
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-base font-bold">Advanced Pre-Season Forecast Output</CardTitle>
                      <Badge variant="blue">Exogenous ML Model</Badge>
                    </div>
                    <CardDescription className="text-xs">
                      {preSeasonAdvResult.state} {preSeasonAdvResult.matched_district ? `(${preSeasonAdvResult.matched_district})` : ''} • Sowing Year {preSeasonAdvResult.year}
                    </CardDescription>
                  </CardHeader>

                  <CardContent className="space-y-5 text-xs">
                    {/* Hero Metric Big Number */}
                    <div className="p-5 rounded-2xl bg-gradient-to-br from-sky-500/10 via-primary/5 to-indigo-500/10 border border-sky-500/20 text-center space-y-1">
                      <span className="text-[11px] uppercase font-bold text-muted-foreground tracking-wider">
                        Predicted Pre-Season Yield
                      </span>
                      <div className="text-4xl font-black text-foreground font-mono">
                        {formatYield(preSeasonAdvResult.predicted_yield)}
                      </div>
                      <p className="text-xs font-semibold text-sky-600 dark:text-sky-400 font-mono">
                        {formatNumber((preSeasonAdvResult.predicted_yield / 100) * 0.404686, 2)} Quintals / Acre
                      </p>
                    </div>

                    {/* DAY 5: PREDICTION RISK INTELLIGENCE CARD */}
                    {riskResult && (
                      <div className="p-4 rounded-xl border border-sky-500/30 bg-sky-500/5 space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="font-bold text-foreground text-xs flex items-center gap-1.5">
                            <ShieldAlert className="w-4 h-4 text-sky-500" />
                            <span>Prediction Risk Assessment</span>
                          </span>
                          <Badge
                            variant={
                              riskResult.risk_level === 'LOW' ? 'success' :
                              riskResult.risk_level === 'MODERATE' ? 'blue' :
                              riskResult.risk_level === 'HIGH' ? 'warning' : 'destructive'
                            }
                          >
                            {riskResult.risk_level} RISK ({riskResult.risk_score.toFixed(1)}/100)
                          </Badge>
                        </div>

                        <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                          <div className="p-2 rounded bg-muted/40">
                            <span className="text-[10px] text-muted-foreground block font-sans">Prediction Interval (P10–P90)</span>
                            <span className="font-bold text-foreground">
                              {formatYield(preSeasonAdvResult.uncertainty.lower_bound_10th_pct)} – {formatYield(preSeasonAdvResult.uncertainty.upper_bound_90th_pct)}
                            </span>
                          </div>
                          <div className="p-2 rounded bg-muted/40">
                            <span className="text-[10px] text-muted-foreground block font-sans">Uncertainty Spread</span>
                            <span className="font-bold text-sky-600 dark:text-sky-400">
                              ±{formatNumber(riskResult.spread / 2, 1)} kg/ha ({riskResult.uncertainty_percent.toFixed(1)}%)
                            </span>
                          </div>
                        </div>

                        <div className="text-[11px] text-muted-foreground space-y-1">
                          <p className="font-semibold text-foreground font-sans">Key Risk Drivers:</p>
                          <ul className="list-disc list-inside space-y-0.5">
                            {riskResult.risk_factors.map((rf, idx) => (
                              <li key={idx}>{rf}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    )}

                    {/* DAY 5: EXPLAINABILITY - WHY THIS PREDICTION? */}
                    {explainResult ? (
                      <div className="space-y-3 p-3.5 rounded-xl border border-border bg-card">
                        <div className="flex items-center justify-between">
                          <span className="font-bold text-foreground text-xs uppercase tracking-wider">
                            Why this prediction? (Feature Attribution)
                          </span>
                          <span className="text-[10px] text-muted-foreground">Tree Feature Attribution</span>
                        </div>

                        <div className="space-y-2">
                          {explainResult.feature_contributions.slice(0, 5).map((fc, idx) => (
                            <div key={idx} className="space-y-1">
                              <div className="flex justify-between text-[11px]">
                                <span className="font-medium text-foreground">
                                  {fc.feature_name} <span className="text-muted-foreground font-mono">({fc.raw_value})</span>
                                </span>
                                <span className={`font-mono font-bold ${fc.direction === 'positive' ? 'text-emerald-500' : (fc.direction === 'negative' ? 'text-amber-500' : 'text-muted-foreground')}`}>
                                  {fc.direction === 'positive' ? '+' : (fc.direction === 'negative' ? '-' : '')}{fc.normalized_percentage}%
                                </span>
                              </div>
                              <div className="w-full bg-muted rounded-full h-1.5 overflow-hidden">
                                <div
                                  className={`h-1.5 rounded-full ${fc.direction === 'positive' ? 'bg-emerald-500' : (fc.direction === 'negative' ? 'bg-amber-500' : 'bg-sky-500')}`}
                                  style={{ width: `${fc.normalized_percentage * 1.5}%` }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>

                        <p className="text-[11px] text-muted-foreground italic leading-relaxed pt-1 border-t border-border/60">
                          {explainResult.summary}
                        </p>
                      </div>
                    ) : null}

                    {/* DAY 5: ANOMALY PANEL */}
                    {anomalyResult && (
                      <div className={`p-3.5 rounded-xl border ${
                        anomalyResult.is_anomaly
                          ? 'border-amber-500/30 bg-amber-500/10 text-amber-900 dark:text-amber-200'
                          : 'border-emerald-500/20 bg-emerald-500/5 text-emerald-800 dark:text-emerald-300'
                      }`}>
                        <div className="flex items-center justify-between font-bold text-xs">
                          <span className="flex items-center gap-1.5">
                            {anomalyResult.is_anomaly ? <AlertTriangle className="w-4 h-4 text-amber-500" /> : <CheckCircle2 className="w-4 h-4 text-emerald-500" />}
                            <span>{anomalyResult.is_anomaly ? `Anomaly Detected: ${anomalyResult.severity} SEVERITY` : 'Normal Agricultural Pattern'}</span>
                          </span>
                          <span className="font-mono text-[11px] opacity-80">
                            Score: {anomalyResult.anomaly_score.toFixed(1)}/100
                          </span>
                        </div>

                        {anomalyResult.reasons.length > 0 && (
                          <ul className="text-[11px] mt-2 space-y-0.5 list-disc list-inside opacity-90">
                            {anomalyResult.reasons.map((r, idx) => (
                              <li key={idx}>{r}</li>
                            ))}
                          </ul>
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>
              ) : (
                <Card className="h-full flex items-center justify-center p-8 text-center text-muted-foreground text-xs">
                  <div className="space-y-3 max-w-sm">
                    <CloudRain className="w-10 h-10 mx-auto text-muted-foreground/40" />
                    <p className="font-semibold text-foreground">Ready for Pre-Season Forecasting</p>
                    <p className="text-[11px]">Select sowing year, state, and cultivated area to generate an operational yield prediction with risk intelligence.</p>
                  </div>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>
      </Tabs>

      {/* ========================================================================= */}
      {/* CLIENT-SIDE PREDICTION HISTORY LOG */}
      {/* ========================================================================= */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <CardTitle className="text-base font-bold flex items-center gap-2">
                <History className="w-4 h-4 text-primary" />
                <span>Prediction Workspace History</span>
              </CardTitle>
              <CardDescription className="text-xs">
                Audit trail of recent predictions computed during this session (stored locally in browser)
              </CardDescription>
            </div>
            {history.length > 0 && (
              <Button
                variant="outline"
                size="sm"
                onClick={clearHistory}
                className="gap-1.5 text-xs text-muted-foreground hover:text-destructive"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>Clear History</span>
              </Button>
            )}
          </div>
        </CardHeader>

        <CardContent>
          {history.length === 0 ? (
            <div className="py-8 text-center text-xs text-muted-foreground">
              No predictions generated yet in this session. Run a calculation above to populate your workspace audit log.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-border bg-muted/40 text-muted-foreground">
                    <th className="py-2.5 px-3 font-semibold">Time</th>
                    <th className="py-2.5 px-3 font-semibold">Mode</th>
                    <th className="py-2.5 px-3 font-semibold">Location</th>
                    <th className="py-2.5 px-3 font-semibold">Year</th>
                    <th className="py-2.5 px-3 font-semibold">Area</th>
                    <th className="py-2.5 px-3 font-semibold">ML Prediction</th>
                    <th className="py-2.5 px-3 font-semibold">Risk Level</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/60 font-mono">
                  {history.map(item => (
                    <tr key={item.id} className="hover:bg-muted/20 transition-colors">
                      <td className="py-2 px-3 text-[11px] text-muted-foreground">{item.timestamp}</td>
                      <td className="py-2 px-3">
                        <Badge
                          variant={item.mode === 'post-harvest' ? 'success' : item.mode === 'pre-season-advanced' ? 'blue' : 'outline'}
                          className="text-[10px] font-sans"
                        >
                          {item.mode === 'post-harvest' ? 'Post-Harvest' : item.mode === 'pre-season-advanced' ? 'Pre-Season (Adv)' : 'Pre-Season (Basic)'}
                        </Badge>
                      </td>
                      <td className="py-2 px-3 font-sans font-medium text-foreground">
                        {item.state} {item.district ? `(${item.district})` : ''}
                      </td>
                      <td className="py-2 px-3">{item.year}</td>
                      <td className="py-2 px-3">{formatNumber(item.area || 0, 1)}k ha</td>
                      <td className="py-2 px-3 font-bold text-foreground">{formatYield(item.predictedYield)}</td>
                      <td className="py-2 px-3 font-sans">
                        {item.riskLevel ? (
                          <Badge
                            variant={
                              item.riskLevel === 'LOW' ? 'success' :
                              item.riskLevel === 'MODERATE' ? 'blue' :
                              item.riskLevel === 'HIGH' ? 'warning' : 'destructive'
                            }
                            className="text-[10px]"
                          >
                            {item.riskLevel}
                          </Badge>
                        ) : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
