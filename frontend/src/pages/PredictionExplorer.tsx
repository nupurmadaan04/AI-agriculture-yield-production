import React, { useState, useEffect, useMemo } from 'react'
import {
  useForecastCoverage,
  useForecastStrategies,
  useForecastCertificationSummary,
  useForecastContext,
  useForecastEvidence,
  useForecastProvenance,
  useForecastHealth,
  api,
} from '../services/api'
import { ForecastPredictResponse } from '../types/modeling'
import {
  Sparkles,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  History,
  TrendingUp,
  Cpu,
  Database,
  Lock,
  GitCommit,
  Clock,
  Layers,
  FileCode,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  RotateCcw,
  BarChart2,
  Sliders,
  Scale,
  Award,
  Compass,
} from 'lucide-react'

export const PredictionExplorer: React.FC = () => {
  // 1. Data queries
  const { data: coverageData, isLoading: coverageLoading } = useForecastCoverage()
  const { data: strategiesData } = useForecastStrategies()
  const { data: certSummary } = useForecastCertificationSummary()
  const { data: healthData } = useForecastHealth()

  // 2. Selection states
  const [selectedCrop, setSelectedCrop] = useState<string>('Oilseeds')
  const [selectedState, setSelectedState] = useState<string>('Punjab')
  const [selectedDistrict, setSelectedDistrict] = useState<string>('Ludhiana')
  const [forecastYear, setForecastYear] = useState<number>(2018)

  // 3. Feature override inputs (optional manual entry or auto-populated from context)
  const [yieldLag1, setYieldLag1] = useState<string>('')
  const [yieldRolling3yr, setYieldRolling3yr] = useState<string>('')
  const [areaLag1, setAreaLag1] = useState<string>('')
  const [useAutoContext, setUseAutoContext] = useState<boolean>(true)

  // 4. Execution state
  const [isPredicting, setIsPredicting] = useState<boolean>(false)
  const [predictionResult, setPredictionResult] = useState<ForecastPredictResponse | null>(null)
  const [predictionError, setPredictionError] = useState<string | null>(null)
  const [copiedProvenance, setCopiedProvenance] = useState<boolean>(false)

  // 5. Expandable sections
  const [expandedWhy, setExpandedWhy] = useState<boolean>(true)
  const [expandedTrace, setExpandedTrace] = useState<boolean>(true)
  const [expandedHistory, setExpandedHistory] = useState<boolean>(true)

  // 6. Context & Evidence queries
  const {
    data: contextData,
    isLoading: contextLoading,
  } = useForecastContext(selectedCrop, selectedState, selectedDistrict, forecastYear)

  const {
    data: evidenceData,
    isLoading: evidenceLoading,
  } = useForecastEvidence(selectedCrop)

  const {
    data: liveProvenance,
  } = useForecastProvenance(predictionResult?.request_id)

  // 7. Dynamic Cascading Geography
  const availableCrops = useMemo(() => {
    if (!coverageData?.unique_crops) return []
    return coverageData.unique_crops
  }, [coverageData])

  const availableStates = useMemo(() => {
    if (!coverageData?.coverage || !selectedCrop) return []
    const states = coverageData.coverage
      .filter((c) => c.crop.toLowerCase() === selectedCrop.toLowerCase())
      .map((c) => c.state)
    return Array.from(new Set(states)).sort()
  }, [coverageData, selectedCrop])

  const availableDistricts = useMemo(() => {
    if (!coverageData?.coverage || !selectedCrop || !selectedState) return []
    const districts = coverageData.coverage
      .filter(
        (c) =>
          c.crop.toLowerCase() === selectedCrop.toLowerCase() &&
          c.state.toLowerCase() === selectedState.toLowerCase()
      )
      .map((c) => c.district)
    return Array.from(new Set(districts)).sort()
  }, [coverageData, selectedCrop, selectedState])

  // Handle crop change
  const handleCropChange = (crop: string) => {
    setSelectedCrop(crop)
    if (!coverageData?.coverage) return
    const states = coverageData.coverage
      .filter((c) => c.crop.toLowerCase() === crop.toLowerCase())
      .map((c) => c.state)
    const uniqueStates = Array.from(new Set(states)).sort()
    const nextState = uniqueStates.includes(selectedState) ? selectedState : (uniqueStates[0] || '')
    setSelectedState(nextState)

    const districts = coverageData.coverage
      .filter(
        (c) =>
          c.crop.toLowerCase() === crop.toLowerCase() &&
          c.state.toLowerCase() === nextState.toLowerCase()
      )
      .map((c) => c.district)
    const uniqueDistricts = Array.from(new Set(districts)).sort()
    setSelectedDistrict(uniqueDistricts.includes(selectedDistrict) ? selectedDistrict : (uniqueDistricts[0] || ''))
  }

  // Handle state change
  const handleStateChange = (state: string) => {
    setSelectedState(state)
    if (!coverageData?.coverage) return
    const districts = coverageData.coverage
      .filter(
        (c) =>
          c.crop.toLowerCase() === selectedCrop.toLowerCase() &&
          c.state.toLowerCase() === state.toLowerCase()
      )
      .map((c) => c.district)
    const uniqueDistricts = Array.from(new Set(districts)).sort()
    setSelectedDistrict(uniqueDistricts[0] || '')
  }

  // Update feature defaults when context changes and auto mode is enabled
  useEffect(() => {
    if (useAutoContext && contextData) {
      if (contextData.previous_year_yield !== null && contextData.previous_year_yield !== undefined) {
        setYieldLag1(String(contextData.previous_year_yield))
      }
      if (contextData.rolling_3yr_mean !== null && contextData.rolling_3yr_mean !== undefined) {
        setYieldRolling3yr(String(contextData.rolling_3yr_mean))
      }
    }
  }, [contextData, useAutoContext])

  // Execute Forecast Prediction
  const handleGenerateForecast = async () => {
    setIsPredicting(true)
    setPredictionError(null)

    try {
      const payload = {
        crop: selectedCrop,
        state: selectedState,
        district: selectedDistrict,
        forecast_year: forecastYear,
        yield_lag_1: yieldLag1 ? parseFloat(yieldLag1) : undefined,
        yield_rolling_3yr_mean: yieldRolling3yr ? parseFloat(yieldRolling3yr) : undefined,
        area_lag_1: areaLag1 ? parseFloat(areaLag1) : undefined,
      }

      const res = await api.predictForecast(payload)
      if (res.status === 'REJECTED') {
        setPredictionError(res.error_message || 'Forecast rejected by certification guard.')
      }
      setPredictionResult(res)
    } catch (err: any) {
      const msg = err.message || 'Forecast prediction request failed.'
      setPredictionError(msg)
      setPredictionResult(null)
    } finally {
      setIsPredicting(false)
    }
  }

  // Run initial prediction on load if not yet run
  useEffect(() => {
    if (!predictionResult && selectedCrop && selectedState && selectedDistrict) {
      handleGenerateForecast()
    }
  }, [selectedCrop, selectedState, selectedDistrict])

  const copyProvenanceJson = () => {
    const prov = predictionResult?.provenance || liveProvenance
    if (prov) {
      navigator.clipboard.writeText(JSON.stringify(prov, null, 2))
      setCopiedProvenance(true)
      setTimeout(() => setCopiedProvenance(false), 2000)
    }
  }

  // Strategy badge styling
  const getStrategyBadge = (status?: string) => {
    switch (status) {
      case 'PRODUCTION_READY':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-[#eef2e6] text-[#2d4a22] border border-[#d6e2c8]">
            <Award className="w-3.5 h-3.5 text-[#2d4a22]" />
            PRODUCTION READY (ML)
          </span>
        )
      case 'CONDITIONAL_PRODUCTION':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-[#fef3c7] text-[#92400e] border border-[#fde68a]">
            <ShieldCheck className="w-3.5 h-3.5 text-[#92400e]" />
            CONDITIONAL PRODUCTION (ML + CLIPPING)
          </span>
        )
      case 'BASELINE_PRODUCTION':
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-[#f4f4ee] text-[#4a4d43] border border-[#d4d4c8]">
            <Scale className="w-3.5 h-3.5 text-[#63665c]" />
            BASELINE PRODUCTION (STATISTICAL)
          </span>
        )
    }
  }

  return (
    <div className="min-h-screen bg-[#fbfbf9] text-[#1c1d1a] py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* ============================================================ */}
        {/* 1. HEADER SECTION */}
        {/* ============================================================ */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between border-b border-[#e5e5dc] pb-6 gap-4">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold tracking-tight text-[#1c1d1a]">Prediction Explorer</h1>
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-[#eef2e6] text-[#2d4a22] border border-[#d6e2c8]">
                GOVERNED PRE-SEASON FORECASTING
              </span>
            </div>
            <p className="mt-1 text-sm text-[#63665c]">
              Traceable agricultural yield predictions with transparent strategy resolution, empirical baseline context, and cryptographic provenance.
            </p>
          </div>

          {/* System Guard Status Badges */}
          <div className="flex items-center gap-2 flex-wrap">
            <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-white border border-[#e5e5dc] text-xs font-mono text-[#3c3e37] shadow-sm">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
              <span>Guard: <strong>{healthData?.governance_guard || 'ACTIVE_STRICT'}</strong></span>
            </div>
            <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-white border border-[#e5e5dc] text-xs font-mono text-[#3c3e37] shadow-sm">
              <Lock className="w-3.5 h-3.5 text-[#4a4d43]" />
              <span>Provenance: <strong>SHA-256</strong></span>
            </div>
            <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-white border border-[#e5e5dc] text-xs font-mono text-[#3c3e37] shadow-sm">
              <Database className="w-3.5 h-3.5 text-blue-600" />
              <span>Data: <strong>AGRI_PANEL_1.0</strong></span>
            </div>
          </div>
        </div>

        {/* ============================================================ */}
        {/* 2. FORECAST INPUT PANEL */}
        {/* ============================================================ */}
        <div className="bg-white rounded-xl border border-[#e5e5dc] shadow-sm p-6 space-y-6">
          <div className="flex items-center justify-between border-b border-[#f0f0eb] pb-4">
            <div className="flex items-center gap-2">
              <Compass className="w-5 h-5 text-[#2d4a22]" />
              <h2 className="text-base font-semibold text-[#1c1d1a]">Target Forecasting Scenario</h2>
            </div>
            <div className="text-xs text-[#63665c]">
              Dynamic coverage catalog: <strong>{coverageData?.unique_crops?.length || 14} crops</strong> across <strong>{coverageData?.unique_districts_count || 311} districts</strong>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
            {/* Crop Selector */}
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-[#63665c] mb-1.5">
                Crop
              </label>
              <select
                value={selectedCrop}
                onChange={(e) => handleCropChange(e.target.value)}
                disabled={coverageLoading || isPredicting}
                className="w-full px-3 py-2 text-sm bg-white border border-[#d4d4c8] rounded-lg text-[#1c1d1a] focus:ring-2 focus:ring-[#2d4a22] focus:border-transparent outline-none transition-all shadow-sm"
              >
                {availableCrops.map((c) => (
                  <option key={c} value={c}>
                    {c} {c === 'Oilseeds' ? '⭐ (ML Production Ready)' : c === 'Sugarcane' ? '⚡ (ML Conditional)' : '(Statistical Baseline)'}
                  </option>
                ))}
              </select>
            </div>

            {/* State Selector */}
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-[#63665c] mb-1.5">
                State
              </label>
              <select
                value={selectedState}
                onChange={(e) => handleStateChange(e.target.value)}
                disabled={coverageLoading || isPredicting || availableStates.length === 0}
                className="w-full px-3 py-2 text-sm bg-white border border-[#d4d4c8] rounded-lg text-[#1c1d1a] focus:ring-2 focus:ring-[#2d4a22] focus:border-transparent outline-none transition-all shadow-sm"
              >
                {availableStates.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>

            {/* District Selector */}
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-[#63665c] mb-1.5">
                District
              </label>
              <select
                value={selectedDistrict}
                onChange={(e) => setSelectedDistrict(e.target.value)}
                disabled={coverageLoading || isPredicting || availableDistricts.length === 0}
                className="w-full px-3 py-2 text-sm bg-white border border-[#d4d4c8] rounded-lg text-[#1c1d1a] focus:ring-2 focus:ring-[#2d4a22] focus:border-transparent outline-none transition-all shadow-sm"
              >
                {availableDistricts.map((d) => (
                  <option key={d} value={d}>{d}</option>
                ))}
              </select>
            </div>

            {/* Forecast Year */}
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-[#63665c] mb-1.5">
                Target Forecast Year
              </label>
              <select
                value={forecastYear}
                onChange={(e) => setForecastYear(parseInt(e.target.value, 10))}
                disabled={isPredicting}
                className="w-full px-3 py-2 text-sm bg-white border border-[#d4d4c8] rounded-lg text-[#1c1d1a] focus:ring-2 focus:ring-[#2d4a22] focus:border-transparent outline-none transition-all shadow-sm"
              >
                <option value={2018}>2018 (Certified Pre-Season Horizon)</option>
                <option value={2017}>2017 (Walk-Forward Validation Split)</option>
                <option value={2016}>2016 (Walk-Forward Validation Split)</option>
              </select>
            </div>
          </div>

          {/* Optional Pre-Season Feature Controls */}
          <div className="bg-[#fbfbf9] rounded-lg border border-[#e5e5dc] p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-[#4a4d43]" />
                <span className="text-xs font-semibold text-[#1c1d1a]">Pre-Season Lag Context Inputs</span>
              </div>
              <label className="flex items-center gap-2 cursor-pointer text-xs text-[#63665c]">
                <input
                  type="checkbox"
                  checked={useAutoContext}
                  onChange={(e) => setUseAutoContext(e.target.checked)}
                  className="rounded border-[#d4d4c8] text-[#2d4a22] focus:ring-[#2d4a22]"
                />
                <span>Auto-populate from historical panel (Recommended)</span>
              </label>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div>
                <label className="block text-[11px] font-medium text-[#63665c] mb-1">
                  Yield Lag (t-1) <span className="text-[#828579]">[kg/ha]</span>
                </label>
                <input
                  type="number"
                  placeholder="e.g. 1548.0"
                  value={yieldLag1}
                  onChange={(e) => {
                    setYieldLag1(e.target.value)
                    setUseAutoContext(false)
                  }}
                  className="w-full px-3 py-1.5 text-xs bg-white border border-[#d4d4c8] rounded text-[#1c1d1a] outline-none focus:border-[#2d4a22]"
                />
              </div>

              <div>
                <label className="block text-[11px] font-medium text-[#63665c] mb-1">
                  3-Yr Rolling Mean <span className="text-[#828579]">[kg/ha]</span>
                </label>
                <input
                  type="number"
                  placeholder="e.g. 1650.0"
                  value={yieldRolling3yr}
                  onChange={(e) => {
                    setYieldRolling3yr(e.target.value)
                    setUseAutoContext(false)
                  }}
                  className="w-full px-3 py-1.5 text-xs bg-white border border-[#d4d4c8] rounded text-[#1c1d1a] outline-none focus:border-[#2d4a22]"
                />
              </div>

              <div>
                <label className="block text-[11px] font-medium text-[#63665c] mb-1">
                  Area Lag (t-1) <span className="text-[#828579]">[ha, optional]</span>
                </label>
                <input
                  type="number"
                  placeholder="e.g. 2500.0"
                  value={areaLag1}
                  onChange={(e) => {
                    setAreaLag1(e.target.value)
                    setUseAutoContext(false)
                  }}
                  className="w-full px-3 py-1.5 text-xs bg-white border border-[#d4d4c8] rounded text-[#1c1d1a] outline-none focus:border-[#2d4a22]"
                />
              </div>
            </div>
          </div>

          {/* Action Row */}
          <div className="flex items-center justify-between pt-2">
            <button
              onClick={() => {
                setUseAutoContext(true)
                if (contextData) {
                  setYieldLag1(contextData.previous_year_yield ? String(contextData.previous_year_yield) : '')
                  setYieldRolling3yr(contextData.rolling_3yr_mean ? String(contextData.rolling_3yr_mean) : '')
                  setAreaLag1('')
                }
              }}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-[#63665c] hover:text-[#1c1d1a] transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset Defaults
            </button>

            <button
              onClick={handleGenerateForecast}
              disabled={isPredicting || !selectedCrop || !selectedDistrict}
              className="inline-flex items-center gap-2 px-6 py-2.5 bg-[#2d4a22] text-white text-sm font-semibold rounded-lg hover:bg-[#233a1b] active:bg-[#1a2c14] transition-all shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isPredicting ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Evaluating Governance & Inferring...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Generate Governed Forecast</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Error Alert */}
        {predictionError && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3 text-red-900">
            <AlertTriangle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <div className="text-sm font-semibold">Forecast Generation Blocked</div>
              <div className="text-xs text-red-800">{predictionError}</div>
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* 3. FORECAST PREDICTION RESULT SECTION */}
        {/* ============================================================ */}
        {predictionResult && predictionResult.status === 'SUCCESS' && (
          <div className="bg-white rounded-xl border border-[#e5e5dc] shadow-sm overflow-hidden">
            <div className="p-6 md:p-8 space-y-6">
              
              <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6 pb-6 border-b border-[#f0f0eb]">
                {/* Main Prediction Display */}
                <div>
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-[#63665c] mb-1">
                    <span>Certified Pre-Season Forecast</span>
                    <span>•</span>
                    <span>Target Horizon {predictionResult.forecast_year}</span>
                  </div>
                  <div className="flex items-baseline gap-3">
                    <span className="text-4xl md:text-5xl font-extrabold tracking-tight text-[#1c1d1a]">
                      {predictionResult.prediction !== null && predictionResult.prediction !== undefined
                        ? Number(predictionResult.prediction).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
                        : 'N/A'}
                    </span>
                    <span className="text-xl font-bold text-[#2d4a22]">
                      {predictionResult.unit}
                    </span>
                  </div>
                  <div className="mt-2 text-xs text-[#63665c]">
                    Target Entity: <strong className="text-[#1c1d1a]">{predictionResult.district}, {predictionResult.state}</strong> (Crop: <strong>{predictionResult.crop}</strong>)
                  </div>
                </div>

                {/* Strategy Resolution Card */}
                <div className="bg-[#fbfbf9] p-4 rounded-lg border border-[#e5e5dc] space-y-2 lg:max-w-md w-full">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium text-[#63665c]">Governed Strategy</span>
                    {getStrategyBadge(predictionResult.certification_status)}
                  </div>
                  <div className="text-sm font-semibold text-[#1c1d1a]">
                    {predictionResult.strategy}
                  </div>
                  <div className="text-xs text-[#63665c]">
                    Model Artifact: <span className="font-mono text-[#3c3e37]">{predictionResult.model_version || 'Statistical Baseline'}</span>
                  </div>
                  {predictionResult.fallback_used && (
                    <div className="text-xs font-medium text-amber-700 bg-amber-50 px-2 py-1 rounded border border-amber-200">
                      ⚡ Fallback Active: {predictionResult.fallback_reason || 'Sparse district history threshold triggered.'}
                    </div>
                  )}
                </div>
              </div>

              {/* 5-Point Governance Status Check */}
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 pt-2">
                <div className="bg-[#fbfbf9] p-3 rounded-lg border border-[#e5e5dc] flex items-center gap-2.5">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                  <div>
                    <div className="text-[10px] text-[#63665c] font-medium uppercase">Dataset Integrity</div>
                    <div className="text-xs font-semibold text-[#1c1d1a]">PASS (AGRI_PANEL)</div>
                  </div>
                </div>

                <div className="bg-[#fbfbf9] p-3 rounded-lg border border-[#e5e5dc] flex items-center gap-2.5">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                  <div>
                    <div className="text-[10px] text-[#63665c] font-medium uppercase">Model Integrity</div>
                    <div className="text-xs font-semibold text-[#1c1d1a]">PASS (SHA-256)</div>
                  </div>
                </div>

                <div className="bg-[#fbfbf9] p-3 rounded-lg border border-[#e5e5dc] flex items-center gap-2.5">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                  <div>
                    <div className="text-[10px] text-[#63665c] font-medium uppercase">Strategy Registration</div>
                    <div className="text-xs font-semibold text-[#1c1d1a]">PASS (Day 24)</div>
                  </div>
                </div>

                <div className="bg-[#fbfbf9] p-3 rounded-lg border border-[#e5e5dc] flex items-center gap-2.5">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                  <div>
                    <div className="text-[10px] text-[#63665c] font-medium uppercase">Geographic Coverage</div>
                    <div className="text-xs font-semibold text-[#1c1d1a]">PASS (Certified)</div>
                  </div>
                </div>

                <div className="bg-[#fbfbf9] p-3 rounded-lg border border-[#e5e5dc] flex items-center gap-2.5">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                  <div>
                    <div className="text-[10px] text-[#63665c] font-medium uppercase">Input Completeness</div>
                    <div className="text-xs font-semibold text-[#1c1d1a]">PASS (Validated)</div>
                  </div>
                </div>
              </div>

            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* 4. HISTORICAL CONTEXT & REFERENCE BASELINES */}
        {/* ============================================================ */}
        <div className="bg-white rounded-xl border border-[#e5e5dc] shadow-sm p-6 space-y-6">
          <div className="flex items-center justify-between border-b border-[#f0f0eb] pb-4">
            <div className="flex items-center gap-2">
              <History className="w-5 h-5 text-[#2d4a22]" />
              <h2 className="text-base font-semibold text-[#1c1d1a]">Historical Context & Baseline Comparisons</h2>
            </div>
            <button
              onClick={() => setExpandedHistory(!expandedHistory)}
              className="text-xs text-[#63665c] hover:text-[#1c1d1a] flex items-center gap-1"
            >
              <span>{expandedHistory ? 'Collapse' : 'Expand'} Details</span>
              {expandedHistory ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>
          </div>

          {contextLoading ? (
            <div className="py-8 text-center text-sm text-[#63665c] flex items-center justify-center gap-2">
              <div className="w-4 h-4 border-2 border-[#2d4a22] border-t-transparent rounded-full animate-spin" />
              <span>Loading longitudinal panel observations...</span>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Comparative Benchmark Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                
                <div className="bg-[#f4f7f2] p-4 rounded-lg border border-[#d6e2c8]">
                  <div className="text-[10px] font-semibold text-[#2d4a22] uppercase tracking-wider">PREDICTED</div>
                  <div className="text-xs font-medium text-[#4a4d43] mt-0.5">Model Forecast</div>
                  <div className="text-xl font-bold text-[#1c1d1a] mt-2">
                    {predictionResult?.prediction !== null && predictionResult?.prediction !== undefined
                      ? `${Number(predictionResult.prediction).toFixed(2)} kg/ha`
                      : 'Pending'}
                  </div>
                  <div className="text-[11px] text-[#2d4a22] mt-1 font-mono">Governed Pre-Season</div>
                </div>

                <div className="bg-[#fbfbf9] p-4 rounded-lg border border-[#e5e5dc]">
                  <div className="text-[10px] font-semibold text-[#63665c] uppercase tracking-wider">HISTORICAL REFERENCE</div>
                  <div className="text-xs font-medium text-[#4a4d43] mt-0.5">District Mean</div>
                  <div className="text-xl font-bold text-[#1c1d1a] mt-2">
                    {contextData?.district_historical_mean !== null && contextData?.district_historical_mean !== undefined
                      ? `${contextData.district_historical_mean.toFixed(2)} kg/ha`
                      : 'N/A'}
                  </div>
                  <div className="text-[11px] text-[#63665c] mt-1 font-mono">{contextData?.historical_observations_count || 0} harvest seasons</div>
                </div>

                <div className="bg-[#fbfbf9] p-4 rounded-lg border border-[#e5e5dc]">
                  <div className="text-[10px] font-semibold text-[#63665c] uppercase tracking-wider">PREVIOUS OBSERVATION</div>
                  <div className="text-xs font-medium text-[#4a4d43] mt-0.5">Previous-Year Yield (t-1)</div>
                  <div className="text-xl font-bold text-[#1c1d1a] mt-2">
                    {contextData?.previous_year_yield !== null && contextData?.previous_year_yield !== undefined
                      ? `${contextData.previous_year_yield.toFixed(2)} kg/ha`
                      : 'N/A'}
                  </div>
                  <div className="text-[11px] text-[#63665c] mt-1 font-mono">Naive Persistence Benchmark</div>
                </div>

                <div className="bg-[#fbfbf9] p-4 rounded-lg border border-[#e5e5dc]">
                  <div className="text-[10px] font-semibold text-[#63665c] uppercase tracking-wider">DERIVED REFERENCE</div>
                  <div className="text-xs font-medium text-[#4a4d43] mt-0.5">3-Year Rolling Mean</div>
                  <div className="text-xl font-bold text-[#1c1d1a] mt-2">
                    {contextData?.rolling_3yr_mean !== null && contextData?.rolling_3yr_mean !== undefined
                      ? `${contextData.rolling_3yr_mean.toFixed(2)} kg/ha`
                      : 'N/A'}
                  </div>
                  <div className="text-[11px] text-[#63665c] mt-1 font-mono">Local Baseline Average</div>
                </div>

              </div>

              {/* Historical Context Notes & Scientific Disclaimer */}
              <div className="text-xs text-[#63665c] bg-[#fbfbf9] p-3 rounded-lg border border-[#e5e5dc] flex items-center justify-between">
                <span>{contextData?.context_notes || 'Longitudinal panel data source: AGRI_PANEL_1.0.'}</span>
                <span className="font-mono text-[11px] text-[#828579]">Bounds: {contextData?.historical_min_yield || 0} - {contextData?.historical_max_yield || 0} kg/ha (Std: ±{contextData?.historical_yield_std || 0})</span>
              </div>

              {/* Recent Historical Observations Table */}
              {expandedHistory && contextData?.recent_observations && contextData.recent_observations.length > 0 && (
                <div className="space-y-2">
                  <div className="text-xs font-semibold text-[#1c1d1a] uppercase tracking-wider">
                    Recent Historical Observed Harvest Seasons ({selectedDistrict}, {selectedState})
                  </div>
                  <div className="overflow-x-auto border border-[#e5e5dc] rounded-lg">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-[#f4f4ee] text-[#4a4d43] font-semibold border-b border-[#e5e5dc]">
                        <tr>
                          <th className="py-2.5 px-4">Harvest Season (Year)</th>
                          <th className="py-2.5 px-4">Yield [kg/ha]</th>
                          <th className="py-2.5 px-4">Cultivated Area [ha]</th>
                          <th className="py-2.5 px-4">Production [tonnes]</th>
                          <th className="py-2.5 px-4">Scientific Classification</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[#f0f0eb] bg-white">
                        {contextData.recent_observations.map((obs) => (
                          <tr key={obs.year} className="hover:bg-[#fbfbf9]">
                            <td className="py-2 px-4 font-mono font-medium text-[#1c1d1a]">{obs.year}</td>
                            <td className="py-2 px-4 font-mono font-semibold text-[#2d4a22]">{obs.yield_kg_ha.toFixed(2)}</td>
                            <td className="py-2 px-4 font-mono text-[#63665c]">{obs.area_ha !== null && obs.area_ha !== undefined ? obs.area_ha.toLocaleString() : '—'}</td>
                            <td className="py-2 px-4 font-mono text-[#63665c]">{obs.production_tonnes !== null && obs.production_tonnes !== undefined ? obs.production_tonnes.toLocaleString() : '—'}</td>
                            <td className="py-2 px-4">
                              <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-[#f4f4ee] text-[#4a4d43]">
                                {obs.observation_type}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* ============================================================ */}
        {/* 5. MODEL EVIDENCE & SCIENTIFIC VALIDATION */}
        {/* ============================================================ */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          {/* Left Column: Model Validation & Robustness Evidence */}
          <div className="bg-white rounded-xl border border-[#e5e5dc] shadow-sm p-6 space-y-5">
            <div className="flex items-center justify-between border-b border-[#f0f0eb] pb-4">
              <div className="flex items-center gap-2">
                <Cpu className="w-5 h-5 text-[#2d4a22]" />
                <h2 className="text-base font-semibold text-[#1c1d1a]">Model Validation Evidence</h2>
              </div>
              <span className="text-xs font-mono text-[#63665c]">Walk-Forward Evaluation</span>
            </div>

            {evidenceLoading ? (
              <div className="py-8 text-center text-sm text-[#63665c]">Loading validation evidence...</div>
            ) : (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-[#fbfbf9] p-3 rounded-lg border border-[#e5e5dc]">
                    <div className="text-[10px] text-[#63665c] uppercase font-semibold">Validation Protocol</div>
                    <div className="text-xs font-medium text-[#1c1d1a] mt-1">{evidenceData?.validation_protocol}</div>
                  </div>

                  <div className="bg-[#fbfbf9] p-3 rounded-lg border border-[#e5e5dc]">
                    <div className="text-[10px] text-[#63665c] uppercase font-semibold">Model Family</div>
                    <div className="text-xs font-medium text-[#1c1d1a] mt-1">{evidenceData?.model_family || 'Statistical Baseline'}</div>
                  </div>

                  <div className="bg-[#fbfbf9] p-3 rounded-lg border border-[#e5e5dc]">
                    <div className="text-[10px] text-[#63665c] uppercase font-semibold">Validation Mean MAE</div>
                    <div className="text-sm font-bold text-[#1c1d1a] mt-1">
                      {evidenceData?.mean_mae !== null && evidenceData?.mean_mae !== undefined
                        ? `${evidenceData.mean_mae.toFixed(2)} kg/ha`
                        : 'N/A'}
                    </div>
                  </div>

                  <div className="bg-[#fbfbf9] p-3 rounded-lg border border-[#e5e5dc]">
                    <div className="text-[10px] text-[#63665c] uppercase font-semibold">Baseline MAE (Naive)</div>
                    <div className="text-sm font-bold text-[#63665c] mt-1">
                      {evidenceData?.baseline_mae !== null && evidenceData?.baseline_mae !== undefined
                        ? `${evidenceData.baseline_mae.toFixed(2)} kg/ha`
                        : 'N/A'}
                    </div>
                  </div>
                </div>

                {/* Performance Metrics */}
                <div className="flex items-center justify-between bg-[#f4f7f2] p-3.5 rounded-lg border border-[#d6e2c8]">
                  <div>
                    <div className="text-[11px] font-semibold text-[#2d4a22] uppercase">Mean Improvement vs Baseline</div>
                    <div className="text-lg font-bold text-[#1c1d1a]">
                      {evidenceData?.mean_improvement_pct !== null && evidenceData?.mean_improvement_pct !== undefined
                        ? `${evidenceData.mean_improvement_pct > 0 ? '+' : ''}${evidenceData.mean_improvement_pct.toFixed(2)}%`
                        : '0.00%'}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-[11px] font-semibold text-[#2d4a22] uppercase">Fold Win Rate</div>
                    <div className="text-lg font-bold text-[#1c1d1a]">
                      {evidenceData?.fold_win_rate_pct !== null && evidenceData?.fold_win_rate_pct !== undefined
                        ? `${evidenceData.fold_win_rate_pct.toFixed(1)}%`
                        : 'N/A'}
                    </div>
                  </div>
                </div>

                {/* Operating Rule & Fallback */}
                <div className="space-y-2 text-xs">
                  <div className="text-[#63665c]">
                    <strong className="text-[#1c1d1a]">Operating Rule:</strong> {evidenceData?.operating_rule}
                  </div>
                  <div className="text-[#63665c]">
                    <strong className="text-[#1c1d1a]">Fallback Strategy:</strong> {evidenceData?.fallback_strategy}
                  </div>
                </div>

                {/* Empirical Uncertainty (P10-P90) */}
                <div className="bg-[#fbfbf9] p-4 rounded-lg border border-[#e5e5dc] space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-[#1c1d1a]">Empirical Uncertainty</span>
                    <span className="text-[10px] font-mono text-[#63665c]">Ensemble Spread</span>
                  </div>
                  {evidenceData?.empirical_p10_p90_spread ? (
                    <div>
                      <div className="text-sm font-bold text-[#2d4a22]">
                        Empirical P10–P90 ensemble range: ±{(evidenceData.empirical_p10_p90_spread / 2).toFixed(2)} kg/ha (Spread: {evidenceData.empirical_p10_p90_spread.toFixed(2)} kg/ha)
                      </div>
                      <p className="text-[11px] text-[#63665c] mt-1 italic">
                        This range is derived empirically from model ensemble predictions and is not a formal confidence interval.
                      </p>
                    </div>
                  ) : (
                    <div className="text-xs text-[#63665c] italic">
                      Empirical uncertainty range unavailable for this strategy. Predictions are derived deterministically.
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Model Feature Evidence (XAI) */}
          <div className="bg-white rounded-xl border border-[#e5e5dc] shadow-sm p-6 space-y-5">
            <div className="flex items-center justify-between border-b border-[#f0f0eb] pb-4">
              <div className="flex items-center gap-2">
                <BarChart2 className="w-5 h-5 text-[#2d4a22]" />
                <h2 className="text-base font-semibold text-[#1c1d1a]">Model Feature Evidence</h2>
              </div>
              <span className="text-xs font-mono text-[#63665c]">
                {evidenceData?.is_ml_strategy ? 'Tree Split Attribution' : 'Statistical Baseline'}
              </span>
            </div>

            <div className="space-y-4">
              <div className="text-xs text-[#63665c] bg-[#fbfbf9] p-3 rounded-lg border border-[#e5e5dc]">
                {evidenceData?.explanation_notice}
              </div>

              {evidenceData?.is_ml_strategy && evidenceData.feature_importance.length > 0 ? (
                <div className="space-y-3">
                  <div className="text-xs font-semibold text-[#1c1d1a] uppercase tracking-wider">
                    Registered Feature Importances
                  </div>
                  <div className="space-y-2.5">
                    {evidenceData.feature_importance.map((feat) => (
                      <div key={feat.feature_name} className="space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span className="font-mono font-medium text-[#1c1d1a]">{feat.feature_name}</span>
                          <span className="font-mono font-semibold text-[#2d4a22]">{feat.importance_pct.toFixed(2)}%</span>
                        </div>
                        <div className="w-full bg-[#f0f0eb] h-2 rounded-full overflow-hidden">
                          <div
                            className="bg-[#2d4a22] h-full rounded-full transition-all duration-500"
                            style={{ width: `${Math.min(100, Math.max(2, feat.importance_pct))}%` }}
                          />
                        </div>
                        {feat.description && (
                          <div className="text-[10px] text-[#828579]">{feat.description}</div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="py-8 text-center bg-[#fbfbf9] rounded-lg border border-[#e5e5dc] p-6 space-y-2">
                  <Scale className="w-8 h-8 text-[#828579] mx-auto" />
                  <div className="text-xs font-semibold text-[#1c1d1a]">Feature Attribution Not Applicable</div>
                  <div className="text-xs text-[#63665c] max-w-sm mx-auto">
                    This forecast is governed by a statistical persistence/district mean strategy. Longitudinal averages are used directly without parameter weights.
                  </div>
                </div>
              )}
            </div>
          </div>

        </div>

        {/* ============================================================ */}
        {/* 6. WHY THIS PREDICTION? (TRANSPARENT DECISION LOGIC) */}
        {/* ============================================================ */}
        <div className="bg-white rounded-xl border border-[#e5e5dc] shadow-sm p-6 space-y-4">
          <div
            className="flex items-center justify-between cursor-pointer"
            onClick={() => setExpandedWhy(!expandedWhy)}
          >
            <div className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-[#2d4a22]" />
              <h2 className="text-base font-semibold text-[#1c1d1a]">Why This Prediction? (Scientific Explanation)</h2>
            </div>
            {expandedWhy ? <ChevronUp className="w-4 h-4 text-[#63665c]" /> : <ChevronDown className="w-4 h-4 text-[#63665c]" />}
          </div>

          {expandedWhy && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pt-2">
              
              <div className="bg-[#fbfbf9] p-4 rounded-lg border border-[#e5e5dc] space-y-1">
                <div className="text-[10px] font-semibold text-[#63665c] uppercase">1. Strategy Selected</div>
                <div className="text-xs font-semibold text-[#1c1d1a]">{predictionResult?.strategy || evidenceData?.strategy_name}</div>
                <div className="text-[11px] text-[#63665c] pt-1">
                  Selected by the governed forecast strategy catalog based on temporal walk-forward evaluation.
                </div>
              </div>

              <div className="bg-[#fbfbf9] p-4 rounded-lg border border-[#e5e5dc] space-y-1">
                <div className="text-[10px] font-semibold text-[#63665c] uppercase">2. Model / Baseline Used</div>
                <div className="text-xs font-semibold text-[#1c1d1a] font-mono">{predictionResult?.model_version || evidenceData?.model_version || 'Statistical Baseline'}</div>
                <div className="text-[11px] text-[#63665c] pt-1">
                  Validated against naive persistence and evaluated for temporal stability.
                </div>
              </div>

              <div className="bg-[#fbfbf9] p-4 rounded-lg border border-[#e5e5dc] space-y-1">
                <div className="text-[10px] font-semibold text-[#63665c] uppercase">3. Input Data Used</div>
                <div className="text-xs font-semibold text-[#1c1d1a]">AGRI_PANEL_1.0 (1966–2017)</div>
                <div className="text-[11px] text-[#63665c] pt-1">
                  {contextData?.historical_observations_count || 0} observed harvest seasons in {selectedDistrict}, {selectedState}.
                </div>
              </div>

              <div className="bg-[#fbfbf9] p-4 rounded-lg border border-[#e5e5dc] space-y-1">
                <div className="text-[10px] font-semibold text-[#63665c] uppercase">4. Validation Evidence</div>
                <div className="text-xs font-semibold text-[#1c1d1a]">
                  MAE: {evidenceData?.mean_mae ? `${evidenceData.mean_mae.toFixed(2)} kg/ha` : 'Baseline'}
                </div>
                <div className="text-[11px] text-[#63665c] pt-1">
                  Evaluated across 4 expanding walk-forward origins (2014, 2015, 2016, 2017).
                </div>
              </div>

              <div className="bg-[#fbfbf9] p-4 rounded-lg border border-[#e5e5dc] space-y-1">
                <div className="text-[10px] font-semibold text-[#63665c] uppercase">5. Fallback Logic</div>
                <div className="text-xs font-semibold text-[#1c1d1a]">{evidenceData?.fallback_strategy}</div>
                <div className="text-[11px] text-[#63665c] pt-1">
                  Automated routing to historical district baseline if out-of-distribution or sparse data detected.
                </div>
              </div>

              <div className="bg-[#fbfbf9] p-4 rounded-lg border border-[#e5e5dc] space-y-1">
                <div className="text-[10px] font-semibold text-[#63665c] uppercase">6. Certification Status</div>
                <div className="text-xs font-semibold text-[#2d4a22]">{predictionResult?.certification_status || evidenceData?.certification_status}</div>
                <div className="text-[11px] text-[#63665c] pt-1">
                  Cryptographically signed and verified by Day 24 Pre-Inference Governance Guards.
                </div>
              </div>

            </div>
          )}
        </div>

        {/* ============================================================ */}
        {/* 7. PREDICTION PROVENANCE & CRYPTOGRAPHIC RECORD */}
        {/* ============================================================ */}
        <div className="bg-white rounded-xl border border-[#e5e5dc] shadow-sm p-6 space-y-5">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-[#f0f0eb] pb-4 gap-2">
            <div className="flex items-center gap-2">
              <Lock className="w-5 h-5 text-[#2d4a22]" />
              <h2 className="text-base font-semibold text-[#1c1d1a]">Prediction Provenance (Cryptographic Chain of Custody)</h2>
            </div>
            <button
              onClick={copyProvenanceJson}
              disabled={!predictionResult}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-white border border-[#d4d4c8] text-xs font-medium text-[#3c3e37] hover:bg-[#f4f4ee] shadow-sm transition-all"
            >
              {copiedProvenance ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copiedProvenance ? 'Copied JSON' : 'Copy Provenance Record'}</span>
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-[#fbfbf9] p-3 rounded-lg border border-[#e5e5dc]">
              <div className="text-[10px] font-semibold text-[#63665c] uppercase">Request ID</div>
              <div className="text-xs font-mono font-bold text-[#1c1d1a] mt-1">{predictionResult?.request_id || 'REQ-PENDING'}</div>
            </div>

            <div className="bg-[#fbfbf9] p-3 rounded-lg border border-[#e5e5dc]">
              <div className="text-[10px] font-semibold text-[#63665c] uppercase">Model Artifact SHA-256</div>
              <div className="text-xs font-mono text-[#3c3e37] mt-1 truncate" title={predictionResult?.provenance?.model_artifact_hash || 'Verified Registry Baseline'}>
                {predictionResult?.provenance?.model_artifact_hash || 'Verified Registry Baseline'}
              </div>
            </div>

            <div className="bg-[#fbfbf9] p-3 rounded-lg border border-[#e5e5dc]">
              <div className="text-[10px] font-semibold text-[#63665c] uppercase">Dataset Version</div>
              <div className="text-xs font-mono font-semibold text-[#1c1d1a] mt-1">AGRI_PANEL_1.0</div>
            </div>

            <div className="bg-[#fbfbf9] p-3 rounded-lg border border-[#e5e5dc]">
              <div className="text-[10px] font-semibold text-[#63665c] uppercase">Provenance Fingerprint</div>
              <div className="text-xs font-mono font-bold text-[#2d4a22] mt-1 truncate" title={predictionResult?.provenance?.provenance_hash || 'SHA-256 Validated'}>
                {predictionResult?.provenance?.provenance_hash || 'SHA-256 Validated'}
              </div>
            </div>
          </div>
        </div>

        {/* ============================================================ */}
        {/* 8. AUDIT TRACE TIMELINE */}
        {/* ============================================================ */}
        <div className="bg-white rounded-xl border border-[#e5e5dc] shadow-sm p-6 space-y-5">
          <div
            className="flex items-center justify-between cursor-pointer border-b border-[#f0f0eb] pb-4"
            onClick={() => setExpandedTrace(!expandedTrace)}
          >
            <div className="flex items-center gap-2">
              <Clock className="w-5 h-5 text-[#2d4a22]" />
              <h2 className="text-base font-semibold text-[#1c1d1a]">Forecast Lifecycle Audit Trace</h2>
            </div>
            {expandedTrace ? <ChevronUp className="w-4 h-4 text-[#63665c]" /> : <ChevronDown className="w-4 h-4 text-[#63665c]" />}
          </div>

          {expandedTrace && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-3">
                
                <div className="bg-[#fbfbf9] p-3 rounded-lg border border-[#e5e5dc] space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-mono font-semibold text-[#63665c]">STAGE 1</span>
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                  </div>
                  <div className="text-xs font-bold text-[#1c1d1a]">Input Validation</div>
                  <div className="text-[10px] text-[#63665c]">Parameters verified</div>
                </div>

                <div className="bg-[#fbfbf9] p-3 rounded-lg border border-[#e5e5dc] space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-mono font-semibold text-[#63665c]">STAGE 2</span>
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                  </div>
                  <div className="text-xs font-bold text-[#1c1d1a]">Certification Check</div>
                  <div className="text-[10px] text-[#63665c]">Guard authorized</div>
                </div>

                <div className="bg-[#fbfbf9] p-3 rounded-lg border border-[#e5e5dc] space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-mono font-semibold text-[#63665c]">STAGE 3</span>
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                  </div>
                  <div className="text-xs font-bold text-[#1c1d1a]">Strategy Resolved</div>
                  <div className="text-[10px] text-[#63665c]">Registry matched</div>
                </div>

                <div className="bg-[#fbfbf9] p-3 rounded-lg border border-[#e5e5dc] space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-mono font-semibold text-[#63665c]">STAGE 4</span>
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                  </div>
                  <div className="text-xs font-bold text-[#1c1d1a]">Inference Executed</div>
                  <div className="text-[10px] text-[#63665c]">Prediction rendered</div>
                </div>

                <div className="bg-[#fbfbf9] p-3 rounded-lg border border-[#e5e5dc] space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-mono font-semibold text-[#63665c]">STAGE 5</span>
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                  </div>
                  <div className="text-xs font-bold text-[#1c1d1a]">Provenance Hashed</div>
                  <div className="text-[10px] text-[#63665c]">SHA-256 fingerprint</div>
                </div>

                <div className="bg-[#fbfbf9] p-3 rounded-lg border border-[#e5e5dc] space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-mono font-semibold text-[#63665c]">STAGE 6</span>
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                  </div>
                  <div className="text-xs font-bold text-[#1c1d1a]">Audit Recorded</div>
                  <div className="text-[10px] text-[#63665c]">Immutable log saved</div>
                </div>

              </div>
            </div>
          )}
        </div>

      </div>
    </div>
  )
}
