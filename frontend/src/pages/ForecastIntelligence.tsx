import React, { useState, useMemo } from 'react'
import {
  useForecastStrategies,
  useForecastCoverage,
  useForecastCertificationSummary,
  useForecastPredict,
  useForecastAuditLogs,
  useForecastHealth,
} from '../services/api'
import {
  ForecastStrategyItem,
  ForecastPredictResponse,
  ForecastAuditItem,
} from '../types/modeling'
import {
  ShieldCheck,
  Cpu,
  Database,
  History,
  AlertTriangle,
  CheckCircle2,
  ChevronRight,
  Info,
  Layers,
  Search,
  Sparkles,
  Lock,
  FileText,
  Activity,
  ArrowUpRight,
  Filter,
  Copy,
  Check,
  RefreshCw,
} from 'lucide-react'

export const ForecastIntelligence: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'wizard' | 'registry' | 'audit'>('wizard')

  // Form state
  const [selectedCrop, setSelectedCrop] = useState<string>('Oilseeds')
  const [selectedState, setSelectedState] = useState<string>('Punjab')
  const [selectedDistrict, setSelectedDistrict] = useState<string>('Ludhiana')
  const [forecastYear, setForecastYear] = useState<number>(2018)
  const [yieldLag1, setYieldLag1] = useState<string>('')
  const [yieldRoll3, setYieldRoll3] = useState<string>('')
  const [areaLag1, setAreaLag1] = useState<string>('')
  const [copiedHash, setCopiedHash] = useState<boolean>(false)

  // API Hooks
  const { data: strategiesData, isLoading: stratLoading } = useForecastStrategies()
  const { data: coverageData, isLoading: covLoading } = useForecastCoverage()
  const { data: certSummary } = useForecastCertificationSummary()
  const { data: auditData, refetch: refetchAudit, isFetching: auditFetching } = useForecastAuditLogs(50)
  const { data: healthData } = useForecastHealth()
  const predictMutation = useForecastPredict()

  // Selected strategy metadata
  const currentStrategy = useMemo(() => {
    return strategiesData?.strategies.find((s) => s.crop.toLowerCase() === selectedCrop.toLowerCase())
  }, [strategiesData, selectedCrop])

  // Available states for selected crop
  const availableStates = useMemo(() => {
    if (!coverageData?.coverage) return []
    const states = coverageData.coverage
      .filter((c) => c.crop.toLowerCase() === selectedCrop.toLowerCase())
      .map((c) => c.state)
    return Array.from(new Set(states)).sort()
  }, [coverageData, selectedCrop])

  // Available districts for selected crop & state
  const availableDistricts = useMemo(() => {
    if (!coverageData?.coverage) return []
    return coverageData.coverage
      .filter(
        (c) =>
          c.crop.toLowerCase() === selectedCrop.toLowerCase() &&
          c.state.toLowerCase() === selectedState.toLowerCase()
      )
      .map((c) => c.district)
      .sort()
  }, [coverageData, selectedCrop, selectedState])

  // Update state/district selection when crop changes
  const handleCropChange = (crop: string) => {
    setSelectedCrop(crop)
    if (coverageData?.coverage) {
      const match = coverageData.coverage.find((c) => c.crop.toLowerCase() === crop.toLowerCase())
      if (match) {
        setSelectedState(match.state)
        setSelectedDistrict(match.district)
      }
    }
  }

  const handleStateChange = (state: string) => {
    setSelectedState(state)
    if (coverageData?.coverage) {
      const match = coverageData.coverage.find(
        (c) => c.crop.toLowerCase() === selectedCrop.toLowerCase() && c.state.toLowerCase() === state.toLowerCase()
      )
      if (match) {
        setSelectedDistrict(match.district)
      }
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    predictMutation.mutate({
      crop: selectedCrop,
      state: selectedState,
      district: selectedDistrict,
      forecast_year: forecastYear,
      yield_lag_1: yieldLag1 ? parseFloat(yieldLag1) : undefined,
      yield_rolling_3yr_mean: yieldRoll3 ? parseFloat(yieldRoll3) : undefined,
      area_lag_1: areaLag1 ? parseFloat(areaLag1) : undefined,
    })
  }

  const handleCopyProvenance = (hash: string) => {
    navigator.clipboard.writeText(hash)
    setCopiedHash(true)
    setTimeout(() => setCopiedHash(false), 2000)
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'PRODUCTION_READY':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <CheckCircle2 className="w-3.5 h-3.5" /> Production Ready ML
          </span>
        )
      case 'CONDITIONAL_PRODUCTION':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20">
            <AlertTriangle className="w-3.5 h-3.5" /> Conditional ML (Clipped)
          </span>
        )
      case 'BASELINE_PRODUCTION':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-blue-500/10 text-blue-400 border border-blue-500/20">
            <Database className="w-3.5 h-3.5" /> Certified Baseline
          </span>
        )
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-slate-500/10 text-slate-400 border border-slate-500/20">
            {status}
          </span>
        )
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 md:p-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-white">
              Production Forecast Serving & Governance
            </h1>
          </div>
          <p className="text-sm text-slate-400 mt-1 max-w-3xl">
            Authoritative, deterministic multi-crop forecast decision engine enforcing Day 23 certification, pre-inference guards, variance clipping, cryptographic provenance, and immutable audit logs.
          </p>
        </div>

        {/* Health & Status Indicator */}
        <div className="flex items-center gap-3 bg-slate-900/80 border border-slate-800 rounded-xl px-4 py-2.5">
          <div className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs font-medium text-slate-300">Guard: {healthData?.governance_guard || 'ACTIVE_STRICT'}</span>
          </div>
          <span className="text-slate-600">|</span>
          <span className="text-xs text-slate-400 font-mono">14 Crops Certified</span>
        </div>
      </div>

      {/* Validation Boundary Limitation Notice */}
      <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-4 text-amber-300/90 text-sm flex items-start gap-3">
        <Info className="w-5 h-5 flex-shrink-0 mt-0.5 text-amber-400" />
        <div>
          <span className="font-semibold text-amber-200">Validation Limitation Notice: </span>
          The available historical dataset ends in 2017 (ICRISAT / IMD panel). No independent post-2017 holdout is available. Forecast strategy certification is therefore based on expanding walk-forward validation over historical origins 2014–2017. Requests for years &gt; 2017 use certified historical lag persistence models.
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex items-center gap-2 border-b border-slate-800">
        <button
          onClick={() => setActiveTab('wizard')}
          className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
            activeTab === 'wizard'
              ? 'border-emerald-400 text-emerald-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Sparkles className="w-4 h-4" />
          Forecast Decision Wizard
        </button>
        <button
          onClick={() => setActiveTab('registry')}
          className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
            activeTab === 'registry'
              ? 'border-emerald-400 text-emerald-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Layers className="w-4 h-4" />
          Certified Strategy Registry ({strategiesData?.total_strategies || 14})
        </button>
        <button
          onClick={() => setActiveTab('audit')}
          className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
            activeTab === 'audit'
              ? 'border-emerald-400 text-emerald-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <History className="w-4 h-4" />
          Prediction Audit Trail ({auditData?.total_events || 0})
        </button>
      </div>

      {/* Tab 1: Forecast Decision Wizard */}
      {activeTab === 'wizard' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Column: Form */}
          <div className="lg:col-span-5 space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-6">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                <Cpu className="w-5 h-5 text-emerald-400" />
                Configure Forecast Request
              </h2>

              <form onSubmit={handleSubmit} className="space-y-4">
                {/* 1. Crop Selection */}
                <div>
                  <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
                    1. Crop Commodity
                  </label>
                  <select
                    value={selectedCrop}
                    onChange={(e) => handleCropChange(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-white focus:outline-none focus:border-emerald-500"
                  >
                    {strategiesData?.strategies.map((s) => (
                      <option key={s.crop} value={s.crop}>
                        {s.crop} ({s.certification_status === 'PRODUCTION_READY' ? 'ML Certified' : s.certification_status === 'CONDITIONAL_PRODUCTION' ? 'Conditional ML' : 'Statistical Baseline'})
                      </option>
                    ))}
                  </select>
                </div>

                {/* Strategy Info Chip */}
                {currentStrategy && (
                  <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg space-y-1.5">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-slate-400">Certified Route:</span>
                      {getStatusBadge(currentStrategy.certification_status)}
                    </div>
                    <div className="text-xs text-slate-300 font-medium">
                      {currentStrategy.primary_strategy}
                    </div>
                  </div>
                )}

                {/* 2. State Selection */}
                <div>
                  <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
                    2. State Geography
                  </label>
                  <select
                    value={selectedState}
                    onChange={(e) => handleStateChange(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-white focus:outline-none focus:border-emerald-500"
                  >
                    {availableStates.map((st) => (
                      <option key={st} value={st}>
                        {st}
                      </option>
                    ))}
                  </select>
                </div>

                {/* 3. District Selection */}
                <div>
                  <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
                    3. District
                  </label>
                  <select
                    value={selectedDistrict}
                    onChange={(e) => setSelectedDistrict(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-white focus:outline-none focus:border-emerald-500"
                  >
                    {availableDistricts.map((d) => (
                      <option key={d} value={d}>
                        {d}
                      </option>
                    ))}
                  </select>
                </div>

                {/* 4. Target Year */}
                <div>
                  <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
                    4. Target Forecast Year
                  </label>
                  <input
                    type="number"
                    value={forecastYear}
                    onChange={(e) => setForecastYear(parseInt(e.target.value) || 2018)}
                    min={1966}
                    max={2030}
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3.5 py-2 text-sm text-white focus:outline-none focus:border-emerald-500"
                  />
                  <span className="text-[11px] text-slate-500 mt-1 block">
                    Historical panel covers 1966–2017. Target year 2018 is standard horizon.
                  </span>
                </div>

                {/* Optional Feature Overrides (Accordion style) */}
                <details className="bg-slate-950/40 border border-slate-800/80 rounded-lg p-3 text-xs">
                  <summary className="font-semibold text-slate-300 cursor-pointer hover:text-emerald-400">
                    Optional Custom Feature Inputs
                  </summary>
                  <div className="grid grid-cols-1 gap-3 mt-3 pt-3 border-t border-slate-800">
                    <div>
                      <label className="text-slate-400 block mb-1">Yield Lag 1 (kg/ha):</label>
                      <input
                        type="number"
                        step="0.01"
                        placeholder="Auto-inferred from history"
                        value={yieldLag1}
                        onChange={(e) => setYieldLag1(e.target.value)}
                        className="w-full bg-slate-900 border border-slate-700 rounded p-1.5 text-xs text-white"
                      />
                    </div>
                    <div>
                      <label className="text-slate-400 block mb-1">Yield Rolling 3-Yr Mean (kg/ha):</label>
                      <input
                        type="number"
                        step="0.01"
                        placeholder="Auto-inferred from history"
                        value={yieldRoll3}
                        onChange={(e) => setYieldRoll3(e.target.value)}
                        className="w-full bg-slate-900 border border-slate-700 rounded p-1.5 text-xs text-white"
                      />
                    </div>
                    <div>
                      <label className="text-slate-400 block mb-1">Cultivated Area Lag 1 (ha):</label>
                      <input
                        type="number"
                        step="0.01"
                        placeholder="Auto-inferred from history"
                        value={areaLag1}
                        onChange={(e) => setAreaLag1(e.target.value)}
                        className="w-full bg-slate-900 border border-slate-700 rounded p-1.5 text-xs text-white"
                      />
                    </div>
                  </div>
                </details>

                {/* Submit Button */}
                <button
                  type="submit"
                  disabled={predictMutation.isPending}
                  className="w-full py-3 px-4 bg-emerald-500 hover:bg-emerald-600 disabled:bg-slate-800 disabled:text-slate-500 text-slate-950 font-semibold rounded-lg shadow-lg shadow-emerald-500/10 flex items-center justify-center gap-2 transition-all mt-4"
                >
                  {predictMutation.isPending ? (
                    <>
                      <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                      Evaluating Governance & Executing...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      Execute Governed Forecast
                    </>
                  )}
                </button>
              </form>
            </div>
          </div>

          {/* Right Column: Prediction Result & Provenance */}
          <div className="lg:col-span-7 space-y-6">
            {predictMutation.data ? (
              <div className="space-y-6">
                {/* Result Card */}
                {predictMutation.data.status === 'SUCCESS' ? (
                  <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-6">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
                      <div>
                        <div className="text-xs text-slate-400 uppercase tracking-wider font-semibold">
                          Forecast Yield Prediction
                        </div>
                        <div className="flex items-baseline gap-2 mt-1">
                          <span className="text-4xl font-extrabold text-white tracking-tight">
                            {predictMutation.data.prediction?.toLocaleString(undefined, {
                              minimumFractionDigits: 2,
                              maximumFractionDigits: 2,
                            })}
                          </span>
                          <span className="text-sm font-semibold text-emerald-400">
                            {predictMutation.data.unit}
                          </span>
                        </div>
                      </div>

                      <div className="flex flex-col md:items-end gap-1.5">
                        {getStatusBadge(predictMutation.data.certification_status)}
                        <span className="text-xs text-slate-400 font-mono">
                          Req ID: {predictMutation.data.request_id}
                        </span>
                      </div>
                    </div>

                    {/* Geographies & Routing Summary */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-slate-950/80 rounded-lg border border-slate-800 text-xs">
                      <div>
                        <div className="text-slate-500 font-medium">Crop</div>
                        <div className="text-white font-semibold mt-0.5">{predictMutation.data.crop}</div>
                      </div>
                      <div>
                        <div className="text-slate-500 font-medium">Location</div>
                        <div className="text-white font-semibold mt-0.5">
                          {predictMutation.data.district}, {predictMutation.data.state}
                        </div>
                      </div>
                      <div>
                        <div className="text-slate-500 font-medium">Strategy</div>
                        <div className="text-white font-semibold mt-0.5">{predictMutation.data.strategy}</div>
                      </div>
                      <div>
                        <div className="text-slate-500 font-medium">Fallback Route</div>
                        <div className="text-white font-semibold mt-0.5">
                          {predictMutation.data.fallback_used ? (
                            <span className="text-amber-400">Applied</span>
                          ) : (
                            <span className="text-emerald-400">Direct Route</span>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Fallback alert if applied */}
                    {predictMutation.data.fallback_used && (
                      <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg text-xs text-amber-300 flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0" />
                        <span>
                          <strong>Fallback Policy Activated: </strong>
                          {predictMutation.data.fallback_reason}
                        </span>
                      </div>
                    )}

                    {/* Strategy Explanation */}
                    <div className="space-y-2">
                      <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                        <Info className="w-4 h-4 text-emerald-400" />
                        Why This Prediction? (Certification Directive)
                      </div>
                      <p className="text-sm text-slate-300 leading-relaxed bg-slate-950/40 p-3.5 rounded-lg border border-slate-800/80">
                        {predictMutation.data.strategy_explanation}
                      </p>
                    </div>

                    {/* Provenance Record Card */}
                    {predictMutation.data.provenance && (
                      <div className="space-y-3 pt-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                            <Lock className="w-4 h-4 text-cyan-400" />
                            Cryptographic Provenance Lineage
                          </span>
                          <button
                            onClick={() =>
                              handleCopyProvenance(predictMutation.data?.provenance?.provenance_hash || '')
                            }
                            className="text-xs text-slate-400 hover:text-emerald-400 flex items-center gap-1 transition-colors"
                          >
                            {copiedHash ? (
                              <>
                                <Check className="w-3.5 h-3.5 text-emerald-400" /> Copied Hash
                              </>
                            ) : (
                              <>
                                <Copy className="w-3.5 h-3.5" /> Copy Fingerprint
                              </>
                            )}
                          </button>
                        </div>

                        <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 space-y-3 font-mono text-xs">
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-slate-300">
                            <div>
                              <span className="text-slate-500">Model Version: </span>
                              {predictMutation.data.provenance.model_version}
                            </div>
                            <div>
                              <span className="text-slate-500">Validation MAE: </span>
                              {predictMutation.data.provenance.validation_mae} kg/ha
                            </div>
                            <div>
                              <span className="text-slate-500">Fold Win Rate: </span>
                              {predictMutation.data.provenance.fold_win_rate_pct}%
                            </div>
                            <div>
                              <span className="text-slate-500">Gain vs Base: </span>
                              +{predictMutation.data.provenance.gain_vs_baseline_pct}%
                            </div>
                          </div>

                          <div className="pt-2 border-t border-slate-800/80 text-[11px] text-slate-400 break-all">
                            <span className="text-slate-500">Lineage Fingerprint: </span>
                            <span className="text-cyan-400">{predictMutation.data.provenance.provenance_hash}</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  /* Rejection Banner */
                  <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-6 space-y-4">
                    <div className="flex items-center gap-3 text-red-400 font-semibold text-lg">
                      <AlertTriangle className="w-6 h-6" />
                      Request Rejected by Governance Guard ({predictMutation.data.error_code})
                    </div>
                    <p className="text-sm text-red-200/90 leading-relaxed">
                      {predictMutation.data.error_message}
                    </p>
                    <div className="p-3 bg-red-950/30 rounded border border-red-500/20 text-xs text-red-300 font-mono">
                      Status: REJECTED | Code: {predictMutation.data.error_code} | Logged to prediction audit log.
                    </div>
                  </div>
                )}
              </div>
            ) : (
              /* Empty State Banner */
              <div className="bg-slate-900/50 border border-slate-800/80 border-dashed rounded-xl p-12 text-center space-y-4">
                <div className="w-12 h-12 rounded-full bg-slate-800 flex items-center justify-center mx-auto text-slate-400">
                  <ShieldCheck className="w-6 h-6 text-emerald-400" />
                </div>
                <div>
                  <h3 className="text-base font-semibold text-white">Ready for Governed Inference</h3>
                  <p className="text-xs text-slate-400 mt-1 max-w-sm mx-auto">
                    Select your crop and geographic parameters on the left to execute certified forecast serving.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Tab 2: Certified Strategy Registry */}
      {activeTab === 'registry' && (
        <div className="space-y-6">
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
              <div className="text-xs text-slate-400 font-medium">Total Certified Crops</div>
              <div className="text-2xl font-bold text-white mt-1">
                {strategiesData?.total_strategies || 14}
              </div>
              <div className="text-xs text-slate-500 mt-1">100% Bitwise Reproducible</div>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
              <div className="text-xs text-slate-400 font-medium">Production Ready ML</div>
              <div className="text-2xl font-bold text-emerald-400 mt-1">
                {certSummary?.production_ready_crops.length || 1} Crop
              </div>
              <div className="text-xs text-emerald-500/80 mt-1">Oilseeds (Random Forest)</div>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
              <div className="text-xs text-slate-400 font-medium">Conditional ML (Variance Clipped)</div>
              <div className="text-2xl font-bold text-amber-400 mt-1">
                {certSummary?.conditional_production_crops.length || 1} Crop
              </div>
              <div className="text-xs text-amber-500/80 mt-1">Sugarcane (3-Sigma Clip)</div>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
              <div className="text-xs text-slate-400 font-medium">Certified Statistical Baselines</div>
              <div className="text-2xl font-bold text-blue-400 mt-1">
                {certSummary?.baseline_production_crops.length || 12} Crops
              </div>
              <div className="text-xs text-blue-500/80 mt-1">District Mean / 3-Yr Rolling</div>
            </div>
          </div>

          {/* Strategy Registry Table */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <Layers className="w-4 h-4 text-emerald-400" />
                Authoritative Multi-Crop Forecast Strategies
              </h3>
              <span className="text-xs text-slate-400 font-mono">
                Temporal Boundary: 1966–2017
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950 text-slate-400 font-semibold border-b border-slate-800">
                  <tr>
                    <th className="py-3 px-4">Crop Commodity</th>
                    <th className="py-3 px-4">Certification Status</th>
                    <th className="py-3 px-4">Primary Strategy</th>
                    <th className="py-3 px-4">Fallback Policy</th>
                    <th className="py-3 px-4 text-right">Strategy MAE</th>
                    <th className="py-3 px-4 text-right">Baseline MAE</th>
                    <th className="py-3 px-4 text-right">Gain vs Base</th>
                    <th className="py-3 px-4 text-right">Win Rate</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  {strategiesData?.strategies.map((s) => (
                    <tr key={s.crop} className="hover:bg-slate-800/30 transition-colors">
                      <td className="py-3.5 px-4 font-semibold text-white">{s.crop}</td>
                      <td className="py-3.5 px-4">{getStatusBadge(s.certification_status)}</td>
                      <td className="py-3.5 px-4 text-slate-200">{s.primary_strategy}</td>
                      <td className="py-3.5 px-4 text-slate-400">{s.fallback_strategy}</td>
                      <td className="py-3.5 px-4 text-right font-mono font-semibold text-white">
                        {s.strategy_mae.toFixed(1)}
                      </td>
                      <td className="py-3.5 px-4 text-right font-mono text-slate-400">
                        {s.baseline_mae.toFixed(1)}
                      </td>
                      <td
                        className={`py-3.5 px-4 text-right font-mono font-semibold ${
                          s.gain_vs_baseline_pct > 0 ? 'text-emerald-400' : 'text-slate-400'
                        }`}
                      >
                        {s.gain_vs_baseline_pct > 0 ? `+${s.gain_vs_baseline_pct.toFixed(1)}%` : '0.0%'}
                      </td>
                      <td className="py-3.5 px-4 text-right font-mono text-slate-300">
                        {s.fold_win_rate_pct.toFixed(0)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Tab 3: Prediction Audit Trail */}
      {activeTab === 'audit' && (
        <div className="space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                  <History className="w-4 h-4 text-emerald-400" />
                  Live Prediction & Rejection Audit Trail
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Immutable log of all forecasting inference requests and governance rejections.
                </p>
              </div>
              <button
                onClick={() => refetchAudit()}
                disabled={auditFetching}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-300 rounded-lg transition-colors"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${auditFetching ? 'animate-spin' : ''}`} />
                Refresh Logs
              </button>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-slate-950 text-slate-400 font-semibold border-b border-slate-800">
                  <tr>
                    <th className="py-3 px-4">Request ID</th>
                    <th className="py-3 px-4">Timestamp (UTC)</th>
                    <th className="py-3 px-4">Crop</th>
                    <th className="py-3 px-4">Location</th>
                    <th className="py-3 px-4">Status</th>
                    <th className="py-3 px-4 text-right">Prediction</th>
                    <th className="py-3 px-4">Fallback</th>
                    <th className="py-3 px-4">Provenance Fingerprint</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  {auditData?.events && auditData.events.length > 0 ? (
                    auditData.events.map((evt) => (
                      <tr key={evt.request_id} className="hover:bg-slate-800/30 transition-colors">
                        <td className="py-3 px-4 text-white font-semibold">{evt.request_id}</td>
                        <td className="py-3 px-4 text-slate-400">{evt.timestamp.split('T')[0]} {evt.timestamp.split('T')[1]?.substring(0, 8)}</td>
                        <td className="py-3 px-4 text-slate-200">{evt.crop}</td>
                        <td className="py-3 px-4 text-slate-300">{evt.district}, {evt.state}</td>
                        <td className="py-3 px-4">
                          {evt.status === 'SUCCESS' ? (
                            <span className="text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                              SUCCESS
                            </span>
                          ) : (
                            <span className="text-red-400 bg-red-500/10 px-2 py-0.5 rounded border border-red-500/20">
                              {evt.error_code || 'REJECTED'}
                            </span>
                          )}
                        </td>
                        <td className="py-3 px-4 text-right font-semibold text-white">
                          {evt.prediction !== '' && evt.prediction !== null ? `${evt.prediction} ${evt.unit}` : '-'}
                        </td>
                        <td className="py-3 px-4">
                          {evt.fallback_used ? (
                            <span className="text-amber-400">YES</span>
                          ) : (
                            <span className="text-slate-500">NO</span>
                          )}
                        </td>
                        <td className="py-3 px-4 text-slate-500 truncate max-w-xs">
                          {evt.provenance_hash || '-'}
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={8} className="py-8 text-center text-slate-500">
                        No audit events recorded yet. Execute a forecast in the wizard tab to log events.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
