import React, { useState } from 'react'
import {
  Activity,
  ShieldCheck,
  AlertTriangle,
  CheckCircle2,
  TrendingUp,
  BarChart3,
  Calendar,
  Layers,
  MapPin,
  Clock,
  ArrowUpRight,
  Info,
  RefreshCw,
  Search,
  Filter,
  Eye,
  Sliders,
  Sparkles,
  Zap,
  Radio,
  FileText
} from 'lucide-react'
import {
  useForecastMonitoringSummary,
  useForecastOperations,
  usePredictionDistributions,
  useForecastDrift,
  useOutcomeEvaluations,
  useErrorDecomposition,
  useBiasDiagnostics,
  useForecastMonitoringAlerts,
  useForecastMonitoringHealth
} from '../services/api'
import { Button } from '../components/ui/Button'

export const ForecastMonitoring: React.FC = () => {
  const [selectedCrop, setSelectedCrop] = useState<string>('Oilseeds')
  const [selectedYear, setSelectedYear] = useState<number>(2017)
  const [errorTab, setErrorTab] = useState<'temporal' | 'geographic' | 'regimes'>('temporal')

  // React Query hooks for Day 30 Forecast Monitoring
  const { data: summary, isLoading: loadingSummary, refetch: refetchSummary } = useForecastMonitoringSummary()
  const { data: operations, isLoading: loadingOps } = useForecastOperations(selectedCrop)
  const { data: distributions, isLoading: loadingDist } = usePredictionDistributions(selectedCrop)
  const { data: drift, isLoading: loadingDrift } = useForecastDrift(selectedCrop)
  const { data: outcomes, isLoading: loadingOutcomes } = useOutcomeEvaluations({
    crop: selectedCrop,
    forecast_year: selectedYear,
  })
  const { data: errors, isLoading: loadingErrors } = useErrorDecomposition(selectedCrop)
  const { data: bias, isLoading: loadingBias } = useBiasDiagnostics(selectedCrop)
  const { data: alerts, isLoading: loadingAlerts } = useForecastMonitoringAlerts()
  const { data: health } = useForecastMonitoringHealth()

  const cropsList = [
    'Oilseeds', 'Sugarcane', 'Rice', 'Wheat', 'Chickpea',
    'Kharif Sorghum', 'Minor Pulses', 'Maize', 'Sesamum',
    'Pigeonpea', 'Rapeseed and Mustard', 'Groundnut', 'Sorghum', 'Pearl Millet'
  ]

  const activeCropDist = distributions?.distributions.find(
    (d) => d.crop.toLowerCase() === selectedCrop.toLowerCase()
  )

  const activeCropBias = bias?.crops.find(
    (b) => b.crop.toLowerCase() === selectedCrop.toLowerCase()
  )

  return (
    <div className="min-h-screen bg-background text-foreground font-sans pb-24">
      {/* ----------------------------------------------------------------- */}
      {/* 1. Header Banner */}
      {/* ----------------------------------------------------------------- */}
      <section className="border-b border-border bg-card pt-10 pb-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-primary/10 text-primary border border-primary/20 flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5" />
                  Governance Layer
                </span>
                <span className="text-xs text-muted-foreground font-mono">AGRI_PANEL_1.0 • Out-of-Time Evaluation</span>
              </div>
              <h1 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
                Forecast Monitoring & Outcome Intelligence
              </h1>
              <p className="mt-1.5 text-base text-muted-foreground max-w-3xl">
                Monitoring live forecast operations, prediction distributions, statistical covariate drift,
                and leak-free post-outcome forecast accuracy.
              </p>
            </div>

            <div className="flex items-center gap-3 self-start md:self-auto">
              <Button
                variant="outline"
                size="sm"
                onClick={() => refetchSummary()}
                className="border-border text-foreground hover:bg-muted/30"
              >
                <RefreshCw className="w-4 h-4 mr-1.5" />
                Refresh Telemetry
              </Button>
            </div>
          </div>

          {/* Quick Filters */}
          <div className="mt-8 flex flex-wrap items-center gap-4 pt-4 border-t border-border/60">
            <div className="flex items-center gap-2">
              <label className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Crop Filter:</label>
              <select
                value={selectedCrop}
                onChange={(e) => setSelectedCrop(e.target.value)}
                className="bg-white border border-border rounded-lg px-3 py-1.5 text-sm font-medium text-foreground shadow-sm focus:outline-none focus:ring-2 focus:ring-primary"
              >
                {cropsList.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>

            <div className="flex items-center gap-2">
              <label className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Evaluation Horizon:</label>
              <select
                value={selectedYear}
                onChange={(e) => setSelectedYear(Number(e.target.value))}
                className="bg-white border border-border rounded-lg px-3 py-1.5 text-sm font-medium text-foreground shadow-sm focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <option value={2017}>2017 (Observed Harvest Fold 4)</option>
                <option value={2016}>2016 (Observed Harvest Fold 3)</option>
                <option value={2015}>2015 (Observed Harvest Fold 2)</option>
                <option value={2014}>2014 (Observed Harvest Fold 1)</option>
                <option value={2026}>2026 (Unharvested Future Horizon)</option>
              </select>
            </div>
          </div>
        </div>
      </section>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-8 space-y-8">
        {/* ----------------------------------------------------------------- */}
        {/* 2. Executive Status & Operational Metrics Cards */}
        {/* ----------------------------------------------------------------- */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
          {/* Status Card */}
          <div className="bg-white border border-border rounded-xl p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">System State</span>
              {summary?.monitoring_status === 'HEALTHY' ? (
                <span className="p-1.5 rounded-lg bg-emerald-50 text-emerald-700">
                  <CheckCircle2 className="w-5 h-5" />
                </span>
              ) : summary?.monitoring_status === 'DRIFT_DETECTED' ? (
                <span className="p-1.5 rounded-lg bg-amber-50 text-amber-700">
                  <AlertTriangle className="w-5 h-5" />
                </span>
              ) : (
                <span className="p-1.5 rounded-lg bg-blue-50 text-blue-700">
                  <Radio className="w-5 h-5" />
                </span>
              )}
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-2xl font-bold text-foreground">{summary?.monitoring_status || 'HEALTHY'}</span>
            </div>
            <p className="mt-2 text-xs text-muted-foreground line-clamp-2">
              {summary?.status_reason || 'All governed operations within acceptable bounds.'}
            </p>
          </div>

          {/* Audit Records Volume */}
          <div className="bg-white border border-border rounded-xl p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Forecast Operations</span>
              <span className="p-1.5 rounded-lg bg-primary/10 text-primary">
                <FileText className="w-5 h-5" />
              </span>
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-2xl font-bold text-foreground">
                {loadingSummary ? '...' : summary?.total_forecast_requests.toLocaleString()}
              </span>
              <span className="text-xs text-muted-foreground">requests logged</span>
            </div>
            <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
              <span className="text-emerald-700 font-medium">✓ {summary?.successful_forecasts} Success</span>
              <span className="text-amber-700 font-medium">✕ {summary?.rejected_requests} Blocked</span>
            </div>
          </div>

          {/* Evaluated Outcomes */}
          <div className="bg-white border border-border rounded-xl p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Evaluated Outcomes</span>
              <span className="p-1.5 rounded-lg bg-indigo-50 text-indigo-700">
                <ShieldCheck className="w-5 h-5" />
              </span>
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-2xl font-bold text-foreground">
                {loadingSummary ? '...' : summary?.evaluated_outcomes_count}
              </span>
              <span className="text-xs text-muted-foreground">folds verified</span>
            </div>
            <p className="mt-2 text-xs text-muted-foreground">
              Walk-forward test years (2014–2017) with zero future outcome leakage.
            </p>
          </div>

          {/* Active Alerts */}
          <div className="bg-white border border-border rounded-xl p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Active Alerts</span>
              <span className="p-1.5 rounded-lg bg-amber-50 text-amber-700">
                <AlertTriangle className="w-5 h-5" />
              </span>
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-2xl font-bold text-foreground">{summary?.active_alerts_count || 0}</span>
              <span className="text-xs text-muted-foreground">evidence signals</span>
            </div>
            <p className="mt-2 text-xs text-muted-foreground">
              Covariate drift and systematic directional bias tracking.
            </p>
          </div>
        </section>

        {/* ----------------------------------------------------------------- */}
        {/* 3. Prediction Distribution vs Historical Baseline Reference */}
        {/* ----------------------------------------------------------------- */}
        <section className="bg-white border border-border rounded-xl p-6 shadow-sm">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between pb-5 border-b border-border/60 gap-2">
            <div>
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 text-xs font-bold rounded bg-sky-500/10 text-sky-600 uppercase">
                  Statistical Moments
                </span>
                <span className="text-xs text-muted-foreground">Semantic Classification: MONITORING</span>
              </div>
              <h2 className="text-lg font-bold text-foreground mt-1">
                Prediction Distribution Monitoring: {selectedCrop}
              </h2>
              <p className="text-xs text-muted-foreground">
                Empirical distribution of live/logged predictions compared against canonical historical reference distribution (1966–2017).
              </p>
            </div>

            {activeCropDist?.distribution_shift_detected && (
              <div className="px-3 py-1.5 rounded-lg bg-amber-50 border border-amber-200 text-amber-800 text-xs font-semibold flex items-center gap-1.5">
                <AlertTriangle className="w-4 h-4 text-amber-600" />
                Distribution Shift Detected: {activeCropDist.shift_value}% {activeCropDist.shift_metric}
              </div>
            )}
          </div>

          <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Live Audit Prediction Moments */}
            <div className="bg-muted/30 border border-border rounded-xl p-5">
              <div className="flex items-center justify-between pb-3 border-b border-border">
                <h3 className="text-sm font-bold text-foreground flex items-center gap-1.5">
                  <Zap className="w-4 h-4 text-primary" />
                  Live Generated Predictions
                </h3>
                <span className="text-xs font-mono bg-card px-2 py-0.5 rounded border border-border">
                  N = {activeCropDist?.current_predictions.count || 0}
                </span>
              </div>

              <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                <div className="bg-white p-3 rounded-lg border border-border">
                  <span className="text-[11px] text-muted-foreground block">Mean</span>
                  <span className="text-base font-bold text-foreground">{activeCropDist?.current_predictions.mean || 0}</span>
                  <span className="text-[10px] text-muted-foreground/60 block">kg/ha</span>
                </div>
                <div className="bg-white p-3 rounded-lg border border-border">
                  <span className="text-[11px] text-muted-foreground block">Median</span>
                  <span className="text-base font-bold text-foreground">{activeCropDist?.current_predictions.median || 0}</span>
                  <span className="text-[10px] text-muted-foreground/60 block">kg/ha</span>
                </div>
                <div className="bg-white p-3 rounded-lg border border-border">
                  <span className="text-[11px] text-muted-foreground block">Std Dev</span>
                  <span className="text-base font-bold text-foreground">{activeCropDist?.current_predictions.std || 0}</span>
                  <span className="text-[10px] text-muted-foreground/60 block">kg/ha</span>
                </div>
                <div className="bg-white p-3 rounded-lg border border-border">
                  <span className="text-[11px] text-muted-foreground block">P10 – P90</span>
                  <span className="text-xs font-bold text-foreground block mt-0.5">
                    {activeCropDist?.current_predictions.p10 || 0} - {activeCropDist?.current_predictions.p90 || 0}
                  </span>
                  <span className="text-[10px] text-muted-foreground/60 block">kg/ha</span>
                </div>
              </div>

              <div className="mt-4 text-xs text-muted-foreground flex items-center justify-between">
                <span>Active Strategy: <strong className="text-foreground">{activeCropDist?.strategy}</strong></span>
                <span>Min: {activeCropDist?.current_predictions.min_val} | Max: {activeCropDist?.current_predictions.max_val}</span>
              </div>
            </div>

            {/* Historical Reference Moments */}
            <div className="bg-muted/30 border border-border rounded-xl p-5">
              <div className="flex items-center justify-between pb-3 border-b border-border">
                <h3 className="text-sm font-bold text-foreground flex items-center gap-1.5">
                  <Calendar className="w-4 h-4 text-sky-600" />
                  Historical Reference Baseline (1966–2017)
                </h3>
                <span className="text-xs font-mono bg-card px-2 py-0.5 rounded border border-border">
                  N = {activeCropDist?.historical_reference.count.toLocaleString() || 0}
                </span>
              </div>

              <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                <div className="bg-white p-3 rounded-lg border border-border">
                  <span className="text-[11px] text-muted-foreground block">Mean</span>
                  <span className="text-base font-bold text-foreground">{activeCropDist?.historical_reference.mean || 0}</span>
                  <span className="text-[10px] text-muted-foreground/60 block">kg/ha</span>
                </div>
                <div className="bg-white p-3 rounded-lg border border-border">
                  <span className="text-[11px] text-muted-foreground block">Median</span>
                  <span className="text-base font-bold text-foreground">{activeCropDist?.historical_reference.median || 0}</span>
                  <span className="text-[10px] text-muted-foreground/60 block">kg/ha</span>
                </div>
                <div className="bg-white p-3 rounded-lg border border-border">
                  <span className="text-[11px] text-muted-foreground block">Std Dev</span>
                  <span className="text-base font-bold text-foreground">{activeCropDist?.historical_reference.std || 0}</span>
                  <span className="text-[10px] text-muted-foreground/60 block">kg/ha</span>
                </div>
                <div className="bg-white p-3 rounded-lg border border-border">
                  <span className="text-[11px] text-muted-foreground block">P10 – P90</span>
                  <span className="text-xs font-bold text-foreground block mt-0.5">
                    {activeCropDist?.historical_reference.p10 || 0} - {activeCropDist?.historical_reference.p90 || 0}
                  </span>
                  <span className="text-[10px] text-muted-foreground/60 block">kg/ha</span>
                </div>
              </div>

              <div className="mt-4 text-xs text-muted-foreground flex items-center justify-between">
                <span>Dataset: <strong className="text-foreground">AGRI_PANEL_1.0</strong></span>
                <span>Min: {activeCropDist?.historical_reference.min_val} | Max: {activeCropDist?.historical_reference.max_val}</span>
              </div>
            </div>
          </div>
        </section>

        {/* ----------------------------------------------------------------- */}
        {/* 4. Post-Outcome Evaluation (Leak-Free) */}
        {/* ----------------------------------------------------------------- */}
        <section className="bg-white border border-border rounded-xl p-6 shadow-sm">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between pb-5 border-b border-border/60 gap-2">
            <div>
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 text-xs font-bold rounded bg-emerald-50 text-emerald-700 uppercase">
                  Strict Temporal Cutoff
                </span>
                <span className="text-xs text-muted-foreground">Semantic Classification: POST_OUTCOME_EVALUATION</span>
              </div>
              <h2 className="text-lg font-bold text-foreground mt-1">
                Post-Outcome Evaluation: {selectedCrop} ({selectedYear})
              </h2>
              <p className="text-xs text-muted-foreground">
                Evaluating frozen forecasts against observed outcomes strictly when forecast origin precedes the harvest horizon.
              </p>
            </div>

            <div className="text-xs font-mono bg-muted/30 border border-border px-3 py-1.5 rounded-lg text-muted-foreground">
              Boundary: forecast_origin &lt; forecast_year
            </div>
          </div>

          {outcomes?.status === 'EVALUATION_UNAVAILABLE' ? (
            <div className="mt-6 p-6 rounded-xl bg-amber-50/60 border border-amber-200 text-center">
              <div className="w-12 h-12 rounded-full bg-amber-100 text-amber-800 mx-auto flex items-center justify-center mb-3">
                <Clock className="w-6 h-6" />
              </div>
              <h3 className="text-base font-bold text-amber-900">Evaluation Unavailable for Horizon {selectedYear}</h3>
              <p className="text-sm text-amber-800 max-w-xl mx-auto mt-1">
                {outcomes.reason}
              </p>
              <div className="mt-4 text-xs text-amber-700 font-mono">
                {outcomes.temporal_boundary_rule}
              </div>
            </div>
          ) : (
            <div className="mt-6 space-y-6">
              {/* Summary KPIs */}
              {outcomes?.summary && (
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
                  <div className="bg-muted/30 p-4 rounded-xl border border-border">
                    <span className="text-xs text-muted-foreground block">Walk-Forward MAE</span>
                    <span className="text-xl font-bold text-foreground">{outcomes.summary.mae}</span>
                    <span className="text-[10px] text-muted-foreground/60 block">kg/ha</span>
                  </div>
                  <div className="bg-muted/30 p-4 rounded-xl border border-border">
                    <span className="text-xs text-muted-foreground block">RMSE</span>
                    <span className="text-xl font-bold text-foreground">{outcomes.summary.rmse}</span>
                    <span className="text-[10px] text-muted-foreground/60 block">kg/ha</span>
                  </div>
                  <div className="bg-muted/30 p-4 rounded-xl border border-border">
                    <span className="text-xs text-muted-foreground block">Median Abs Error</span>
                    <span className="text-xl font-bold text-foreground">{outcomes.summary.median_absolute_error}</span>
                    <span className="text-[10px] text-muted-foreground/60 block">kg/ha</span>
                  </div>
                  <div className="bg-muted/30 p-4 rounded-xl border border-border">
                    <span className="text-xs text-muted-foreground block">Mean Signed Bias</span>
                    <span className={`text-xl font-bold ${outcomes.summary.mean_signed_bias >= 0 ? 'text-amber-700' : 'text-blue-700'}`}>
                      {outcomes.summary.mean_signed_bias > 0 ? `+${outcomes.summary.mean_signed_bias}` : outcomes.summary.mean_signed_bias}
                    </span>
                    <span className="text-[10px] text-muted-foreground/60 block">kg/ha</span>
                  </div>
                  <div className="bg-muted/30 p-4 rounded-xl border border-border">
                    <span className="text-xs text-muted-foreground block">Relative Error (MAPE)</span>
                    <span className="text-xl font-bold text-foreground">{outcomes.summary.mape || 'N/A'}%</span>
                    <span className="text-[10px] text-muted-foreground/60 block">mean % deviation</span>
                  </div>
                </div>
              )}

              {/* Verified Outcome Records Table */}
              <div className="overflow-x-auto border border-border rounded-xl">
                <table className="min-w-full divide-y divide-border text-sm">
                  <thead className="bg-muted/30">
                    <tr>
                      <th className="px-4 py-3 text-left font-semibold text-muted-foreground">Crop / Fold</th>
                      <th className="px-4 py-3 text-left font-semibold text-muted-foreground">Origin / Target</th>
                      <th className="px-4 py-3 text-right font-semibold text-muted-foreground">Predicted (kg/ha)</th>
                      <th className="px-4 py-3 text-right font-semibold text-muted-foreground">Observed (kg/ha)</th>
                      <th className="px-4 py-3 text-right font-semibold text-muted-foreground">Signed Error</th>
                      <th className="px-4 py-3 text-right font-semibold text-muted-foreground">Abs Error</th>
                      <th className="px-4 py-3 text-center font-semibold text-muted-foreground">Strategy</th>
                      <th className="px-4 py-3 text-center font-semibold text-muted-foreground">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border bg-card">
                    {outcomes?.records.map((rec, i) => (
                      <tr key={i} className="hover:bg-muted/30">
                        <td className="px-4 py-3 font-medium text-foreground">
                          {rec.crop}
                          <span className="text-xs text-muted-foreground block">{rec.district}</span>
                        </td>
                        <td className="px-4 py-3 text-xs text-muted-foreground font-mono">
                          {rec.forecast_origin} → {rec.forecast_year}
                        </td>
                        <td className="px-4 py-3 text-right font-semibold text-foreground">{rec.predicted_yield}</td>
                        <td className="px-4 py-3 text-right font-semibold text-foreground">{rec.observed_yield}</td>
                        <td className={`px-4 py-3 text-right font-mono font-medium ${rec.signed_error >= 0 ? 'text-amber-700' : 'text-blue-700'}`}>
                          {rec.signed_error > 0 ? `+${rec.signed_error}` : rec.signed_error}
                        </td>
                        <td className="px-4 py-3 text-right font-semibold text-foreground">{rec.absolute_error}</td>
                        <td className="px-4 py-3 text-center text-xs font-mono text-muted-foreground">{rec.strategy}</td>
                        <td className="px-4 py-3 text-center">
                          <span className="px-2 py-0.5 text-[11px] font-semibold rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                            {rec.evaluation_status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </section>

        {/* ----------------------------------------------------------------- */}
        {/* 5. Stratified Error Diagnostics & Directional Bias */}
        {/* ----------------------------------------------------------------- */}
        <section className="bg-white border border-border rounded-xl p-6 shadow-sm">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between pb-5 border-b border-border/60 gap-2">
            <div>
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 text-xs font-bold rounded bg-purple-50 text-purple-700 uppercase">
                  Decomposition
                </span>
                <span className="text-xs text-muted-foreground">Semantic Classification: POST_OUTCOME_EVALUATION</span>
              </div>
              <h2 className="text-lg font-bold text-foreground mt-1">
                Stratified Error Decomposition: {selectedCrop}
              </h2>
            </div>

            {/* Error Tabs */}
            <div className="flex items-center gap-1 bg-muted p-1 rounded-lg">
              <button
                onClick={() => setErrorTab('temporal')}
                className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors ${
                  errorTab === 'temporal' ? 'bg-white text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                Temporal Folds
              </button>
              <button
                onClick={() => setErrorTab('geographic')}
                className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors ${
                  errorTab === 'geographic' ? 'bg-white text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                District Slice
              </button>
              <button
                onClick={() => setErrorTab('regimes')}
                className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors ${
                  errorTab === 'regimes' ? 'bg-white text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                Yield Regimes
              </button>
            </div>
          </div>

          <div className="mt-6">
            {errorTab === 'temporal' && (
              <div className="overflow-x-auto border border-border rounded-xl">
                <table className="min-w-full divide-y divide-border text-sm">
                  <thead className="bg-muted/30">
                    <tr>
                      <th className="px-4 py-3 text-left font-semibold text-muted-foreground">Validation Year</th>
                      <th className="px-4 py-3 text-right font-semibold text-muted-foreground">Sample Size (N)</th>
                      <th className="px-4 py-3 text-right font-semibold text-muted-foreground">MAE (kg/ha)</th>
                      <th className="px-4 py-3 text-right font-semibold text-muted-foreground">RMSE (kg/ha)</th>
                      <th className="px-4 py-3 text-right font-semibold text-muted-foreground">Mean Bias</th>
                      <th className="px-4 py-3 text-right font-semibold text-muted-foreground">IQR Error (P25 - P75)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border bg-card">
                    {errors?.temporal_breakdown.map((t) => (
                      <tr key={t.year} className="hover:bg-muted/30">
                        <td className="px-4 py-3 font-semibold text-foreground">{t.year}</td>
                        <td className="px-4 py-3 text-right text-muted-foreground font-mono">{t.evaluated_forecasts}</td>
                        <td className="px-4 py-3 text-right font-bold text-foreground">{t.mae}</td>
                        <td className="px-4 py-3 text-right font-semibold text-foreground">{t.rmse}</td>
                        <td className={`px-4 py-3 text-right font-mono ${t.bias >= 0 ? 'text-amber-700' : 'text-blue-700'}`}>
                          {t.bias > 0 ? `+${t.bias}` : t.bias}
                        </td>
                        <td className="px-4 py-3 text-right text-xs text-muted-foreground font-mono">{t.p25_error} – {t.p75_error}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {errorTab === 'geographic' && (
              <div className="overflow-x-auto border border-border rounded-xl max-h-80 overflow-y-auto">
                <table className="min-w-full divide-y divide-border text-sm">
                  <thead className="bg-muted/30 sticky top-0">
                    <tr>
                      <th className="px-4 py-3 text-left font-semibold text-muted-foreground">State</th>
                      <th className="px-4 py-3 text-left font-semibold text-muted-foreground">District</th>
                      <th className="px-4 py-3 text-right font-semibold text-muted-foreground">Observations</th>
                      <th className="px-4 py-3 text-right font-semibold text-muted-foreground">MAE (kg/ha)</th>
                      <th className="px-4 py-3 text-right font-semibold text-muted-foreground">RMSE (kg/ha)</th>
                      <th className="px-4 py-3 text-right font-semibold text-muted-foreground">Signed Bias</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border bg-card">
                    {errors?.geographic_breakdown.map((g, i) => (
                      <tr key={i} className="hover:bg-muted/30">
                        <td className="px-4 py-2.5 text-foreground">{g.state}</td>
                        <td className="px-4 py-2.5 font-medium text-foreground">{g.district}</td>
                        <td className="px-4 py-2.5 text-right font-mono text-xs text-muted-foreground">{g.evaluated_forecasts}</td>
                        <td className="px-4 py-2.5 text-right font-bold text-foreground">{g.mae}</td>
                        <td className="px-4 py-2.5 text-right text-muted-foreground">{g.rmse}</td>
                        <td className={`px-4 py-2.5 text-right font-mono ${g.bias >= 0 ? 'text-amber-700' : 'text-blue-700'}`}>
                          {g.bias > 0 ? `+${g.bias}` : g.bias}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {errorTab === 'regimes' && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {errors?.regime_breakdown.map((r) => (
                  <div key={r.regime} className="bg-muted/30 border border-border rounded-xl p-4">
                    <div className="flex items-center justify-between pb-2 border-b border-border">
                      <span className="text-xs font-bold text-foreground">{r.regime} Yield Regime</span>
                      <span className="text-xs font-mono text-muted-foreground">N={r.sample_count}</span>
                    </div>
                    <div className="mt-3 space-y-1.5 text-xs text-muted-foreground">
                      <div className="flex justify-between">
                        <span>MAE:</span>
                        <strong className="text-foreground">{r.mae} kg/ha</strong>
                      </div>
                      <div className="flex justify-between">
                        <span>RMSE:</span>
                        <strong className="text-foreground">{r.rmse} kg/ha</strong>
                      </div>
                      <div className="flex justify-between">
                        <span>Signed Bias:</span>
                        <strong className={r.mean_signed_bias >= 0 ? 'text-amber-700' : 'text-blue-700'}>
                          {r.mean_signed_bias > 0 ? `+${r.mean_signed_bias}` : r.mean_signed_bias} kg/ha
                        </strong>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Directional Systematic Bias Card */}
          {activeCropBias && (
            <div className="mt-6 p-4 rounded-xl bg-muted/30 border border-border flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
              <div>
                <span className="text-xs font-semibold text-muted-foreground uppercase">Directional Bias Assessment</span>
                <h4 className="text-sm font-bold text-foreground mt-0.5">
                  Status: {activeCropBias.bias_status} (NME = {activeCropBias.normalized_mean_error_pct}%)
                </h4>
                <p className="text-xs text-muted-foreground mt-0.5">{activeCropBias.bias_description}</p>
              </div>
              <div className="text-xs text-muted-foreground bg-card px-3 py-1.5 rounded-lg border border-border">
                Rule: {activeCropBias.bias_threshold_rule}
              </div>
            </div>
          )}
        </section>

        {/* ----------------------------------------------------------------- */}
        {/* 6. Covariate Drift & Distribution Shift Tracking (PSI / KS) */}
        {/* ----------------------------------------------------------------- */}
        <section className="bg-white border border-border rounded-xl p-6 shadow-sm">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between pb-5 border-b border-border/60 gap-2">
            <div>
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 text-xs font-bold rounded bg-amber-50 text-amber-700 uppercase">
                  PSI & KS Diagnostics
                </span>
                <span className="text-xs text-muted-foreground">Semantic Classification: MONITORING</span>
              </div>
              <h2 className="text-lg font-bold text-foreground mt-1">
                Statistical Feature Drift & Dataset Stability
              </h2>
              <p className="text-xs text-muted-foreground">
                Population Stability Index (PSI) evaluating covariate distribution shifts between reference (2010–2015) and evaluation (2016–2017) sets.
              </p>
            </div>

            <div className="text-xs font-mono bg-muted/30 border border-border px-3 py-1.5 rounded-lg text-muted-foreground">
              Overall State: <strong>{drift?.overall_drift_status || 'STABLE'}</strong>
            </div>
          </div>

          <div className="mt-6 overflow-x-auto border border-border rounded-xl">
            <table className="min-w-full divide-y divide-border text-sm">
              <thead className="bg-muted/30">
                <tr>
                  <th className="px-4 py-3 text-left font-semibold text-muted-foreground">Monitored Feature</th>
                  <th className="px-4 py-3 text-right font-semibold text-muted-foreground">Metric</th>
                  <th className="px-4 py-3 text-right font-semibold text-muted-foreground">Observed Value</th>
                  <th className="px-4 py-3 text-right font-semibold text-muted-foreground">Threshold</th>
                  <th className="px-4 py-3 text-center font-semibold text-muted-foreground">Status</th>
                  <th className="px-4 py-3 text-left font-semibold text-muted-foreground">Windows (Ref vs Eval)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border bg-card">
                {drift?.features.map((f, i) => (
                  <tr key={i} className="hover:bg-muted/30">
                    <td className="px-4 py-3 font-medium text-foreground font-mono text-xs">{f.feature_name}</td>
                    <td className="px-4 py-3 text-right font-semibold text-muted-foreground">{f.metric}</td>
                    <td className="px-4 py-3 text-right font-bold text-foreground font-mono">{f.observed_value}</td>
                    <td className="px-4 py-3 text-right text-xs text-muted-foreground font-mono">{f.threshold}</td>
                    <td className="px-4 py-3 text-center">
                      <span className={`px-2.5 py-0.5 text-xs font-semibold rounded-full border ${
                        f.status === 'NO_DRIFT'
                          ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                          : f.status === 'MODERATE_DRIFT'
                          ? 'bg-amber-50 text-amber-700 border-amber-200'
                          : 'bg-red-50 text-red-700 border-red-200'
                      }`}>
                        {f.status}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-xs text-muted-foreground">
                      {f.reference_window} (N={f.reference_samples}) vs {f.evaluation_window} (N={f.evaluation_samples})
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* ----------------------------------------------------------------- */}
        {/* 7. Evidence-First Monitoring Alerts */}
        {/* ----------------------------------------------------------------- */}
        <section className="bg-white border border-border rounded-xl p-6 shadow-sm">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between pb-5 border-b border-border/60 gap-2">
            <div>
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 text-xs font-bold rounded bg-red-50 text-red-700 uppercase">
                  Evidence-First
                </span>
                <span className="text-xs text-muted-foreground">Semantic Classification: MONITORING</span>
              </div>
              <h2 className="text-lg font-bold text-foreground mt-1">
                Active Operational & Statistical Alerts
              </h2>
              <p className="text-xs text-muted-foreground">
                Signals generated strictly from verified feature drift, directional bias diagnostics, and operational request telemetry.
              </p>
            </div>

            <span className="text-xs font-mono bg-muted/30 border border-border px-3 py-1.5 rounded-lg text-muted-foreground">
              Total Active: {alerts?.total_alerts || 0}
            </span>
          </div>

          <div className="mt-6">
            {!alerts || alerts.active_alerts.length === 0 ? (
              <div className="p-8 text-center bg-muted/30 rounded-xl border border-border">
                <CheckCircle2 className="w-8 h-8 text-emerald-600 mx-auto mb-2" />
                <h4 className="text-sm font-bold text-foreground">No Active Monitoring Alerts</h4>
                <p className="text-xs text-muted-foreground mt-1">
                  All monitored feature distributions, forecast operations, and model biases are operating within certified tolerances.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {alerts.active_alerts.map((a) => (
                  <div key={a.alert_id} className="bg-muted/30 border border-border rounded-xl p-5 hover:border-border transition-colors">
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 pb-3 border-b border-border">
                      <div className="flex items-center gap-2">
                        <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold uppercase ${
                          a.severity === 'WARNING'
                            ? 'bg-amber-100 text-amber-800'
                            : a.severity === 'CRITICAL'
                            ? 'bg-red-100 text-red-800'
                            : 'bg-blue-100 text-blue-800'
                        }`}>
                          {a.severity}
                        </span>
                        <span className="text-xs font-mono text-muted-foreground">{a.alert_id}</span>
                        <span className="text-xs font-semibold text-foreground">[{a.category}] {a.signal}</span>
                      </div>
                      <span className="text-xs text-muted-foreground/60 font-mono">{a.timestamp}</span>
                    </div>

                    <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                      <div>
                        <span className="text-muted-foreground block font-semibold mb-1">Evidence & Observed Metric:</span>
                        <p className="text-foreground">{a.evidence}</p>
                        <div className="mt-2 flex gap-4 text-muted-foreground font-mono">
                          <span>Observed: <strong className="text-foreground">{a.observed_value}</strong></span>
                          {a.threshold && <span>Threshold: <strong className="text-foreground">{a.threshold}</strong></span>}
                          {a.sample_size && <span>Sample Size: <strong className="text-foreground">N={a.sample_size}</strong></span>}
                        </div>
                      </div>

                      <div>
                        <span className="text-muted-foreground block font-semibold mb-1">Recommended Platform Action:</span>
                        <p className="text-foreground">{a.recommended_action}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  )
}
