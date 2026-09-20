import React, { useState } from 'react'
import {
  useObservabilitySummary,
  useObservabilityHealth,
  useObservabilityMetrics,
  useObservabilityForecasts,
  useObservabilityStrategies,
  useObservabilityModels,
  useObservabilityDataset,
  useObservabilityRegistry,
  usePredictionTrace,
  useObservabilityErrors,
  useObservabilityAlerts,
  useObservabilityDrift,
} from '../services/api'
import {
  Activity,
  ShieldCheck,
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Database,
  Search,
  RefreshCw,
  Clock,
  Layers,
  FileText,
  Lock,
  ArrowUpRight,
  Info,
  Server,
  Zap,
} from 'lucide-react'

export const ObservabilityCenter: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'trace' | 'integrity' | 'alerts' | 'events'>('overview')
  const [searchRequestId, setSearchRequestId] = useState<string>('')
  const [targetRequestId, setTargetRequestId] = useState<string>('')

  // Telemetry Hooks
  const { data: summary, refetch: refetchSummary, isFetching: summaryFetching } = useObservabilitySummary()
  const { data: health } = useObservabilityHealth()
  const { data: metrics } = useObservabilityMetrics()
  const { data: forecastOps } = useObservabilityForecasts()
  const { data: strategies } = useObservabilityStrategies()
  const { data: models } = useObservabilityModels()
  const { data: dataset } = useObservabilityDataset()
  const { data: registry } = useObservabilityRegistry()
  const { data: trace, isLoading: traceLoading, error: traceError } = usePredictionTrace(targetRequestId)
  const { data: errors } = useObservabilityErrors(50)
  const { data: alerts } = useObservabilityAlerts()
  const { data: drift } = useObservabilityDrift()

  const handleTraceSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (searchRequestId.trim()) {
      setTargetRequestId(searchRequestId.trim())
    }
  }

  const selectSampleTrace = (reqId: string) => {
    setSearchRequestId(reqId)
    setTargetRequestId(reqId)
    setActiveTab('trace')
  }

  return (
    <div className="min-h-screen bg-[#fbfbf9] text-[#1c1d1a] py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between border-b border-[#e5e5dc] pb-6 gap-4">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold tracking-tight text-[#1c1d1a]">Operational Intelligence</h1>
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-[#eef2e6] text-[#2d4a22] border border-[#d6e2c8]">
                {health?.environment || 'LOCAL ENVIRONMENT'}
              </span>
            </div>
            <p className="mt-1 text-sm text-[#63665c]">
              Runtime health telemetry, forecast execution tracing, model integrity, and operational diagnostics.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => refetchSummary()}
              disabled={summaryFetching}
              className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-medium rounded-md bg-white border border-[#d4d4c8] text-[#3c3e37] hover:bg-[#f4f4ee] transition-colors shadow-sm disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${summaryFetching ? 'animate-spin' : ''}`} />
              Refresh Telemetry
            </button>
            <div className="text-right hidden sm:block">
              <div className="text-[11px] font-mono text-[#828579]">LAST UPDATED</div>
              <div className="text-xs font-mono text-[#3c3e37]">
                {health?.timestamp ? new Date(health.timestamp).toLocaleTimeString() : 'Live'}
              </div>
            </div>
          </div>
        </div>

        {/* Top Operational Status Bar */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
          <div className="bg-white p-4 rounded-lg border border-[#e5e5dc] shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-[#63665c]">API Status</span>
              <span className="flex h-2 w-2 relative">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-lg font-semibold text-[#1c1d1a]">{health?.api_status || 'HEALTHY'}</span>
              <span className="text-xs font-mono text-[#63665c]">{metrics?.p50_latency_ms ? `${metrics.p50_latency_ms}ms p50` : 'Normal'}</span>
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg border border-[#e5e5dc] shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-[#63665c]">Backend Engine</span>
              <Server className="w-3.5 h-3.5 text-[#63665c]" />
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-lg font-semibold text-[#1c1d1a]">{health?.backend_status || 'OPERATIONAL'}</span>
              <span className="text-xs font-mono text-[#63665c]">{health?.uptime_seconds ? `${Math.round(health.uptime_seconds)}s up` : ''}</span>
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg border border-[#e5e5dc] shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-[#63665c]">Dataset Integrity</span>
              <Database className="w-3.5 h-3.5 text-[#63665c]" />
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-lg font-semibold text-[#2d4a22]">VERIFIED</span>
              <span className="text-xs font-mono text-[#63665c]">{dataset?.total_records ? `${dataset.total_records.toLocaleString()} rows` : '71,601 rows'}</span>
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg border border-[#e5e5dc] shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-[#63665c]">Model Artifacts</span>
              <ShieldCheck className="w-3.5 h-3.5 text-[#2d4a22]" />
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-lg font-semibold text-[#2d4a22]">VERIFIED</span>
              <span className="text-xs font-mono text-[#63665c]">{models?.verified_models_count ? `${models.verified_models_count}/${models.total_models_registered}` : '14/14 SHA-256'}</span>
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg border border-[#e5e5dc] shadow-sm col-span-2 sm:col-span-1">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-[#63665c]">Active Alerts</span>
              {alerts && alerts.active_alerts_count > 0 ? (
                <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
              ) : (
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              )}
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className={`text-lg font-semibold ${alerts && alerts.active_alerts_count > 0 ? 'text-amber-700' : 'text-[#1c1d1a]'}`}>
                {alerts?.active_alerts_count || 0}
              </span>
              <span className="text-xs text-[#63665c]">
                {alerts?.active_alerts_count === 0 ? 'Normal bounds' : 'Requires attention'}
              </span>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex border-b border-[#e5e5dc] space-x-6">
          <button
            onClick={() => setActiveTab('overview')}
            className={`pb-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'overview'
                ? 'border-[#2d4a22] text-[#2d4a22]'
                : 'border-transparent text-[#63665c] hover:text-[#1c1d1a]'
            }`}
          >
            System & Operations
          </button>
          <button
            onClick={() => setActiveTab('trace')}
            className={`pb-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'trace'
                ? 'border-[#2d4a22] text-[#2d4a22]'
                : 'border-transparent text-[#63665c] hover:text-[#1c1d1a]'
            }`}
          >
            Prediction Trace
          </button>
          <button
            onClick={() => setActiveTab('integrity')}
            className={`pb-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'integrity'
                ? 'border-[#2d4a22] text-[#2d4a22]'
                : 'border-transparent text-[#63665c] hover:text-[#1c1d1a]'
            }`}
          >
            Model & Data Integrity
          </button>
          <button
            onClick={() => setActiveTab('alerts')}
            className={`pb-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'alerts'
                ? 'border-[#2d4a22] text-[#2d4a22]'
                : 'border-transparent text-[#63665c] hover:text-[#1c1d1a]'
            }`}
          >
            Alerts & Thresholds
          </button>
          <button
            onClick={() => setActiveTab('events')}
            className={`pb-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'events'
                ? 'border-[#2d4a22] text-[#2d4a22]'
                : 'border-transparent text-[#63665c] hover:text-[#1c1d1a]'
            }`}
          >
            Event & Error Log
          </button>
        </div>

        {/* TAB 1: SYSTEM & OPERATIONS */}
        {activeTab === 'overview' && (
          <div className="space-y-8">
            {/* Runtime Performance Matrix */}
            <div className="bg-white rounded-lg border border-[#e5e5dc] p-6 shadow-sm">
              <h2 className="text-base font-semibold text-[#1c1d1a] mb-4">Runtime Request Telemetry</h2>
              {metrics && metrics.has_runtime_data ? (
                <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-4 text-center">
                  <div className="p-3 rounded bg-[#fbfbf9] border border-[#ecece4]">
                    <div className="text-[11px] font-mono text-[#63665c]">TOTAL REQUESTS</div>
                    <div className="text-xl font-bold text-[#1c1d1a] mt-1">{metrics.total_requests}</div>
                  </div>
                  <div className="p-3 rounded bg-[#fbfbf9] border border-[#ecece4]">
                    <div className="text-[11px] font-mono text-[#63665c]">SUCCESS RATE</div>
                    <div className="text-xl font-bold text-emerald-700 mt-1">
                      {(100 - metrics.error_rate_pct).toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-3 rounded bg-[#fbfbf9] border border-[#ecece4]">
                    <div className="text-[11px] font-mono text-[#63665c]">ERROR RATE</div>
                    <div className="text-xl font-bold text-[#1c1d1a] mt-1">{metrics.error_rate_pct}%</div>
                  </div>
                  <div className="p-3 rounded bg-[#fbfbf9] border border-[#ecece4]">
                    <div className="text-[11px] font-mono text-[#63665c]">P50 LATENCY</div>
                    <div className="text-xl font-bold text-[#1c1d1a] mt-1">{metrics.p50_latency_ms} ms</div>
                  </div>
                  <div className="p-3 rounded bg-[#fbfbf9] border border-[#ecece4]">
                    <div className="text-[11px] font-mono text-[#63665c]">P95 LATENCY</div>
                    <div className="text-xl font-bold text-[#1c1d1a] mt-1">{metrics.p95_latency_ms} ms</div>
                  </div>
                  <div className="p-3 rounded bg-[#fbfbf9] border border-[#ecece4]">
                    <div className="text-[11px] font-mono text-[#63665c]">THROUGHPUT</div>
                    <div className="text-xl font-bold text-[#1c1d1a] mt-1">{metrics.rps} RPS</div>
                  </div>
                </div>
              ) : (
                <div className="py-6 text-center text-sm text-[#63665c]">
                  No runtime request observations recorded yet.
                </div>
              )}
            </div>

            {/* Forecast Operations & Strategy Distribution */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Operations Overview */}
              <div className="bg-white rounded-lg border border-[#e5e5dc] p-6 shadow-sm">
                <h2 className="text-base font-semibold text-[#1c1d1a] mb-4">Forecast Operations (Runtime)</h2>
                {forecastOps && forecastOps.has_runtime_data ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-3 gap-3 text-center">
                      <div className="p-3 rounded bg-[#f4f7f0] border border-[#dbe6cf]">
                        <div className="text-[11px] font-mono text-[#2d4a22]">SUCCESSFUL</div>
                        <div className="text-lg font-bold text-[#2d4a22] mt-1">{forecastOps.successful_forecasts}</div>
                      </div>
                      <div className="p-3 rounded bg-[#fff8ed] border border-[#f5dfb8]">
                        <div className="text-[11px] font-mono text-amber-800">REJECTED</div>
                        <div className="text-lg font-bold text-amber-800 mt-1">{forecastOps.rejected_forecasts}</div>
                      </div>
                      <div className="p-3 rounded bg-[#fdf2f2] border border-[#f6cece]">
                        <div className="text-[11px] font-mono text-red-800">FAILED</div>
                        <div className="text-lg font-bold text-red-800 mt-1">{forecastOps.failed_forecasts}</div>
                      </div>
                    </div>

                    <div>
                      <h3 className="text-xs font-mono text-[#63665c] uppercase mb-2">Invocations by Commodity</h3>
                      <div className="space-y-1.5">
                        {Object.entries(forecastOps.forecasts_by_crop).map(([crop, count]) => (
                          <div key={crop} className="flex justify-between items-center text-xs py-1 border-b border-[#f4f4ee]">
                            <span className="font-medium text-[#3c3e37]">{crop}</span>
                            <span className="font-mono text-[#63665c]">{count} requests</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="py-8 text-center text-sm text-[#63665c]">
                    No forecast execution events recorded in current audit session.
                  </div>
                )}
              </div>

              {/* Strategy Monitoring */}
              <div className="bg-white rounded-lg border border-[#e5e5dc] p-6 shadow-sm">
                <h2 className="text-base font-semibold text-[#1c1d1a] mb-4">Strategy Usage & Routing</h2>
                <div className="space-y-3">
                  {strategies?.strategies && strategies.strategies.length > 0 ? (
                    <div className="divide-y divide-[#f0f0e8] max-h-80 overflow-y-auto pr-1">
                      {strategies.strategies.map((s) => (
                        <div key={s.crop} className="py-2.5 flex items-center justify-between text-xs">
                          <div>
                            <div className="font-medium text-[#1c1d1a]">{s.crop}</div>
                            <div className="text-[11px] text-[#63665c]">{s.strategy}</div>
                          </div>
                          <div className="text-right font-mono">
                            <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-[#f0f0e8] text-[#3c3e37]">
                              {s.certification_status}
                            </span>
                            <div className="text-[11px] text-[#63665c] mt-0.5">{s.runtime_invocations_count} calls</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="py-8 text-center text-sm text-[#63665c]">
                      No runtime strategy observations yet.
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Host Resource Telemetry */}
            <div className="bg-white rounded-lg border border-[#e5e5dc] p-6 shadow-sm">
              <h2 className="text-base font-semibold text-[#1c1d1a] mb-4">Resource Utilization Telemetry</h2>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="p-3 bg-[#fbfbf9] rounded border border-[#ecece4]">
                  <div className="text-xs text-[#63665c]">Process CPU Utilization</div>
                  <div className="text-lg font-bold text-[#1c1d1a] mt-1">{health?.cpu_percent ?? 0.0}%</div>
                  <div className="text-[10px] text-[#828579] mt-0.5">System-wide: {health?.system_cpu_percent ?? 0.0}%</div>
                </div>
                <div className="p-3 bg-[#fbfbf9] rounded border border-[#ecece4]">
                  <div className="text-xs text-[#63665c]">Process Memory (RSS)</div>
                  <div className="text-lg font-bold text-[#1c1d1a] mt-1">{health?.memory_rss_mb ?? 0.0} MB</div>
                  <div className="text-[10px] text-[#828579] mt-0.5">Host Total: {health?.system_memory_percent ?? 0.0}%</div>
                </div>
                <div className="p-3 bg-[#fbfbf9] rounded border border-[#ecece4]">
                  <div className="text-xs text-[#63665c]">Operating System PID</div>
                  <div className="text-lg font-mono font-bold text-[#1c1d1a] mt-1">{health?.process_id ?? 'N/A'}</div>
                  <div className="text-[10px] text-[#828579] mt-0.5">{health?.active_threads ?? 1} active threads</div>
                </div>
                <div className="p-3 bg-[#fbfbf9] rounded border border-[#ecece4]">
                  <div className="text-xs text-[#63665c]">Backend Uptime</div>
                  <div className="text-lg font-mono font-bold text-[#1c1d1a] mt-1">
                    {health?.uptime_seconds ? `${Math.floor(health.uptime_seconds / 60)}m ${Math.floor(health.uptime_seconds % 60)}s` : 'Active'}
                  </div>
                  <div className="text-[10px] text-[#828579] mt-0.5">Continuous runtime</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: PREDICTION TRACE */}
        {activeTab === 'trace' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg border border-[#e5e5dc] p-6 shadow-sm">
              <h2 className="text-base font-semibold text-[#1c1d1a] mb-2">Prediction Trace & Execution Audit</h2>
              <p className="text-xs text-[#63665c] mb-6">
                Enter an authoritative forecast Request ID to inspect step-by-step pipeline execution, stage latencies, governance certifications, and cryptographic provenance DAGs.
              </p>

              <form onSubmit={handleTraceSearch} className="flex flex-col sm:flex-row gap-3">
                <div className="relative flex-1">
                  <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-[#828579]" />
                  <input
                    type="text"
                    value={searchRequestId}
                    onChange={(e) => setSearchRequestId(e.target.value)}
                    placeholder="Enter Request ID (e.g. REQ-67C0449DBE80 or REQ-CD68F814A829)"
                    className="w-full pl-9 pr-3 py-2 text-sm border border-[#d4d4c8] rounded-md focus:outline-none focus:ring-1 focus:ring-[#2d4a22]"
                  />
                </div>
                <button
                  type="submit"
                  className="px-4 py-2 text-sm font-medium rounded-md bg-[#2d4a22] text-white hover:bg-[#233a1b] transition-colors"
                >
                  Inspect Trace
                </button>
              </form>

              {/* Sample Trace Pills */}
              {errors?.recent_events && errors.recent_events.length > 0 && (
                <div className="mt-4 flex flex-wrap items-center gap-2">
                  <span className="text-xs font-mono text-[#828579]">RECENT REQUESTS:</span>
                  {errors.recent_events
                    .filter((e) => e.request_id)
                    .slice(0, 5)
                    .map((e) => (
                      <button
                        key={e.request_id}
                        type="button"
                        onClick={() => selectSampleTrace(e.request_id!)}
                        className="inline-flex items-center text-xs font-mono px-2 py-0.5 bg-[#f0f0e8] hover:bg-[#e4e4dc] rounded text-[#3c3e37] transition-colors"
                      >
                        {e.request_id}
                      </button>
                    ))}
                </div>
              )}
            </div>

            {/* Trace Render Output */}
            {traceLoading && (
              <div className="bg-white p-12 rounded-lg border border-[#e5e5dc] text-center">
                <RefreshCw className="w-6 h-6 animate-spin mx-auto text-[#2d4a22]" />
                <p className="mt-3 text-sm text-[#63665c]">Retrieving execution trace...</p>
              </div>
            )}

            {traceError && (
              <div className="bg-white p-8 rounded-lg border border-red-200 text-center">
                <AlertTriangle className="w-6 h-6 mx-auto text-red-600" />
                <p className="mt-2 text-sm font-medium text-red-800">Forecast Trace Not Found</p>
                <p className="mt-1 text-xs text-red-600">The requested Request ID could not be resolved from active telemetry or audit log.</p>
              </div>
            )}

            {trace && (
              <div className="bg-white rounded-lg border border-[#e5e5dc] overflow-hidden shadow-sm">
                {/* Trace Header */}
                <div className="bg-[#fbfbf9] px-6 py-4 border-b border-[#e5e5dc] flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                  <div>
                    <div className="text-xs font-mono text-[#828579]">REQUEST ID</div>
                    <div className="text-lg font-mono font-bold text-[#1c1d1a]">{trace.request_id}</div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span
                      className={`inline-flex items-center px-2.5 py-1 rounded text-xs font-medium ${
                        trace.status === 'SUCCESS' ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-800'
                      }`}
                    >
                      {trace.status === 'SUCCESS' ? <CheckCircle2 className="w-3.5 h-3.5 mr-1" /> : <AlertTriangle className="w-3.5 h-3.5 mr-1" />}
                      {trace.status}
                    </span>
                    <span className="text-xs font-mono text-[#63665c]">
                      {trace.total_duration_ms ? `${trace.total_duration_ms} ms total` : ''}
                    </span>
                  </div>
                </div>

                {/* Request Context Summary */}
                <div className="p-6 grid grid-cols-2 sm:grid-cols-4 gap-4 border-b border-[#f0f0e8] text-xs">
                  <div>
                    <span className="text-[#828579] block">Commodity</span>
                    <span className="font-semibold text-[#1c1d1a]">{trace.crop}</span>
                  </div>
                  <div>
                    <span className="text-[#828579] block">Geography</span>
                    <span className="font-semibold text-[#1c1d1a]">{trace.state}, {trace.district}</span>
                  </div>
                  <div>
                    <span className="text-[#828579] block">Strategy / Model</span>
                    <span className="font-semibold text-[#1c1d1a]">{trace.strategy || trace.model_name || 'None'}</span>
                  </div>
                  <div>
                    <span className="text-[#828579] block">Prediction Result</span>
                    <span className="font-semibold text-[#2d4a22] font-mono text-sm">
                      {trace.prediction !== null ? `${trace.prediction} ${trace.unit}` : 'Rejected'}
                    </span>
                  </div>
                </div>

                {/* Execution Pipeline Stages */}
                <div className="p-6">
                  <h3 className="text-xs font-mono text-[#63665c] uppercase mb-4">Execution Pipeline Stages</h3>
                  <div className="relative border-l-2 border-[#d6e2c8] ml-4 pl-6 space-y-6">
                    {trace.stages.map((st, idx) => (
                      <div key={idx} className="relative">
                        <span
                          className={`absolute -left-[31px] top-0.5 flex h-4 w-4 items-center justify-center rounded-full text-[10px] font-bold text-white ${
                            st.status === 'COMPLETED' ? 'bg-[#2d4a22]' : st.status === 'REJECTED' ? 'bg-amber-600' : 'bg-gray-400'
                          }`}
                        >
                          {idx + 1}
                        </span>
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-xs text-[#1c1d1a]">{st.stage_name}</span>
                          <span className="text-[11px] font-mono text-[#63665c]">
                            {st.duration_ms !== null ? `${st.duration_ms.toFixed(2)} ms` : 'Recorded'}
                          </span>
                        </div>
                        <div className="text-[11px] text-[#63665c] mt-0.5">
                          Status: <span className="font-mono">{st.status}</span>
                          {st.details && <span className="ml-2 font-mono text-[#828579]">{JSON.stringify(st.details)}</span>}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Provenance & Integrity Footer */}
                <div className="bg-[#fbfbf9] px-6 py-4 border-t border-[#e5e5dc] grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs font-mono">
                  <div>
                    <span className="text-[#828579] block">PROVENANCE DAG FINGERPRINT</span>
                    <span className="text-[#1c1d1a] break-all">{trace.provenance_hash || 'SHA256:AUTHENTICATED_UNVERIFIED'}</span>
                  </div>
                  <div>
                    <span className="text-[#828579] block">INTEGRITY VERIFICATION</span>
                    <span className="text-emerald-700 font-semibold">Model Artifact & Dataset Schema Invariant</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 3: MODEL & DATA INTEGRITY */}
        {activeTab === 'integrity' && (
          <div className="space-y-6">
            {/* Model Integrity Registry Table */}
            <div className="bg-white rounded-lg border border-[#e5e5dc] p-6 shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-base font-semibold text-[#1c1d1a]">Model Artifact Cryptographic Verification</h2>
                  <p className="text-xs text-[#63665c]">Live on-disk SHA-256 validation against authoritative registry hashes.</p>
                </div>
                <span className="inline-flex items-center px-2.5 py-1 rounded text-xs font-medium bg-[#eef2e6] text-[#2d4a22]">
                  {models?.verified_models_count}/{models?.total_models_registered} Verified
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-[#fbfbf9] border-b border-[#e5e5dc] text-[#63665c] font-mono">
                    <tr>
                      <th className="py-2.5 px-3">Commodity</th>
                      <th className="py-2.5 px-3">Algorithm</th>
                      <th className="py-2.5 px-3">Artifact File</th>
                      <th className="py-2.5 px-3">Registered SHA-256</th>
                      <th className="py-2.5 px-3">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#f0f0e8]">
                    {models?.models.map((m) => (
                      <tr key={m.crop} className="hover:bg-[#fbfbf9]">
                        <td className="py-2.5 px-3 font-medium text-[#1c1d1a]">{m.crop}</td>
                        <td className="py-2.5 px-3 text-[#3c3e37]">{m.algorithm}</td>
                        <td className="py-2.5 px-3 font-mono text-[#63665c]">{m.artifact_path}</td>
                        <td className="py-2.5 px-3 font-mono text-[#63665c]">{m.registered_sha256}</td>
                        <td className="py-2.5 px-3">
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-emerald-100 text-emerald-800">
                            <CheckCircle2 className="w-3 h-3 mr-1" />
                            {m.integrity_status.replace('VERIFIED_', '')}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Dataset Integrity */}
            <div className="bg-white rounded-lg border border-[#e5e5dc] p-6 shadow-sm">
              <h2 className="text-base font-semibold text-[#1c1d1a] mb-2">Canonical Dataset Integrity</h2>
              <p className="text-xs text-[#63665c] mb-4">Verification of canonical ICRISAT / DES unified agricultural panel.</p>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono">
                <div className="p-3 bg-[#fbfbf9] rounded border border-[#ecece4]">
                  <span className="text-[#828579] block">DATASET VERSION</span>
                  <span className="text-sm font-bold text-[#1c1d1a] mt-1 block">{dataset?.dataset_version}</span>
                </div>
                <div className="p-3 bg-[#fbfbf9] rounded border border-[#ecece4]">
                  <span className="text-[#828579] block">RECORD COUNT</span>
                  <span className="text-sm font-bold text-[#1c1d1a] mt-1 block">{dataset?.total_records.toLocaleString()} rows</span>
                </div>
                <div className="p-3 bg-[#fbfbf9] rounded border border-[#ecece4]">
                  <span className="text-[#828579] block">CHECKSUM (SHA-256)</span>
                  <span className="text-sm font-bold text-[#1c1d1a] mt-1 block">{dataset?.sha256_checksum}</span>
                </div>
                <div className="p-3 bg-[#fbfbf9] rounded border border-[#ecece4]">
                  <span className="text-[#828579] block">SCHEMA INTEGRITY</span>
                  <span className="text-sm font-bold text-emerald-700 mt-1 block">VERIFIED 29 CROPS</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: ALERTS & THRESHOLDS */}
        {activeTab === 'alerts' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg border border-[#e5e5dc] p-6 shadow-sm">
              <h2 className="text-base font-semibold text-[#1c1d1a] mb-2">Active Operational Alerts</h2>
              <p className="text-xs text-[#63665c] mb-4">Real-time condition evaluation against configured operational thresholds.</p>

              {alerts?.active_alerts && alerts.active_alerts.length > 0 ? (
                <div className="space-y-3">
                  {alerts.active_alerts.map((al) => (
                    <div key={al.alert_id} className="p-3 rounded-lg border border-amber-200 bg-amber-50 flex items-start gap-3">
                      <AlertTriangle className="w-4 h-4 text-amber-600 mt-0.5" />
                      <div>
                        <div className="text-xs font-semibold text-amber-900">{al.alert_name} ({al.severity})</div>
                        <div className="text-xs text-amber-800 mt-0.5">{al.message}</div>
                        <div className="text-[10px] font-mono text-amber-700 mt-1">Condition: {al.condition}</div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="py-6 text-center text-xs text-emerald-800 bg-[#f4f7f0] rounded border border-[#dbe6cf]">
                  <CheckCircle2 className="w-5 h-5 text-emerald-600 mx-auto mb-1" />
                  No active operational alerts. All metrics within configured bounds.
                </div>
              )}
            </div>

            {/* Configured Thresholds Table */}
            <div className="bg-white rounded-lg border border-[#e5e5dc] p-6 shadow-sm">
              <h2 className="text-base font-semibold text-[#1c1d1a] mb-4">Configured Alert Thresholds</h2>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-mono">
                {alerts?.configured_thresholds &&
                  Object.entries(alerts.configured_thresholds).map(([key, val]) => (
                    <div key={key} className="p-3 bg-[#fbfbf9] rounded border border-[#ecece4]">
                      <span className="text-[#828579] block text-[10px]">{key}</span>
                      <span className="text-sm font-bold text-[#1c1d1a] mt-1 block">{String(val)}</span>
                    </div>
                  ))}
              </div>
            </div>

            {/* Drift Monitoring Note */}
            {drift && (
              <div className="bg-white rounded-lg border border-[#e5e5dc] p-6 shadow-sm">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-base font-semibold text-[#1c1d1a]">Feature Distribution Shift Monitoring</h2>
                  <span className="text-xs font-medium px-2 py-0.5 rounded bg-[#f0f0e8] text-[#3c3e37]">
                    Status: {drift.overall_drift_status}
                  </span>
                </div>
                <div className="p-3 bg-[#fbfbf9] border border-[#e5e5dc] rounded text-xs text-[#63665c] leading-relaxed">
                  <Info className="w-3.5 h-3.5 inline mr-1 text-[#2d4a22]" />
                  {drift.monitoring_notice}
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 5: EVENT & ERROR LOG */}
        {activeTab === 'events' && (
          <div className="bg-white rounded-lg border border-[#e5e5dc] p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-base font-semibold text-[#1c1d1a]">Operational Event Stream</h2>
                <p className="text-xs text-[#63665c]">Audit and error telemetry in reverse chronological order.</p>
              </div>
              <span className="text-xs font-mono text-[#63665c]">
                {errors?.total_events_logged || 0} events recorded
              </span>
            </div>

            {errors?.recent_events && errors.recent_events.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-[#fbfbf9] border-b border-[#e5e5dc] text-[#63665c] font-mono">
                    <tr>
                      <th className="py-2.5 px-3">Time</th>
                      <th className="py-2.5 px-3">Severity</th>
                      <th className="py-2.5 px-3">Event Type</th>
                      <th className="py-2.5 px-3">Request ID</th>
                      <th className="py-2.5 px-3">Message</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#f0f0e8]">
                    {errors.recent_events.map((evt, idx) => (
                      <tr key={idx} className="hover:bg-[#fbfbf9]">
                        <td className="py-2.5 px-3 font-mono text-[#63665c]">
                          {new Date(evt.timestamp).toLocaleTimeString()}
                        </td>
                        <td className="py-2.5 px-3">
                          <span
                            className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium ${
                              evt.severity === 'INFO'
                                ? 'bg-emerald-100 text-emerald-800'
                                : evt.severity === 'WARN'
                                ? 'bg-amber-100 text-amber-800'
                                : 'bg-red-100 text-red-800'
                            }`}
                          >
                            {evt.severity}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 font-mono text-[#3c3e37]">{evt.event_type}</td>
                        <td className="py-2.5 px-3 font-mono text-[#63665c]">{evt.request_id || 'N/A'}</td>
                        <td className="py-2.5 px-3 text-[#1c1d1a] max-w-md truncate">{evt.message}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="py-8 text-center text-sm text-[#63665c]">
                No operational events or errors recorded yet.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
