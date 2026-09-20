import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  useMonitoringOverview,
  useMonitoringTimeline,
  useMonitoringAlerts,
  useWarningMap,
  useMonitoringHealth,
} from '../services/api'
import { MonitoringOverview } from '../components/monitoring/MonitoringOverview'
import { WarningMap } from '../components/monitoring/WarningMap'
import { AlertFeed } from '../components/monitoring/AlertFeed'
import { RegionalRiskRanking } from '../components/monitoring/RegionalRiskRanking'
import { TemporalTrendChart } from '../components/monitoring/TemporalTrendChart'
import { SignalTimeline } from '../components/monitoring/SignalTimeline'
import { MonitoringHealth } from '../components/monitoring/MonitoringHealth'
import { WarningBacktestPanel } from '../components/monitoring/WarningBacktestPanel'
import { AlertDetailDrawer } from '../components/monitoring/AlertDetailDrawer'
import {
  Bell,
  Sparkles,
  ExternalLink
} from 'lucide-react'
import type { Alert } from '../types/monitoring'

export const AgriculturalMonitoring: React.FC = () => {
  const navigate = useNavigate()

  const [selectedState, setSelectedState] = useState('Punjab')
  const [selectedDistrict, setSelectedDistrict] = useState<string | undefined>(undefined)
  const [selectedMetric, setSelectedMetric] = useState('yield')
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null)

  // API Data Hooks
  const { data: overview, isLoading: overviewLoading } = useMonitoringOverview()
  const { data: warningMapData, isLoading: mapLoading } = useWarningMap()
  const { data: alerts, isLoading: alertsLoading } = useMonitoringAlerts({
    state: selectedState === 'ALL' ? undefined : selectedState,
    limit: 100,
  })
  const { data: timelineData, isLoading: timelineLoading } = useMonitoringTimeline(
    selectedState === 'ALL' ? 'Punjab' : selectedState,
    selectedDistrict,
    selectedMetric
  )
  const { data: healthData, isLoading: healthLoading } = useMonitoringHealth()

  const handleSelectAlert = (alert: Alert) => {
    setSelectedAlert(alert)
    setSelectedState(alert.state)
    setSelectedDistrict(alert.district)
  }

  const handleStateClick = (state: string) => {
    setSelectedState(state)
    setSelectedDistrict(undefined)
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-6 bg-gradient-to-r from-slate-900 via-slate-900/90 to-emerald-950/40 border border-slate-800 rounded-2xl shadow-xl">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <span className="p-2 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400">
              <Bell className="w-6 h-6" />
            </span>
            <h1 className="text-2xl font-extrabold text-slate-100 tracking-tight">
              Agricultural Monitoring & Early Warning Command Center
            </h1>
          </div>
          <p className="text-xs text-slate-400 max-w-3xl leading-relaxed">
            Continuously tracks empirical risk signals, detects multi-period structural breaks, generates evidence-backed early warnings, and prioritizes agricultural regions requiring agronomic review.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate('/explainability')}
            className="px-3.5 py-2 text-xs font-semibold rounded-xl bg-emerald-950/40 hover:bg-emerald-900/50 border border-emerald-500/40 text-emerald-300 flex items-center gap-1.5 transition-all shadow"
          >
            <Sparkles className="w-4 h-4" />
            Explain Alerts (XAI)
          </button>
          <button
            onClick={() => navigate('/scenario')}
            className="px-3.5 py-2 text-xs font-semibold rounded-xl bg-purple-950/40 hover:bg-purple-900/50 border border-purple-500/40 text-purple-300 flex items-center gap-1.5 transition-all shadow"
          >
            <Sparkles className="w-4 h-4" />
            Run Scenario Stress Test
          </button>
          <button
            onClick={() => navigate('/copilot')}
            className="px-3.5 py-2 text-xs font-semibold rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white flex items-center gap-1.5 transition-all shadow"
          >
            Ask AI Copilot
            <ExternalLink className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* 1. Top-Level Summary KPIs */}
      <MonitoringOverview data={overview} isLoading={overviewLoading} />

      {/* 2. India Warning Map */}
      <WarningMap
        statesData={warningMapData ?? []}
        isLoading={mapLoading}
        selectedState={selectedState}
        onSelectState={handleStateClick}
      />

      {/* 3. Alert Stream & Regional Priority Rankings */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7">
          <AlertFeed
            alerts={alerts ?? []}
            isLoading={alertsLoading}
            onSelectAlert={handleSelectAlert}
            selectedAlertId={selectedAlert?.alert_id}
          />
        </div>
        <div className="lg:col-span-5">
          <RegionalRiskRanking
            alerts={alerts ?? []}
            isLoading={alertsLoading}
            onSelectAlert={handleSelectAlert}
          />
        </div>
      </div>

      {/* 4. Temporal Dynamics & Signal Timeline */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-8">
          <TemporalTrendChart
            data={timelineData}
            isLoading={timelineLoading}
            selectedMetric={selectedMetric}
            onMetricChange={setSelectedMetric}
          />
        </div>
        <div className="lg:col-span-4">
          <SignalTimeline alert={selectedAlert} />
        </div>
      </div>

      {/* 5. Warning Rule Historical Backtesting */}
      <WarningBacktestPanel />

      {/* 6. 5-Pillar Monitoring Health & Governance */}
      <MonitoringHealth data={healthData} isLoading={healthLoading} />

      {/* Alert Detail Drawer Modal */}
      {selectedAlert && (
        <AlertDetailDrawer
          alert={selectedAlert}
          onClose={() => setSelectedAlert(null)}
          onExploreScenario={(st) => navigate('/scenario')}
          onViewGeospatial={(st) => navigate('/geospatial')}
        />
      )}
    </div>
  )
}
