import React, { useState, useMemo } from 'react'
import {
  Search,
  AlertTriangle,
  ShieldAlert,
  ExternalLink,
  CheckCircle2
} from 'lucide-react'
import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { Input } from '../ui/Input'
import { Select } from '../ui/Select'
import type { Alert } from '../../types/monitoring'

interface Props {
  alerts: Alert[]
  isLoading?: boolean
  onSelectAlert: (alert: Alert) => void
  selectedAlertId?: string
}

export const AlertFeed: React.FC<Props> = ({
  alerts,
  isLoading,
  onSelectAlert,
  selectedAlertId,
}) => {
  const [search, setSearch] = useState('')
  const [stateFilter, setStateFilter] = useState('ALL')
  const [severityFilter, setSeverityFilter] = useState('ALL')

  // Extract unique states for filter
  const uniqueStates = useMemo(() => {
    const states = Array.from(new Set(alerts.map((a) => a.state))).sort()
    return ['ALL', ...states]
  }, [alerts])

  // Filtered alerts
  const filteredAlerts = useMemo(() => {
    return alerts.filter((a) => {
      const matchesSearch =
        search === '' ||
        a.location.toLowerCase().includes(search.toLowerCase()) ||
        a.dominant_signal.toLowerCase().includes(search.toLowerCase()) ||
        a.alert_id.toLowerCase().includes(search.toLowerCase())

      const matchesState = stateFilter === 'ALL' || a.state === stateFilter
      const matchesSeverity = severityFilter === 'ALL' || a.severity === severityFilter

      return matchesSearch && matchesState && matchesSeverity
    })
  }, [alerts, search, stateFilter, severityFilter])

  const getSeverityBadgeVariant = (severity: string) => {
    switch (severity.toUpperCase()) {
      case 'CRITICAL':
        return 'red' as const
      case 'HIGH':
        return 'rose' as const
      case 'ELEVATED':
        return 'amber' as const
      case 'WATCH':
        return 'amber' as const
      default:
        return 'blue' as const
    }
  }

  return (
    <Card className="p-5 border border-slate-800 bg-slate-900/60 backdrop-blur space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div>
          <h3 className="text-base font-semibold text-slate-100 flex items-center gap-2">
            <ShieldAlert className="w-5 h-5 text-amber-400" />
            Prioritized Alert Stream
          </h3>
          <p className="text-xs text-slate-400">
            {filteredAlerts.length} active warning signals ranked by multi-signal risk severity
          </p>
        </div>

        {/* Filters */}
        <div className="flex items-center gap-2">
          <Input
            placeholder="Search district, signal, ID..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="h-8 text-xs w-44 bg-slate-950/60 border-slate-700"
          />
          <Select
            value={stateFilter}
            onChange={(e) => setStateFilter(e.target.value)}
            className="h-8 text-xs w-32 bg-slate-950/60 border-slate-700"
          >
            {uniqueStates.map((s) => (
              <option key={s} value={s}>
                {s === 'ALL' ? 'All States' : s}
              </option>
            ))}
          </Select>
          <Select
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
            className="h-8 text-xs w-28 bg-slate-950/60 border-slate-700"
          >
            <option value="ALL">All Tiers</option>
            <option value="CRITICAL">Critical</option>
            <option value="HIGH">High</option>
            <option value="ELEVATED">Elevated</option>
            <option value="WATCH">Watch</option>
            <option value="INFO">Info</option>
          </Select>
        </div>
      </div>

      {/* Alert Items List */}
      <div className="space-y-2.5 max-h-[440px] overflow-y-auto pr-1">
        {isLoading ? (
          <div className="space-y-3">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="h-20 bg-slate-800/40 rounded-lg animate-pulse" />
            ))}
          </div>
        ) : filteredAlerts.length === 0 ? (
          <div className="py-12 text-center text-slate-500 text-xs">
            No alerts matching the selected filters.
          </div>
        ) : (
          filteredAlerts.map((alert) => {
            const isSelected = selectedAlertId === alert.alert_id
            return (
              <div
                key={alert.alert_id}
                onClick={() => onSelectAlert(alert)}
                className={`p-3.5 rounded-xl border transition-all cursor-pointer flex flex-col sm:flex-row sm:items-center justify-between gap-3 ${
                  isSelected
                    ? 'bg-emerald-950/30 border-emerald-500/60 shadow-lg shadow-emerald-950/40'
                    : 'bg-slate-950/40 border-slate-800/80 hover:bg-slate-800/40 hover:border-slate-700'
                }`}
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-xs font-mono font-semibold text-slate-300">
                      {alert.alert_id}
                    </span>
                    <Badge variant={getSeverityBadgeVariant(alert.severity)} className="text-[10px] px-1.5 py-0.5">
                      {alert.severity}
                    </Badge>
                    <span className="text-xs font-semibold text-slate-200">
                      {alert.location}
                    </span>
                    {alert.priority_rank && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
                        Rank #{alert.priority_rank}
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-slate-300 flex items-center gap-1.5 font-medium">
                    <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                    {alert.dominant_signal}
                  </div>
                  <div className="text-[11px] text-slate-500">
                    {alert.signal_count} active stream(s) • Evidence strength:{' '}
                    <span className="text-slate-400 font-medium">{alert.evidence_strength}</span> • Model R²: {alert.model_validation.r2}
                  </div>
                </div>

                <div className="flex sm:flex-col items-end justify-between shrink-0 gap-2">
                  <div className="text-right">
                    <div className="text-[10px] text-slate-500">Composite Score</div>
                    <div className="text-sm font-bold text-slate-100 font-mono">
                      {alert.composite_risk_score.toFixed(1)}/100
                    </div>
                  </div>
                  <button className="text-[11px] text-emerald-400 hover:text-emerald-300 flex items-center gap-1 font-medium">
                    Why this alert?
                    <ExternalLink className="w-3 h-3" />
                  </button>
                </div>
              </div>
            )
          })
        )}
      </div>
    </Card>
  )
}
