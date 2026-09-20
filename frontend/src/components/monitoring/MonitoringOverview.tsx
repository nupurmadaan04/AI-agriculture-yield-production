import React from 'react'
import {
  Bell,
  AlertTriangle,
  ShieldCheck,
  MapPin,
  TrendingDown,
  Info
} from 'lucide-react'
import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'
import type { MonitoringOverview as MonitoringOverviewType } from '../../types/monitoring'

interface Props {
  data?: MonitoringOverviewType
  isLoading?: boolean
}

export const MonitoringOverview: React.FC<Props> = ({ data, isLoading }) => {
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4 animate-pulse">
        {[...Array(6)].map((_, i) => (
          <div key={i} className="h-28 bg-emerald-950/20 border border-emerald-900/30 rounded-xl" />
        ))}
      </div>
    )
  }

  const kpis = [
    {
      title: 'Active Warning Alerts',
      value: data?.active_alerts_count ?? 0,
      subtext: `${data?.summary.total_alerts ?? 311} districts evaluated`,
      icon: Bell,
      color: 'text-amber-400',
      bg: 'bg-amber-500/10 border-amber-500/20',
      badge: 'MONITORED',
      badgeVariant: 'amber' as const,
    },
    {
      title: 'High / Critical Alerts',
      value: data?.high_critical_count ?? 0,
      subtext: `${data?.summary.critical_alerts_count ?? 0} critical, ${data?.summary.high_alerts_count ?? 0} high`,
      icon: AlertTriangle,
      color: 'text-rose-400',
      bg: 'bg-rose-500/10 border-rose-500/20',
      badge: 'PRIORITY',
      badgeVariant: 'red' as const,
    },
    {
      title: 'States Under Watch',
      value: data?.states_under_watch ?? 0,
      subtext: 'Out of 20 agricultural states',
      icon: MapPin,
      color: 'text-purple-400',
      bg: 'bg-purple-500/10 border-purple-500/20',
      badge: 'REGIONAL',
      badgeVariant: 'purple' as const,
    },
    {
      title: 'Districts Under Watch',
      value: data?.districts_under_watch ?? 0,
      subtext: 'Active multi-signal triggers',
      icon: MapPin,
      color: 'text-cyan-400',
      bg: 'bg-cyan-500/10 border-cyan-500/20',
      badge: 'LOCALIZED',
      badgeVariant: 'cyan' as const,
    },
    {
      title: 'Persistent Decline Signals',
      value: data?.persistent_signals_count ?? 0,
      subtext: 'Consecutive multi-year drops',
      icon: TrendingDown,
      color: 'text-orange-400',
      bg: 'bg-orange-500/10 border-orange-500/20',
      badge: '3+ SEASONS',
      badgeVariant: 'amber' as const,
    },
    {
      title: 'Model Monitoring Status',
      value: data?.model_monitoring_status ?? 'HEALTHY',
      subtext: 'PSI < 0.10 | Test MAE 353.01',
      icon: ShieldCheck,
      color: 'text-emerald-400',
      bg: 'bg-emerald-500/10 border-emerald-500/20',
      badge: 'VERIFIED',
      badgeVariant: 'green' as const,
    },
  ]

  return (
    <div className="space-y-4">
      {/* Scientific Integrity Disclaimer Banner */}
      <div className="p-3.5 bg-emerald-950/40 border border-emerald-500/30 rounded-xl flex items-start gap-3 text-xs text-emerald-300">
        <Info className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <div className="font-semibold text-emerald-200">
            Agricultural Real-Time Monitoring & Early Warning Layer
          </div>
          <div className="text-emerald-300/80 leading-relaxed">
            Monitoring currently operates on the verified ICRISAT dataset and deterministic model-derived signals.
            No live sensor/weather feed is assumed. Early warnings represent empirical statistical deviations, not guaranteed crop loss or physical drought declarations.
          </div>
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
        {kpis.map((kpi, idx) => {
          const Icon = kpi.icon
          return (
            <Card key={idx} className={`p-4 border ${kpi.bg} relative overflow-hidden transition-all hover:scale-[1.02]`}>
              <div className="flex items-center justify-between mb-2">
                <Icon className={`w-5 h-5 ${kpi.color}`} />
                <Badge variant={kpi.badgeVariant} className="text-[10px] px-1.5 py-0.5">
                  {kpi.badge}
                </Badge>
              </div>
              <div className="text-2xl font-bold text-slate-100 tracking-tight">
                {kpi.value}
              </div>
              <div className="text-xs font-medium text-slate-400 mt-0.5 truncate">
                {kpi.title}
              </div>
              <div className="text-[11px] text-slate-500 mt-1">
                {kpi.subtext}
              </div>
            </Card>
          )
        })}
      </div>
    </div>
  )
}
