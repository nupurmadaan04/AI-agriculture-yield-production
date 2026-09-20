import React, { useState } from 'react'
import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { Input } from '../ui/Input'
import { Table, ArrowUpDown } from 'lucide-react'
import type { Alert } from '../../types/monitoring'

interface Props {
  alerts: Alert[]
  isLoading?: boolean
  onSelectAlert: (alert: Alert) => void
}

export const RegionalRiskRanking: React.FC<Props> = ({
  alerts,
  isLoading,
  onSelectAlert,
}) => {
  const [search, setSearch] = useState('')
  const [sortBy, setSortBy] = useState<'score' | 'rank' | 'state'>('rank')

  const filteredAndSorted = alerts
    .filter((a) => {
      return (
        search === '' ||
        a.location.toLowerCase().includes(search.toLowerCase()) ||
        a.dominant_signal.toLowerCase().includes(search.toLowerCase())
      )
    })
    .sort((a, b) => {
      if (sortBy === 'score') {
        return b.composite_risk_score - a.composite_risk_score
      }
      if (sortBy === 'state') {
        return a.state.localeCompare(b.state)
      }
      return (a.priority_rank ?? 999) - (b.priority_rank ?? 999)
    })

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
            <Table className="w-5 h-5 text-purple-400" />
            Regional Priority Ranking
          </h3>
          <p className="text-xs text-slate-400">
            Multi-attribute prioritization across severity, persistence, and deviation
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Input
            placeholder="Filter regions..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="h-8 text-xs w-44 bg-slate-950/60 border-slate-700"
          />
          <button
            onClick={() => setSortBy(sortBy === 'rank' ? 'score' : 'rank')}
            className="px-2.5 py-1 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg flex items-center gap-1 transition-colors"
          >
            <ArrowUpDown className="w-3.5 h-3.5 text-slate-400" />
            Sort by {sortBy === 'rank' ? 'Score' : 'Rank'}
          </button>
        </div>
      </div>

      <div className="overflow-x-auto max-h-[380px] overflow-y-auto">
        <table className="w-full text-left text-xs">
          <thead className="sticky top-0 bg-slate-900 border-b border-slate-800 text-[11px] font-semibold text-slate-400">
            <tr>
              <th className="py-2.5 px-3">Priority</th>
              <th className="py-2.5 px-3">Region / District</th>
              <th className="py-2.5 px-3">Severity</th>
              <th className="py-2.5 px-3">Dominant Signal</th>
              <th className="py-2.5 px-3 text-right">Composite Score</th>
              <th className="py-2.5 px-3 text-center">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 text-slate-300">
            {isLoading ? (
              [...Array(5)].map((_, i) => (
                <tr key={i} className="animate-pulse">
                  <td colSpan={6} className="py-3 px-3 h-10 bg-slate-800/20" />
                </tr>
              ))
            ) : filteredAndSorted.length === 0 ? (
              <tr>
                <td colSpan={6} className="py-8 text-center text-slate-500">
                  No monitored regions match the search query.
                </td>
              </tr>
            ) : (
              filteredAndSorted.slice(0, 50).map((alert) => (
                <tr
                  key={alert.alert_id}
                  onClick={() => onSelectAlert(alert)}
                  className="hover:bg-slate-800/40 cursor-pointer transition-colors"
                >
                  <td className="py-2.5 px-3 font-mono font-bold text-slate-400">
                    #{alert.priority_rank ?? '-'}
                  </td>
                  <td className="py-2.5 px-3 font-medium text-slate-200">
                    {alert.location}
                  </td>
                  <td className="py-2.5 px-3">
                    <Badge variant={getSeverityBadgeVariant(alert.severity)} className="text-[10px]">
                      {alert.severity}
                    </Badge>
                  </td>
                  <td className="py-2.5 px-3 text-slate-400 truncate max-w-xs">
                    {alert.dominant_signal}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono font-bold text-slate-200">
                    {alert.composite_risk_score.toFixed(1)}
                  </td>
                  <td className="py-2.5 px-3 text-center">
                    <button className="text-emerald-400 hover:text-emerald-300 font-medium">
                      Inspect
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </Card>
  )
}
