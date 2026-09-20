import React, { useState } from 'react'
import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { MapPin, Globe } from 'lucide-react'
import type { StateWarningMapItem } from '../../types/monitoring'

interface Props {
  statesData: StateWarningMapItem[]
  isLoading?: boolean
  selectedState: string
  onSelectState: (state: string) => void
}

export const WarningMap: React.FC<Props> = ({
  statesData,
  isLoading,
  selectedState,
  onSelectState,
}) => {
  const [hoveredState, setHoveredState] = useState<StateWarningMapItem | null>(null)

  const getSeverityBg = (severity: string, isSelected: boolean) => {
    if (isSelected) {
      return 'bg-emerald-500/20 border-emerald-400 ring-2 ring-emerald-500/50 shadow-lg shadow-emerald-950/50'
    }
    switch (severity.toUpperCase()) {
      case 'CRITICAL':
        return 'bg-red-950/40 border-red-500/60 text-red-300 hover:bg-red-900/50'
      case 'HIGH':
        return 'bg-rose-950/40 border-rose-500/60 text-rose-300 hover:bg-rose-900/50'
      case 'ELEVATED':
        return 'bg-amber-950/40 border-amber-500/60 text-amber-300 hover:bg-amber-900/50'
      case 'WATCH':
        return 'bg-amber-950/20 border-amber-500/30 text-amber-200 hover:bg-amber-900/30'
      default:
        return 'bg-slate-950/40 border-slate-800 text-slate-300 hover:bg-slate-800/40'
    }
  }

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
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-800">
        <div>
          <h3 className="text-base font-semibold text-slate-100 flex items-center gap-2">
            <Globe className="w-5 h-5 text-cyan-400" />
            National Early Warning & Regional Risk Map
          </h3>
          <p className="text-xs text-slate-400">
            State-level aggregated warning tiers and signal density (20 verified ICRISAT states)
          </p>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-2 text-[11px] text-slate-400 flex-wrap">
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded bg-red-500" /> Critical
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded bg-rose-500" /> High
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded bg-amber-500" /> Elevated
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded bg-blue-500" /> Info
          </span>
        </div>
      </div>

      {/* State Grid Map */}
      <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-5 gap-2.5">
        {isLoading ? (
          [...Array(20)].map((_, i) => (
            <div key={i} className="h-20 bg-slate-800/40 rounded-xl animate-pulse" />
          ))
        ) : (
          statesData.map((st) => {
            const isSelected = selectedState.toLowerCase() === st.state.toLowerCase()
            return (
              <div
                key={st.state}
                onClick={() => onSelectState(st.state)}
                onMouseEnter={() => setHoveredState(st)}
                onMouseLeave={() => setHoveredState(null)}
                className={`p-3 rounded-xl border transition-all cursor-pointer flex flex-col justify-between space-y-2 ${getSeverityBg(
                  st.severity,
                  isSelected
                )}`}
              >
                <div className="flex items-start justify-between">
                  <span className="text-xs font-bold text-slate-100 truncate pr-1">
                    {st.state}
                  </span>
                  <Badge variant={getSeverityBadgeVariant(st.severity)} className="text-[9px] px-1 py-0.5">
                    {st.severity}
                  </Badge>
                </div>

                <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1 border-t border-slate-800/60">
                  <span>{st.total_districts_monitored} dists</span>
                  <span className="font-mono text-slate-300 font-medium">
                    {st.average_yield_kg_ha.toLocaleString()} kg/ha
                  </span>
                </div>
              </div>
            )
          })
        )}
      </div>

      {/* Selected / Hovered State Details Strip */}
      {hoveredState && (
        <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-xl flex items-center justify-between text-xs text-slate-300 animate-fadeIn">
          <div className="flex items-center gap-2">
            <MapPin className="w-4 h-4 text-cyan-400" />
            <span className="font-bold text-slate-100">{hoveredState.state}</span>
            <span className="text-slate-500">•</span>
            <span>{hoveredState.dominant_concern}</span>
          </div>
          <div className="flex items-center gap-3">
            <span>
              Active Signals: <span className="font-bold text-amber-400">{hoveredState.active_signals_count}</span>
            </span>
            <span>
              Avg Yield: <span className="font-bold text-emerald-400">{hoveredState.average_yield_kg_ha} kg/ha</span>
            </span>
          </div>
        </div>
      )}
    </Card>
  )
}
