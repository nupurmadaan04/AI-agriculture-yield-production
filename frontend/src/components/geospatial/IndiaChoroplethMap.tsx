import React, { useState } from 'react'
import { StateSpatialItem } from '../../types/geospatial'
import { MapPin, TrendingUp, TrendingDown, AlertTriangle, ShieldCheck, Layers } from 'lucide-react'

interface IndiaChoroplethMapProps {
  states: StateSpatialItem[]
  selectedMetric: 'yield' | 'risk' | 'anomaly' | 'forecast' | 'warning' | 'trend'
  selectedState: string | null
  onSelectState: (stateName: string) => void
}

// Layout positions for 20 states in a topological grid representation of India
const STATE_MAP_NODES: Record<string, { x: number; y: number; code: string }> = {
  'Punjab': { x: 30, y: 16, code: 'PB' },
  'Himachal Pradesh': { x: 42, y: 12, code: 'HP' },
  'Uttarakhand': { x: 46, y: 18, code: 'UK' },
  'Haryana': { x: 34, y: 22, code: 'HR' },
  'Rajasthan': { x: 22, y: 28, code: 'RJ' },
  'Uttar Pradesh': { x: 44, y: 28, code: 'UP' },
  'Bihar': { x: 58, y: 30, code: 'BR' },
  'Assam': { x: 80, y: 28, code: 'AS' },
  'West Bengal': { x: 68, y: 38, code: 'WB' },
  'Jharkhand': { x: 58, y: 38, code: 'JH' },
  'Madhya Pradesh': { x: 38, y: 40, code: 'MP' },
  'Gujarat': { x: 18, y: 42, code: 'GJ' },
  'Chhattisgarh': { x: 50, y: 48, code: 'CG' },
  'Orissa': { x: 62, y: 48, code: 'OR' },
  'Maharashtra': { x: 30, y: 56, code: 'MH' },
  'Telangana': { x: 42, y: 62, code: 'TG' },
  'Andhra Pradesh': { x: 48, y: 72, code: 'AP' },
  'Karnataka': { x: 32, y: 74, code: 'KA' },
  'Tamil Nadu': { x: 42, y: 86, code: 'TN' },
  'Kerala': { x: 34, y: 88, code: 'KL' },
}

export const IndiaChoroplethMap: React.FC<IndiaChoroplethMapProps> = ({
  states,
  selectedMetric,
  selectedState,
  onSelectState,
}) => {
  const [hoveredState, setHoveredState] = useState<StateSpatialItem | null>(null)

  const stateMap = React.useMemo(() => {
    const map = new Map<string, StateSpatialItem>()
    states.forEach((s) => map.set(s.state.toLowerCase(), s))
    return map
  }, [states])

  // Compute color based on metric
  const getNodeColor = (item?: StateSpatialItem) => {
    if (!item) return '#334155'

    if (selectedMetric === 'yield') {
      const y = item.average_yield_kg_ha
      if (y >= 3600) return '#059669' // Emerald-600
      if (y >= 2800) return '#10b981' // Emerald-500
      if (y >= 2200) return '#34d399' // Emerald-400
      if (y >= 1600) return '#6ee7b7' // Emerald-300
      return '#a7f3d0' // Light emerald
    }

    if (selectedMetric === 'risk') {
      const r = item.risk_score
      if (r >= 65) return '#dc2626' // Red-600
      if (r >= 50) return '#ea580c' // Orange-600
      if (r >= 35) return '#f59e0b' // Amber-500
      return '#10b981' // Green
    }

    if (selectedMetric === 'anomaly') {
      const a = item.anomaly_count
      if (a >= 8) return '#7c3aed' // Violet-600
      if (a >= 4) return '#8b5cf6' // Violet-500
      if (a >= 1) return '#a78bfa' // Violet-400
      return '#334155' // Slate
    }

    if (selectedMetric === 'forecast') {
      const f = item.forecast_1yr_kg_ha
      if (f >= 3600) return '#0284c7' // Sky-600
      if (f >= 2800) return '#0ea5e9' // Sky-500
      if (f >= 2200) return '#38bdf8' // Sky-400
      return '#7dd3fc'
    }

    if (selectedMetric === 'trend') {
      const dir = item.trend_direction.toUpperCase()
      if (dir.includes('STRONG INCREASING')) return '#059669'
      if (dir.includes('INCREASING')) return '#10b981'
      if (dir.includes('STABLE')) return '#64748b'
      if (dir.includes('STRONG DECREASING')) return '#dc2626'
      if (dir.includes('DECREASING')) return '#ea580c'
      return '#64748b'
    }

    // Default warning
    return '#f59e0b'
  }

  const getMetricDisplayValue = (item: StateSpatialItem) => {
    switch (selectedMetric) {
      case 'yield':
        return `${item.average_yield_kg_ha.toLocaleString()} kg/ha`
      case 'risk':
        return `Risk ${item.risk_score.toFixed(1)} (${item.risk_level})`
      case 'anomaly':
        return `${item.anomaly_count} Anomalies`
      case 'forecast':
        return `FC: ${item.forecast_1yr_kg_ha.toLocaleString()} kg/ha`
      case 'trend':
        return `${item.theil_sen_slope > 0 ? '+' : ''}${item.theil_sen_slope.toFixed(1)} kg/ha/yr`
      default:
        return `${item.average_yield_kg_ha} kg/ha`
    }
  }

  return (
    <div className="relative bg-slate-900/90 border border-slate-800 rounded-2xl p-6 overflow-hidden shadow-2xl backdrop-blur-md">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Layers className="w-5 h-5 text-emerald-400" />
          <h3 className="text-base font-bold text-white tracking-wide">
            India Agricultural Geographic Topology
          </h3>
        </div>
        <span className="text-xs text-slate-400 font-mono bg-slate-800/80 px-2.5 py-1 rounded-md border border-slate-700">
          Metric: <strong className="text-emerald-300 uppercase">{selectedMetric}</strong>
        </span>
      </div>

      {/* SVG Spatial Canvas */}
      <div className="relative w-full h-[520px] bg-slate-950/60 rounded-xl border border-slate-800/80 flex items-center justify-center p-4">
        <svg
          viewBox="0 0 100 100"
          className="w-full h-full max-h-[500px]"
          style={{ filter: 'drop-shadow(0 4px 12px rgba(0,0,0,0.5))' }}
        >
          {/* Background Grid Lines */}
          <defs>
            <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
              <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#1e293b" strokeWidth="0.3" />
            </pattern>
          </defs>
          <rect width="100" height="100" fill="url(#grid)" />

          {/* Render State Spatial Hexagons / Circles */}
          {Object.entries(STATE_MAP_NODES).map(([name, pos]) => {
            const data = stateMap.get(name.toLowerCase())
            const isSelected = selectedState?.toLowerCase() === name.toLowerCase()
            const isHovered = hoveredState?.state.toLowerCase() === name.toLowerCase()
            const color = getNodeColor(data)

            return (
              <g
                key={name}
                className="cursor-pointer transition-all duration-200"
                onClick={() => onSelectState(name)}
                onMouseEnter={() => data && setHoveredState(data)}
                onMouseLeave={() => setHoveredState(null)}
              >
                {/* Connecting Pulse Glow if Selected */}
                {isSelected && (
                  <circle
                    cx={pos.x}
                    cy={pos.y}
                    r="6.5"
                    fill="none"
                    stroke="#10b981"
                    strokeWidth="0.8"
                    strokeDasharray="2 1"
                    className="animate-spin"
                  />
                )}

                {/* State Interactive Node */}
                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r={isSelected ? 5.2 : isHovered ? 4.8 : 4.0}
                  fill={color}
                  stroke={isSelected ? '#ffffff' : '#0f172a'}
                  strokeWidth={isSelected ? 1.2 : 0.8}
                  style={{
                    filter: isHovered || isSelected ? 'brightness(1.2) drop-shadow(0 0 6px rgba(16,185,129,0.6))' : 'none',
                    transition: 'all 0.2s ease',
                  }}
                />

                {/* State Label Code */}
                <text
                  x={pos.x}
                  y={pos.y + 1.2}
                  textAnchor="middle"
                  fill="#ffffff"
                  fontSize="2.4"
                  fontWeight="bold"
                  pointerEvents="none"
                  fontFamily="monospace"
                >
                  {pos.code}
                </text>
              </g>
            )
          })}
        </svg>

        {/* Floating Tooltip */}
        {hoveredState && (
          <div className="absolute bottom-4 right-4 bg-slate-900/95 border border-emerald-500/40 rounded-xl p-3.5 shadow-2xl backdrop-blur-md max-w-xs text-xs z-20 pointer-events-none animate-in fade-in">
            <div className="flex items-center justify-between gap-3 border-b border-slate-800 pb-2 mb-2">
              <span className="font-bold text-white text-sm">{hoveredState.state}</span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                {hoveredState.region}
              </span>
            </div>
            <div className="space-y-1 text-slate-300">
              <div className="flex justify-between">
                <span className="text-slate-400">Mean Yield:</span>
                <strong className="text-white">{hoveredState.average_yield_kg_ha.toLocaleString()} kg/ha</strong>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Risk Score:</span>
                <strong className={hoveredState.risk_score >= 50 ? 'text-amber-400' : 'text-emerald-400'}>
                  {hoveredState.risk_score.toFixed(1)} ({hoveredState.risk_level})
                </strong>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Trend:</span>
                <strong className="text-slate-200">{hoveredState.trend_direction}</strong>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Districts:</span>
                <strong className="text-slate-200">{hoveredState.district_count} districts</strong>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Map Legend */}
      <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
        <div className="flex items-center gap-2">
          <span>Scale:</span>
          <div className="flex items-center gap-1.5 font-mono text-[10px]">
            <span className="px-2 py-0.5 rounded bg-emerald-700 text-white">High Productivity</span>
            <span className="px-2 py-0.5 rounded bg-amber-600 text-white">Moderate / Risk</span>
            <span className="px-2 py-0.5 rounded bg-red-600 text-white">Vulnerable / Shock</span>
          </div>
        </div>
        <span className="text-[11px] text-slate-500 italic">
          Click any state node to inspect regional profile & districts
        </span>
      </div>
    </div>
  )
}
