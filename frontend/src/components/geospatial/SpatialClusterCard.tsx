import React from 'react'
import { SpatialClusterItem } from '../../types/geospatial'
import { Layers, CheckCircle2, TrendingUp, ShieldAlert } from 'lucide-react'

interface SpatialClusterCardProps {
  cluster: SpatialClusterItem
  isSelected: boolean
  onSelect: (clusterId: number) => void
}

export const SpatialClusterCard: React.FC<SpatialClusterCardProps> = ({
  cluster,
  isSelected,
  onSelect,
}) => {
  const getBadgeColor = (profile: string) => {
    if (profile.includes('LOW')) return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
    if (profile.includes('MODERATE')) return 'bg-amber-500/10 text-amber-400 border-amber-500/20'
    return 'bg-red-500/10 text-red-400 border-red-500/20'
  }

  return (
    <div
      onClick={() => onSelect(cluster.cluster_id)}
      className={`cursor-pointer rounded-xl p-4 border transition-all duration-200 ${
        isSelected
          ? 'bg-slate-800/90 border-emerald-500 shadow-lg ring-1 ring-emerald-500/50'
          : 'bg-slate-900/60 border-slate-800 hover:border-slate-700 hover:bg-slate-900/80'
      }`}
    >
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-xs">
            {cluster.cluster_id}
          </div>
          <h4 className="font-bold text-white text-sm">{cluster.cluster_name}</h4>
        </div>
        <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${getBadgeColor(cluster.risk_profile)}`}>
          {cluster.risk_profile}
        </span>
      </div>

      <p className="text-xs text-slate-400 mb-3 leading-relaxed">
        {cluster.archetype}
      </p>

      <div className="grid grid-cols-3 gap-2 py-2 border-t border-b border-slate-800/80 text-[11px] mb-3">
        <div>
          <span className="text-slate-500 block text-[10px]">Avg Yield</span>
          <strong className="text-white font-mono">{cluster.avg_yield_kg_ha.toLocaleString()} kg/ha</strong>
        </div>
        <div>
          <span className="text-slate-500 block text-[10px]">Volatility</span>
          <strong className="text-slate-200 font-mono">±{cluster.avg_volatility_pct}%</strong>
        </div>
        <div>
          <span className="text-slate-500 block text-[10px]">Districts</span>
          <strong className="text-emerald-400 font-mono">{cluster.district_count} dists</strong>
        </div>
      </div>

      <div className="flex items-center justify-between text-[11px] text-slate-400">
        <span className="truncate">
          Dominant: <strong className="text-slate-300">{Object.keys(cluster.dominant_states).slice(0, 3).join(', ')}</strong>
        </span>
        <span className="text-[10px] text-slate-500 font-mono">
          {cluster.avg_theil_sen_slope > 0 ? '+' : ''}{cluster.avg_theil_sen_slope} kg/ha/yr
        </span>
      </div>
    </div>
  )
}
