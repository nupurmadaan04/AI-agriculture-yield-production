import React, { useState } from 'react'
import { SpatialOutlierItem } from '../../types/geospatial'
import { AlertCircle, ArrowUpRight, ArrowDownRight, Search, ShieldAlert } from 'lucide-react'

interface SpatialOutlierTableProps {
  outliers: SpatialOutlierItem[]
  onSelectDistrict?: (state: string, district: string) => void
}

export const SpatialOutlierTable: React.FC<SpatialOutlierTableProps> = ({
  outliers,
  onSelectDistrict,
}) => {
  const [searchTerm, setSearchTerm] = useState('')

  const filtered = outliers.filter(
    (o) =>
      o.district.toLowerCase().includes(searchTerm.toLowerCase()) ||
      o.state.toLowerCase().includes(searchTerm.toLowerCase())
  )

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 backdrop-blur-md">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-5 h-5 text-amber-400" />
            <h3 className="text-base font-bold text-white">Within-State Spatial Outliers</h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Districts exhibiting statistical yield departures (|z| ≥ 1.8) or excessive volatility relative to state baseline
          </p>
        </div>

        <div className="relative w-full sm:w-64">
          <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="Filter by state or district..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-slate-800/80 border border-slate-700 text-xs text-white rounded-lg pl-8 pr-3 py-1.5 focus:outline-none focus:border-emerald-500"
          />
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400 font-mono text-[11px]">
              <th className="py-2.5 px-3">State / District</th>
              <th className="py-2.5 px-3">Yield (kg/ha)</th>
              <th className="py-2.5 px-3">State Baseline</th>
              <th className="py-2.5 px-3">Within-State z-Score</th>
              <th className="py-2.5 px-3">Relative Ratio</th>
              <th className="py-2.5 px-3">Deviation Factors</th>
              <th className="py-2.5 px-3 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={7} className="text-center py-6 text-slate-500 italic">
                  No spatial outliers matched the current search filter.
                </td>
              </tr>
            ) : (
              filtered.slice(0, 8).map((item, idx) => (
                <tr key={`${item.state}-${item.district}-${idx}`} className="hover:bg-slate-800/40 transition-colors">
                  <td className="py-2.5 px-3">
                    <div className="font-bold text-white">{item.district}</div>
                    <div className="text-[10px] text-slate-400">{item.state}</div>
                  </td>
                  <td className="py-2.5 px-3 font-mono text-white font-bold">
                    {item.yield_kg_ha.toLocaleString()}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-slate-400">
                    {item.state_mean_yield.toLocaleString()}
                  </td>
                  <td className="py-2.5 px-3">
                    <span
                      className={`inline-flex items-center gap-1 font-mono px-2 py-0.5 rounded text-[11px] font-bold ${
                        item.within_state_zscore < 0
                          ? 'bg-red-500/10 text-red-400 border border-red-500/20'
                          : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                      }`}
                    >
                      {item.within_state_zscore < 0 ? (
                        <ArrowDownRight className="w-3 h-3" />
                      ) : (
                        <ArrowUpRight className="w-3 h-3" />
                      )}
                      {item.within_state_zscore > 0 ? '+' : ''}
                      {item.within_state_zscore.toFixed(2)}σ
                    </span>
                  </td>
                  <td className="py-2.5 px-3 font-mono text-slate-300">
                    {item.relative_yield_ratio.toFixed(2)}x
                  </td>
                  <td className="py-2.5 px-3 max-w-xs truncate text-[11px] text-slate-400" title={item.reasons.join('; ')}>
                    {item.reasons[0]}
                  </td>
                  <td className="py-2.5 px-3 text-right">
                    {onSelectDistrict && (
                      <button
                        onClick={() => onSelectDistrict(item.state, item.district)}
                        className="px-2.5 py-1 text-[11px] rounded bg-slate-800 hover:bg-slate-700 text-emerald-400 hover:text-emerald-300 font-medium transition-colors border border-slate-700"
                      >
                        Inspect
                      </button>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
