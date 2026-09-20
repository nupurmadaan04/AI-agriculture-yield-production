import React, { useState } from 'react'
import { BarChart3, ArrowUpDown, Check, AlertCircle, Info } from 'lucide-react'
import { GlobalImportanceResponse } from '../../types/explainability'

interface GlobalFeatureImportanceProps {
  data?: GlobalImportanceResponse
  isLoading?: boolean
}

export const GlobalFeatureImportance: React.FC<GlobalFeatureImportanceProps> = ({
  data,
  isLoading
}) => {
  const [sortMethod, setSortMethod] = useState<'native' | 'permutation'>('native')

  if (isLoading) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 animate-pulse">
        <div className="h-6 w-48 bg-slate-800 rounded mb-4" />
        <div className="space-y-3">
          {[1, 2, 3, 4, 5].map((n) => (
            <div key={n} className="h-10 bg-slate-800/60 rounded" />
          ))}
        </div>
      </div>
    )
  }

  if (!data) return null

  const sortedFeatures = [...data.features].sort((a, b) => {
    return sortMethod === 'native'
      ? b.native_importance - a.native_importance
      : b.permutation_importance - a.permutation_importance
  })

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 mb-6">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4 mb-6">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold text-white">Global Model Feature Importance</h3>
            <span className="px-2 py-0.5 text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded">
              v{data.version}
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Evaluating {data.total_features_evaluated} features across tree split Gini vs holdout permutation loss
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400">Sort by:</span>
          <div className="flex bg-slate-800 p-0.5 rounded-lg border border-slate-700">
            <button
              onClick={() => setSortMethod('native')}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                sortMethod === 'native' ? 'bg-emerald-500 text-slate-950 font-semibold' : 'text-slate-400 hover:text-white'
              }`}
            >
              Model-Native (Gini)
            </button>
            <button
              onClick={() => setSortMethod('permutation')}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                sortMethod === 'permutation' ? 'bg-emerald-500 text-slate-950 font-semibold' : 'text-slate-400 hover:text-white'
              }`}
            >
              Permutation (Holdout)
            </button>
          </div>
        </div>
      </div>

      <div className="space-y-3 mb-6">
        {sortedFeatures.map((item) => {
          const nativePct = Math.round(item.native_importance * 100)
          const permPct = Math.round(item.permutation_importance * 100)

          return (
            <div key={item.feature} className="p-3 bg-slate-800/40 rounded-lg border border-slate-800/80 hover:border-slate-700 transition-colors">
              <div className="flex items-center justify-between gap-3 mb-2">
                <div className="flex items-center gap-2.5">
                  <span className="w-5 h-5 rounded-full bg-slate-800 text-slate-300 text-[11px] font-mono flex items-center justify-center border border-slate-700">
                    {sortMethod === 'native' ? item.native_rank : item.permutation_rank}
                  </span>
                  <div>
                    <span className="text-sm font-medium text-white">{item.feature_label}</span>
                    <span className="text-[11px] font-mono text-slate-400 ml-2">({item.feature})</span>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  {item.rank_agreement ? (
                    <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1">
                      <Check className="w-3 h-3" /> Rank Consensus
                    </span>
                  ) : (
                    <span className="text-[10px] px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20 flex items-center gap-1">
                      <AlertCircle className="w-3 h-3" /> Rank Divergence (Gini #{item.native_rank} vs Perm #{item.permutation_rank})
                    </span>
                  )}
                  <div className="text-right min-w-[70px]">
                    <span className="text-xs font-semibold text-emerald-400">
                      {sortMethod === 'native' ? `${(item.native_importance * 100).toFixed(1)}%` : `${(item.permutation_importance * 100).toFixed(1)}%`}
                    </span>
                  </div>
                </div>
              </div>

              {/* Dual Comparative Bar */}
              <div className="space-y-1.5 pt-1">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-slate-400 w-16">Native Gini</span>
                  <div className="flex-1 bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-800">
                    <div className="bg-emerald-500 h-full rounded-full transition-all duration-500" style={{ width: `${nativePct * 2.5}%` }} />
                  </div>
                  <span className="text-[10px] text-slate-400 w-10 text-right">{(item.native_importance * 100).toFixed(1)}%</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-slate-400 w-16">Permutation</span>
                  <div className="flex-1 bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-800">
                    <div className="bg-blue-500 h-full rounded-full transition-all duration-500" style={{ width: `${permPct * 2.5}%` }} />
                  </div>
                  <span className="text-[10px] text-slate-400 w-10 text-right">{(item.permutation_importance * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      <div className="p-3 bg-slate-800/30 rounded-lg border border-slate-800 flex items-start gap-2.5 text-xs text-slate-400">
        <Info className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
        <div>
          {data.scientific_disclaimer}
        </div>
      </div>
    </div>
  )
}
