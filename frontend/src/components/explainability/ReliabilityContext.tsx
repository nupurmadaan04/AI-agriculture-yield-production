import React from 'react'
import { ShieldCheck, Database, Award, AlertTriangle } from 'lucide-react'

interface ReliabilityContextProps {
  modelVersion?: string
  datasetVersion?: string
  r2?: number
  mae?: number
  driftStatus?: string
  explanationMethod?: string
}

export const ReliabilityContext: React.FC<ReliabilityContextProps> = ({
  modelVersion = '2.1.0',
  datasetVersion = 'ICRISAT 1966–2017 Panel',
  r2 = 0.7866,
  mae = 353.01,
  driftStatus = 'NORMAL',
  explanationMethod = 'Marginal Reference Perturbation Attribution'
}) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 mb-6">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4 mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-emerald-500/10 text-emerald-400 rounded-lg border border-emerald-500/20">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white">Active Explanatory Model</h3>
            <p className="text-xs text-slate-400">Exogenous Random Forest Forecaster · v{modelVersion}</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="px-2.5 py-1 text-xs font-medium rounded-md bg-slate-800 text-slate-300 border border-slate-700 flex items-center gap-1.5">
            <Database className="w-3.5 h-3.5 text-blue-400" />
            {datasetVersion}
          </span>
          <span className="px-2.5 py-1 text-xs font-medium rounded-md bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1.5">
            <Award className="w-3.5 h-3.5" />
            Drift: {driftStatus}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-800">
          <p className="text-xs text-slate-400 mb-1">Out-of-Time R²</p>
          <p className="text-base font-bold text-white">{(r2 * 100).toFixed(1)}%</p>
          <p className="text-[10px] text-emerald-400 mt-0.5">High Validation Accuracy</p>
        </div>
        <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-800">
          <p className="text-xs text-slate-400 mb-1">Validation MAE</p>
          <p className="text-base font-bold text-white">{mae.toFixed(1)} <span className="text-xs font-normal text-slate-400">kg/ha</span></p>
          <p className="text-[10px] text-slate-400 mt-0.5">Holdout Residual Spread</p>
        </div>
        <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-800 col-span-2 sm:col-span-2">
          <p className="text-xs text-slate-400 mb-1">XAI Attribution Method</p>
          <p className="text-xs font-semibold text-emerald-400 truncate">{explanationMethod}</p>
          <p className="text-[10px] text-slate-400 mt-0.5 flex items-center gap-1">
            <AlertTriangle className="w-3 h-3 text-amber-400" /> Non-causal empirical response curves
          </p>
        </div>
      </div>
    </div>
  )
}
