import React from 'react'
import { Badge } from '../ui/Badge'
import { StatusBadge } from '../common/StatusBadge'
import { formatNumber } from '../../lib/utils'

interface ModelLeaderboardTableProps {
  entries: Array<{
    id: string
    modelName?: string
    model_name?: string
    modelType?: string
    model_type?: string
    featureSet?: string
    feature_set?: string
    randomR2?: number
    random_r2?: number
    temporalR2?: number
    temporal_r2?: number
    groupKFoldR2?: number
    cv_r2?: number
    randomMae?: number
    random_mae?: number
    randomRmse?: number
    random_rmse?: number
    status: any
  }>
}

export const ModelLeaderboardTable: React.FC<ModelLeaderboardTableProps> = ({ entries }) => {
  return (
    <div className="rounded-xl border border-border bg-card shadow-subtle overflow-hidden">
      <div className="p-4 border-b border-border flex items-center justify-between">
        <div>
          <h3 className="font-semibold text-sm text-foreground">Model & Baseline Benchmark Leaderboard</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Empirically verified metrics across Random (80/20), Temporal (Holdout &gt; 2015), and 5-Fold Cross-Validation
          </p>
        </div>
        <Badge variant="blue">{entries.length} Estimators Evaluated</Badge>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="bg-muted/40 text-muted-foreground uppercase text-[10px] tracking-wider border-b border-border">
            <tr>
              <th className="p-3 font-semibold">Model / Estimator</th>
              <th className="p-3 font-semibold">Category</th>
              <th className="p-3 font-semibold">Predictor Features</th>
              <th className="p-3 font-semibold text-right">Random R²</th>
              <th className="p-3 font-semibold text-right">Temporal R²</th>
              <th className="p-3 font-semibold text-right">5-Fold CV R²</th>
              <th className="p-3 font-semibold text-right">MAE (kg/ha)</th>
              <th className="p-3 font-semibold text-right">RMSE (kg/ha)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/60 font-mono text-xs">
            {entries.map((entry) => {
              const modelName = entry.model_name || entry.modelName || 'Model'
              const modelType = entry.model_type || entry.modelType || 'ml'
              const featureSet = entry.feature_set || entry.featureSet || 'Features'
              const rR2 = entry.random_r2 !== undefined ? entry.random_r2 : entry.randomR2
              const tR2 = entry.temporal_r2 !== undefined ? entry.temporal_r2 : entry.temporalR2
              const cvR2 = entry.cv_r2 !== undefined ? entry.cv_r2 : entry.groupKFoldR2
              const rMae = entry.random_mae !== undefined ? entry.random_mae : entry.randomMae
              const rRmse = entry.random_rmse !== undefined ? entry.random_rmse : entry.randomRmse

              const isDet = modelType === 'deterministic'
              return (
                <tr
                  key={entry.id}
                  className={`hover:bg-muted/30 transition-colors ${
                    isDet ? 'bg-sky-500/5 font-semibold text-foreground' : ''
                  }`}
                >
                  <td className="p-3 font-sans font-medium text-foreground">
                    <div className="flex items-center gap-2">
                      <span>{modelName}</span>
                      {isDet && (
                        <span className="text-[10px] bg-primary/10 text-primary px-1.5 py-0.5 rounded font-sans font-bold">
                          TOP BENCHMARK
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="p-3 font-sans">
                    <StatusBadge status={entry.status as any} />
                  </td>
                  <td className="p-3 font-sans text-muted-foreground text-[11px]">{featureSet}</td>
                  <td className="p-3 text-right text-foreground font-semibold">
                    {rR2 !== undefined ? formatNumber(rR2, 4) : '—'}
                  </td>
                  <td className="p-3 text-right text-foreground font-semibold">
                    {tR2 !== undefined ? formatNumber(tR2, 4) : '—'}
                  </td>
                  <td className="p-3 text-right text-foreground font-semibold">
                    {cvR2 !== undefined ? formatNumber(cvR2, 4) : '—'}
                  </td>
                  <td className="p-3 text-right text-foreground">
                    {rMae !== undefined ? formatNumber(rMae, 2) : '—'}
                  </td>
                  <td className="p-3 text-right text-foreground">
                    {rRmse !== undefined ? formatNumber(rRmse, 2) : '—'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
