import React from 'react'
import { HorizonForecastItem } from '../../types/temporal'
import { formatYield } from '../../lib/utils'
import { Badge } from '../ui/Badge'

interface ForecastTableProps {
  forecasts: HorizonForecastItem[]
  latestObservedYield: number
}

export const ForecastTable: React.FC<ForecastTableProps> = ({
  forecasts,
  latestObservedYield
}) => {
  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden text-xs">
      <table className="w-full text-left border-collapse">
        <thead>
          <tr className="border-b border-border bg-muted/40 text-muted-foreground">
            <th className="py-2.5 px-3 font-semibold">Forecast Year</th>
            <th className="py-2.5 px-3 font-semibold">Horizon</th>
            <th className="py-2.5 px-3 font-semibold">Model Forecast</th>
            <th className="py-2.5 px-3 font-semibold">Shift vs Baseline</th>
            <th className="py-2.5 px-3 font-semibold">Prediction Interval (P10–P90)</th>
            <th className="py-2.5 px-3 font-semibold">Spread %</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border/60">
          {forecasts.map((fc) => {
            const shift = fc.predicted_yield - latestObservedYield
            const shiftPct = latestObservedYield > 0 ? (shift / latestObservedYield) * 100.0 : 0.0

            return (
              <tr key={fc.forecast_year} className="hover:bg-muted/20">
                <td className="py-2.5 px-3 font-bold text-foreground font-mono">
                  {fc.forecast_year}
                </td>
                <td className="py-2.5 px-3">
                  <Badge variant="blue" className="text-[10px]">
                    {fc.horizon_years}-Year Forward
                  </Badge>
                </td>
                <td className="py-2.5 px-3 font-mono font-bold text-emerald-600 dark:text-emerald-400">
                  {formatYield(fc.predicted_yield)}
                </td>
                <td className="py-2.5 px-3 font-mono">
                  <span className={shift >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}>
                    {shift >= 0 ? `+${shift.toFixed(1)}` : shift.toFixed(1)} ({shiftPct >= 0 ? `+${shiftPct.toFixed(1)}` : shiftPct.toFixed(1)}%)
                  </span>
                </td>
                <td className="py-2.5 px-3 font-mono text-muted-foreground text-[11px]">
                  {formatYield(fc.lower_bound_p10)} – {formatYield(fc.upper_bound_p90)}
                </td>
                <td className="py-2.5 px-3 font-mono font-semibold">
                  ±{(fc.uncertainty_pct / 2).toFixed(1)}%
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
