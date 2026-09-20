import React from 'react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts'
import { ErrorSummaryResponse } from '../../types/validation'

interface ErrorDistributionChartProps {
  errorSummary?: ErrorSummaryResponse
}

export const ErrorDistributionChart: React.FC<ErrorDistributionChartProps> = ({ errorSummary }) => {
  const data = errorSummary?.residual_bins || []
  const sev = errorSummary?.severity_breakdown

  return (
    <Card className="border-border/80">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base font-bold text-foreground">Residual & Error Distribution</CardTitle>
            <CardDescription className="text-xs">
              Observed minus Predicted yields (Residuals) across 618 out-of-time test observations
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs">Mean Res: {errorSummary?.mean_absolute_error ? `${errorSummary.mean_absolute_error} kg/ha` : '0 kg/ha'}</Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
              <XAxis dataKey="bin_label" tick={{ fontSize: 10 }} interval={0} angle={-20} textAnchor="end" height={40} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip
                contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))', borderRadius: '8px', fontSize: '11px' }}
                formatter={(val: any) => [`${val} records`, 'Frequency']}
              />
              <Bar dataKey="count" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {sev && (
          <div className="grid grid-cols-3 gap-2 pt-2 border-t border-border/60 text-center text-xs">
            <div className="p-2 rounded bg-emerald-500/10 border border-emerald-500/20">
              <div className="text-[10px] text-muted-foreground uppercase font-semibold">Low Error (&lt;250 kg/ha)</div>
              <div className="text-sm font-bold font-mono text-emerald-600 dark:text-emerald-400">{sev.low_error_pct}% ({sev.low_error_count})</div>
            </div>
            <div className="p-2 rounded bg-amber-500/10 border border-amber-500/20">
              <div className="text-[10px] text-muted-foreground uppercase font-semibold">Moderate (250–600 kg/ha)</div>
              <div className="text-sm font-bold font-mono text-amber-600 dark:text-amber-400">{sev.moderate_error_pct}% ({sev.moderate_error_count})</div>
            </div>
            <div className="p-2 rounded bg-destructive/10 border border-destructive/20">
              <div className="text-[10px] text-muted-foreground uppercase font-semibold">High (&gt;600 kg/ha)</div>
              <div className="text-sm font-bold font-mono text-destructive">{sev.high_error_pct}% ({sev.high_error_count})</div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
