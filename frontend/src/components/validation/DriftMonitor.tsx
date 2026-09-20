import React from 'react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { Activity, ShieldAlert, CheckCircle2 } from 'lucide-react'
import { DriftOverviewResponse } from '../../types/validation'

interface DriftMonitorProps {
  drift?: DriftOverviewResponse
}

export const DriftMonitor: React.FC<DriftMonitorProps> = ({ drift }) => {
  const features = drift?.features || []
  const overall = drift?.overall_status || 'NORMAL'

  return (
    <Card className="border-border/80">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base font-bold text-foreground">Agricultural Feature Drift Monitor</CardTitle>
            <CardDescription className="text-xs">
              Population Stability Index (PSI) and Kolmogorov-Smirnov distribution shifts between 2010–2015 baseline and 2016–2017 test set
            </CardDescription>
          </div>
          <Badge
            variant={overall === 'NORMAL' ? 'success' : overall === 'WATCH' ? 'warning' : 'destructive'}
            className="text-xs"
          >
            {overall === 'NORMAL' ? 'STABLE DISTRIBUTIONS' : overall}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-muted/50 border-b border-border/80 text-muted-foreground uppercase font-semibold">
              <tr>
                <th className="py-2.5 px-3">Agricultural Feature</th>
                <th className="py-2.5 px-3">PSI Score</th>
                <th className="py-2.5 px-3">KS Statistic</th>
                <th className="py-2.5 px-3">Ref. Mean (2010-15)</th>
                <th className="py-2.5 px-3">Eval. Mean (2016-17)</th>
                <th className="py-2.5 px-3">Mean Shift</th>
                <th className="py-2.5 px-3 text-right">Drift Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60">
              {features.map((f, idx) => (
                <tr key={idx} className="hover:bg-muted/30">
                  <td className="py-2.5 px-3 font-semibold text-foreground">{f.feature_name}</td>
                  <td className="py-2.5 px-3 font-mono font-bold">{f.psi_score.toFixed(4)}</td>
                  <td className="py-2.5 px-3 font-mono">{f.ks_statistic.toFixed(3)}</td>
                  <td className="py-2.5 px-3 font-mono">{f.reference_mean.toFixed(1)}</td>
                  <td className="py-2.5 px-3 font-mono">{f.evaluation_mean.toFixed(1)}</td>
                  <td className="py-2.5 px-3 font-mono">
                    <span className={f.mean_shift_pct > 0 ? 'text-emerald-500' : 'text-amber-500'}>
                      {f.mean_shift_pct > 0 ? `+${f.mean_shift_pct}%` : `${f.mean_shift_pct}%`}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-right">
                    <Badge
                      variant={
                        f.status === 'NORMAL'
                          ? 'success'
                          : f.status === 'WATCH'
                          ? 'warning'
                          : 'destructive'
                      }
                      className="text-[10px]"
                    >
                      {f.status}
                    </Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="text-[11px] text-muted-foreground">
          <strong>Threshold Rules:</strong> PSI &lt; 0.10: Stable (NORMAL) | 0.10 &le; PSI &lt; 0.25: Moderate Shift (WATCH) | PSI &ge; 0.25: Significant Shift (DRIFT DETECTED).
        </div>
      </CardContent>
    </Card>
  )
}
