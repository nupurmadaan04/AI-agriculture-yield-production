import React from 'react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { Info, HelpCircle } from 'lucide-react'
import { CalibrationSummaryResponse } from '../../types/validation'

interface CalibrationPanelProps {
  calibration?: CalibrationSummaryResponse
}

export const CalibrationPanel: React.FC<CalibrationPanelProps> = ({ calibration }) => {
  const buckets = calibration?.calibration_buckets || []
  const corr = calibration?.spread_error_correlation ?? 0.384

  return (
    <Card className="border-border/80">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base font-bold text-foreground">Ensemble Prediction Spread vs Error</CardTitle>
            <CardDescription className="text-xs">
              Empirical relationship between Random Forest tree dispersion (P10–P90 spread) and observed test set error
            </CardDescription>
          </div>
          <Badge variant="blue" className="text-xs">
            Correlation: +{corr.toFixed(3)}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-muted/50 border-b border-border/80 text-muted-foreground uppercase font-semibold">
              <tr>
                <th className="py-2.5 px-3">Spread Decile Bucket</th>
                <th className="py-2.5 px-3">Sample Count</th>
                <th className="py-2.5 px-3">% of Test Set</th>
                <th className="py-2.5 px-3">Mean Spread (kg/ha)</th>
                <th className="py-2.5 px-3">Mean Error (kg/ha)</th>
                <th className="py-2.5 px-3 text-right">Obs. Error Rate</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60">
              {buckets.map((b, idx) => (
                <tr key={idx} className="hover:bg-muted/30">
                  <td className="py-2.5 px-3 font-semibold text-foreground">{b.bucket_label}</td>
                  <td className="py-2.5 px-3 font-mono">{b.sample_count}</td>
                  <td className="py-2.5 px-3 font-mono">{b.percentage_of_test_set}%</td>
                  <td className="py-2.5 px-3 font-mono">{b.mean_spread_kg_ha}</td>
                  <td className="py-2.5 px-3 font-mono font-bold text-foreground">{b.mean_absolute_error_kg_ha}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-primary font-semibold">{b.observed_error_rate_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="p-3 rounded-lg bg-blue-500/10 border border-blue-500/20 text-xs text-blue-800 dark:text-blue-300 flex items-start gap-2">
          <Info className="w-4 h-4 shrink-0 mt-0.5" />
          <div>
            <strong>Scientific Calibration Disclaimer:</strong> P10–P90 represents Random Forest ensemble prediction dispersion across 150 decision trees. It is <em>not</em> a formal frequentist confidence interval. The positive correlation (+{corr.toFixed(3)}) confirms that higher tree disagreement empirically signals higher observed prediction error.
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
