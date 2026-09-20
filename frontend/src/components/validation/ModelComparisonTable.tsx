import React from 'react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { BenchmarkComparisonItem } from '../../types/validation'

interface ModelComparisonTableProps {
  benchmarks: BenchmarkComparisonItem[]
}

export const ModelComparisonTable: React.FC<ModelComparisonTableProps> = ({ benchmarks }) => {
  return (
    <Card className="border-border/80">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base font-bold text-foreground">Chronological Out-of-Time Model Benchmark</CardTitle>
            <CardDescription className="text-xs">
              Rigorous out-of-time evaluation on unseen test period (2016–2017) across 618 district observations
            </CardDescription>
          </div>
          <Badge variant="outline" className="font-mono text-xs">Training: &le; 2015</Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-muted/50 border-b border-border/80 text-muted-foreground uppercase font-semibold">
              <tr>
                <th className="py-2.5 px-3">Model Architecture</th>
                <th className="py-2.5 px-3">MAE (kg/ha)</th>
                <th className="py-2.5 px-3">RMSE (kg/ha)</th>
                <th className="py-2.5 px-3">R² Score</th>
                <th className="py-2.5 px-3">MAPE (%)</th>
                <th className="py-2.5 px-3 text-right">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60">
              {benchmarks.map((b, idx) => {
                const isSelected = b.model.includes('Selected') || b.model.includes('Random Forest Forecaster')
                return (
                  <tr key={idx} className={isSelected ? 'bg-primary/5 font-semibold text-foreground' : 'hover:bg-muted/30 text-muted-foreground'}>
                    <td className="py-2.5 px-3 flex items-center gap-2">
                      <span className={isSelected ? 'text-primary font-bold' : ''}>{b.model}</span>
                      {isSelected && <Badge variant="blue" className="text-[10px] py-0">Production</Badge>}
                    </td>
                    <td className="py-2.5 px-3 font-mono">{b.mae.toFixed(1)}</td>
                    <td className="py-2.5 px-3 font-mono">{b.rmse.toFixed(1)}</td>
                    <td className="py-2.5 px-3 font-mono font-bold text-foreground">{b.r2.toFixed(4)}</td>
                    <td className="py-2.5 px-3 font-mono">{b.mape.toFixed(1)}%</td>
                    <td className="py-2.5 px-3 text-right">
                      {isSelected ? (
                        <Badge variant="success">DEPLOYED</Badge>
                      ) : (
                        <Badge variant="outline">BENCHMARK</Badge>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  )
}
