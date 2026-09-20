import React from 'react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts'
import { ScatterPointItem } from '../../types/validation'

interface PredictionComparisonChartProps {
  scatterPoints: ScatterPointItem[]
}

export const PredictionComparisonChart: React.FC<PredictionComparisonChartProps> = ({ scatterPoints }) => {
  return (
    <Card className="border-border/80">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base font-bold text-foreground">Observed vs Predicted Yields</CardTitle>
            <CardDescription className="text-xs">
              Direct parity comparison on 2016–2017 out-of-time test sample (Ideal predictions lie on 45° diagonal)
            </CardDescription>
          </div>
          <Badge variant="outline" className="text-xs">R² = 0.7866</Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 10, left: 0, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis
                type="number"
                dataKey="observed"
                name="Observed Yield"
                unit=" kg/ha"
                domain={[500, 5000]}
                tick={{ fontSize: 10 }}
              />
              <YAxis
                type="number"
                dataKey="predicted"
                name="Predicted Yield"
                unit=" kg/ha"
                domain={[500, 5000]}
                tick={{ fontSize: 10 }}
              />
              <Tooltip
                cursor={{ strokeDasharray: '3 3' }}
                contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))', borderRadius: '8px', fontSize: '11px' }}
                formatter={(value: any, name: any) => [`${value} kg/ha`, name]}
                labelFormatter={(_: any, payload: any) => {
                  if (payload && payload.length > 0) {
                    const p = payload[0].payload
                    return `${p.district}, ${p.state} (${p.year})`
                  }
                  return ''
                }}
              />
              <Scatter
                name="Test Observations"
                data={scatterPoints}
                fill="hsl(var(--primary))"
                fillOpacity={0.65}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="text-[11px] text-muted-foreground text-center pt-2">
          Points clustered tightly along the center axis demonstrate consistent variance tracking across both low-yielding plateau and high-yielding irrigated regions.
        </div>
      </CardContent>
    </Card>
  )
}
