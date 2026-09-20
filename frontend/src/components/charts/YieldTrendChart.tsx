import React from 'react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Area,
  ComposedChart
} from 'recharts'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../ui/Card'
import { formatNumber } from '../../lib/utils'

interface YieldTrendChartProps {
  data: Array<{
    year: number
    avgYield?: number
    average_yield?: number
  }>
  title?: string
  subtitle?: string
}

export const YieldTrendChart: React.FC<YieldTrendChartProps> = ({
  data,
  title = "Historical Rice Yield Trend (2010–2017)",
  subtitle = "Average yield (kg/ha) across reporting districts in India"
}) => {
  const chartData = data.map(d => ({
    year: d.year,
    yieldVal: d.average_yield !== undefined ? d.average_yield : (d.avgYield !== undefined ? d.avgYield : 0)
  }))

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{subtitle}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <defs>
                <linearGradient id="yieldGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#0284c7" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#0284c7" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
              <XAxis dataKey="year" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} />
              <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} domain={['auto', 'auto']} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--card))',
                  borderColor: 'hsl(var(--border))',
                  borderRadius: '0.5rem',
                  fontSize: '12px',
                  boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
                }}
                formatter={(value: any) => [`${formatNumber(Number(value), 1)} kg/ha`, 'Avg Yield']}
              />
              <Area type="monotone" dataKey="yieldVal" stroke="#0284c7" strokeWidth={2.5} fillOpacity={1} fill="url(#yieldGradient)" />
              <Line type="monotone" dataKey="yieldVal" stroke="#0284c7" strokeWidth={2.5} dot={{ r: 4, fill: '#0284c7' }} activeDot={{ r: 6 }} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}

interface ProductionAreaScatterProps {
  title?: string
  subtitle?: string
}

export const ProductionAreaScatter: React.FC<ProductionAreaScatterProps> = ({
  title = "Production vs. Cultivated Area Distribution",
  subtitle = "Demonstrates linear ratio slope indicating mathematical yield formulation"
}) => {
  const scatterPoints = [
    { area: 540.2, prod: 1674.62, yield: 3100.0, state: 'West Bengal' },
    { area: 412.0, prod: 1359.60, yield: 3300.0, state: 'Andhra Pradesh' },
    { area: 258.4, prod: 1072.36, yield: 4150.0, state: 'Punjab' },
    { area: 182.1, prod: 746.61, yield: 4100.0, state: 'Punjab' },
    { area: 168.5, prod: 623.45, yield: 3700.0, state: 'Haryana' },
    { area: 175.2, prod: 578.16, yield: 3300.0, state: 'Tamil Nadu' },
    { area: 245.0, prod: 416.50, yield: 1700.0, state: 'Chhattisgarh' },
    { area: 260.0, prod: 442.00, yield: 1700.0, state: 'Madhya Pradesh' },
    { area: 180.4, prod: 360.80, yield: 2000.0, state: 'Orissa' },
    { area: 135.0, prod: 337.50, yield: 2500.0, state: 'Uttar Pradesh' },
    { area: 95.0, prod: 285.00, yield: 3000.0, state: 'Karnataka' },
    { area: 125.6, prod: 238.64, yield: 1900.0, state: 'Bihar' },
    { area: 110.2, prod: 231.42, yield: 2100.0, state: 'Assam' },
    { area: 82.5, prod: 231.00, yield: 2800.0, state: 'Kerala' },
    { area: 42.0, prod: 126.00, yield: 3000.0, state: 'Rajasthan' },
    { area: 38.0, prod: 64.60, yield: 1700.0, state: 'Himachal Pradesh' },
  ]

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{subtitle}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={scatterPoints} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
              <XAxis dataKey="area" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} unit="k ha" />
              <YAxis dataKey="prod" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} unit="k t" />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--card))',
                  borderColor: 'hsl(var(--border))',
                  borderRadius: '0.5rem',
                  fontSize: '12px'
                }}
                formatter={(value: any, name: any) => [
                  name === 'prod' ? `${formatNumber(Number(value))} '000 tons` : `${formatNumber(Number(value))} '000 ha`,
                  name === 'prod' ? 'Production' : 'Area'
                ]}
              />
              <Line type="monotone" dataKey="prod" stroke="#10b981" strokeWidth={0} dot={{ r: 5, fill: '#10b981' }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}
