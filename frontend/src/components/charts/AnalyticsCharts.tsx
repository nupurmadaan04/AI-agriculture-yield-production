import React from 'react'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell
} from 'recharts'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../ui/Card'
import { formatNumber } from '../../lib/utils'

interface StatePerformanceBarProps {
  data: Array<{
    state?: string
    stateName?: string
    avgYield?: number
    average_yield?: number
  }>
  title?: string
  subtitle?: string
}

export const StatePerformanceBar: React.FC<StatePerformanceBarProps> = ({
  data,
  title = "Average Rice Yield by State",
  subtitle = "State rankings sorted by average yield in kg/ha (2010–2017)"
}) => {
  const normalizedData = data.map(d => ({
    name: d.state || d.stateName || 'State',
    yieldVal: d.average_yield !== undefined ? d.average_yield : (d.avgYield !== undefined ? d.avgYield : 0)
  }))

  const sortedData = [...normalizedData].sort((a, b) => b.yieldVal - a.yieldVal).slice(0, 10)

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{subtitle}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={sortedData} layout="vertical" margin={{ top: 5, right: 20, left: 40, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="hsl(var(--border))" />
              <XAxis type="number" stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} unit=" kg/ha" />
              <YAxis type="category" dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--card))',
                  borderColor: 'hsl(var(--border))',
                  borderRadius: '0.5rem',
                  fontSize: '12px'
                }}
                formatter={(value: any) => [`${formatNumber(Number(value), 1)} kg/ha`, 'Avg Yield']}
              />
              <Bar dataKey="yieldVal" radius={[0, 4, 4, 0]}>
                {sortedData.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={index < 3 ? '#0284c7' : '#38bdf8'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}

export const YieldDistributionChart: React.FC = () => {
  const distributionData = [
    { range: '0 kg/ha (Zero/Failure)', count: 236 },
    { range: '1–1000 kg/ha', count: 215 },
    { range: '1000–2000 kg/ha', count: 820 },
    { range: '2000–3000 kg/ha', count: 864 },
    { range: '3000–4000 kg/ha', count: 278 },
    { range: '>4000 kg/ha', count: 56 },
  ]

  return (
    <Card>
      <CardHeader>
        <CardTitle>Yield Distribution Across Records</CardTitle>
        <CardDescription>Histogram of district-year observations in ICRISAT dataset (2,469 samples)</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={distributionData} margin={{ top: 10, right: 10, left: -10, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
              <XAxis dataKey="range" stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} angle={-15} textAnchor="end" />
              <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--card))',
                  borderColor: 'hsl(var(--border))',
                  borderRadius: '0.5rem',
                  fontSize: '12px'
                }}
                formatter={(value: any) => [`${value} districts`, 'Observations']}
              />
              <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}

export const CropComparisonChart: React.FC = () => {
  const comparisonData = [
    { metric: 'Avg Yield (kg/ha)', Rice: 2062.8, Wheat: 3120.0, Maize: 2450.0 },
    { metric: 'Harvest Duration (days)', Rice: 120, Wheat: 140, Maize: 95 },
    { metric: 'Water Need (mm)', Rice: 1250, Wheat: 450, Maize: 500 },
  ]

  return (
    <Card>
      <CardHeader>
        <CardTitle>Multi-Crop Benchmark Profile</CardTitle>
        <CardDescription>Comparative agronomic indicators across Indian primary staples</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={comparisonData} margin={{ top: 10, right: 10, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
              <XAxis dataKey="metric" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} />
              <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--card))',
                  borderColor: 'hsl(var(--border))',
                  borderRadius: '0.5rem',
                  fontSize: '12px'
                }}
              />
              <Bar dataKey="Rice" fill="#0284c7" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Wheat" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Maize" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}
