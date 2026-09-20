import React from 'react'
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  ReferenceLine
} from 'recharts'
import { HistoricalSeriesPoint, HorizonForecastItem } from '../../types/temporal'
import { formatYield } from '../../lib/utils'

interface ForecastChartProps {
  historicalSeries: HistoricalSeriesPoint[]
  forecasts: HorizonForecastItem[]
  stateName: string
  districtName?: string
}

export const ForecastChart: React.FC<ForecastChartProps> = ({
  historicalSeries,
  forecasts,
  stateName,
  districtName
}) => {
  // Combine historical and forecast data points
  const lastObserved = historicalSeries.length > 0 ? historicalSeries[historicalSeries.length - 1] : null
  const splitYear = lastObserved ? lastObserved.year : 2017

  const chartData: any[] = []

  // Add historical points
  historicalSeries.forEach(pt => {
    chartData.push({
      year: pt.year,
      observedYield: pt.yield,
      forecastYield: null,
      lowerBound: null,
      upperBound: null,
      spreadRange: null,
      isForecast: false
    })
  })

  // Bridge point (connecting historical to forecast)
  if (lastObserved && forecasts.length > 0) {
    const bridge = chartData.find(d => d.year === lastObserved.year)
    if (bridge) {
      bridge.forecastYield = lastObserved.yield
      bridge.lowerBound = lastObserved.yield
      bridge.upperBound = lastObserved.yield
    }
  }

  // Add forecast points
  forecasts.forEach(fc => {
    chartData.push({
      year: fc.forecast_year,
      observedYield: null,
      forecastYield: fc.predicted_yield,
      lowerBound: fc.lower_bound_p10,
      upperBound: fc.upper_bound_p90,
      spreadRange: [fc.lower_bound_p10, fc.upper_bound_p90],
      isForecast: true
    })
  })

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between text-xs">
        <div>
          <span className="font-bold text-foreground">
            {stateName} {districtName && districtName !== 'Regional Average' ? `(${districtName})` : ''} Trajectory & 3-Year Outlook
          </span>
          <p className="text-[11px] text-muted-foreground">
            Solid line indicates verified ICRISAT historical records; dashed line denotes Random Forest forecast with P10–P90 spread.
          </p>
        </div>
      </div>

      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-border/40" />
            <XAxis
              dataKey="year"
              stroke="currentColor"
              className="text-muted-foreground text-[11px]"
              tickLine={false}
            />
            <YAxis
              stroke="currentColor"
              className="text-muted-foreground text-[11px]"
              tickFormatter={(v) => `${v}`}
              domain={['auto', 'auto']}
              tickLine={false}
            />
            <Tooltip
              content={({ active, payload, label }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload
                  return (
                    <div className="rounded-lg border border-border bg-popover/95 p-3 shadow-md text-xs backdrop-blur-sm space-y-1">
                      <div className="font-bold text-foreground">Year {label} {data.isForecast ? '(Forecast)' : '(Observed)'}</div>
                      {data.observedYield !== null && (
                        <div className="text-sky-600 dark:text-sky-400 font-mono">
                          Observed Yield: <strong>{formatYield(data.observedYield)}</strong>
                        </div>
                      )}
                      {data.forecastYield !== null && (
                        <div className="text-emerald-600 dark:text-emerald-400 font-mono">
                          Model Forecast: <strong>{formatYield(data.forecastYield)}</strong>
                        </div>
                      )}
                      {data.lowerBound !== null && data.upperBound !== null && (
                        <div className="text-[10px] text-muted-foreground font-mono">
                          P10–P90 Spread: {formatYield(data.lowerBound)} – {formatYield(data.upperBound)}
                        </div>
                      )}
                    </div>
                  )
                }
                return null
              }}
            />
            <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
            
            <ReferenceLine
              x={splitYear}
              stroke="#f59e0b"
              strokeDasharray="4 4"
              label={{ value: 'Forecast Horizon', position: 'top', fill: '#f59e0b', fontSize: 10 }}
            />

            {/* Prediction Interval Ribbon */}
            <Area
              type="monotone"
              dataKey="upperBound"
              stroke="none"
              fill="#10b981"
              fillOpacity={0.15}
              name="Prediction Spread (P10–P90)"
            />

            {/* Historical Observed Line */}
            <Line
              type="monotone"
              dataKey="observedYield"
              stroke="#0284c7"
              strokeWidth={3}
              dot={{ r: 4, fill: '#0284c7' }}
              activeDot={{ r: 6 }}
              name="Historical Reported Yield"
              connectNulls={false}
            />

            {/* Model Forecast Line */}
            <Line
              type="monotone"
              dataKey="forecastYield"
              stroke="#10b981"
              strokeWidth={2.5}
              strokeDasharray="5 5"
              dot={{ r: 4, fill: '#10b981' }}
              name="Exogenous RF Forecast"
              connectNulls={true}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
