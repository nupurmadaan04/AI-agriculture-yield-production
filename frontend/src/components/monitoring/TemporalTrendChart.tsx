import React, { useState } from 'react'
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ReferenceLine
} from 'recharts'
import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { TrendingUp, BarChart3 } from 'lucide-react'
import type { TemporalSignal } from '../../types/monitoring'

interface Props {
  data?: TemporalSignal
  isLoading?: boolean
  selectedMetric: string
  onMetricChange: (metric: string) => void
}

export const TemporalTrendChart: React.FC<Props> = ({
  data,
  isLoading,
  selectedMetric,
  onMetricChange,
}) => {
  if (isLoading) {
    return (
      <Card className="p-5 border border-slate-800 bg-slate-900/60 h-80 flex items-center justify-center animate-pulse">
        <span className="text-xs text-slate-500">Loading temporal monitoring trajectory...</span>
      </Card>
    )
  }

  const trajectory = data?.trajectory ?? []

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const pData = payload[0]?.payload
      return (
        <div className="bg-slate-900 border border-slate-700 p-3 rounded-xl shadow-xl text-xs space-y-1 z-50">
          <div className="font-bold text-slate-200">Year {label}</div>
          <div className="text-emerald-400 font-medium">
            Observed Value: {pData?.value?.toLocaleString()} {selectedMetric === 'yield' ? 'kg/ha' : 'k units'}
          </div>
          <div className="text-cyan-400">
            3-Yr Rolling Mean: {pData?.rolling_3yr?.toLocaleString()}
          </div>
          <div className="text-purple-400">
            Historical Baseline: {pData?.historical_mean?.toLocaleString()}
          </div>
          {pData?.yoy_change_pct !== undefined && (
            <div className={`text-[11px] font-semibold ${pData.yoy_change_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
              YoY Change: {pData.yoy_change_pct >= 0 ? '+' : ''}{pData.yoy_change_pct}%
            </div>
          )}
        </div>
      )
    }
    return null
  }

  return (
    <Card className="p-5 border border-slate-800 bg-slate-900/60 backdrop-blur space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold text-slate-100 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-emerald-400" />
              Temporal Trajectory & Rolling Dynamics
            </h3>
            <Badge variant="cyan" className="text-[10px]">
              {data?.location ?? 'State Baseline'}
            </Badge>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Chronological multi-period tracking across {data?.record_count ?? 0} annual observations
          </p>
        </div>

        {/* Metric Selector Pills */}
        <div className="flex items-center gap-1.5 p-1 bg-slate-950/60 rounded-lg border border-slate-800">
          {['yield', 'area', 'production'].map((m) => (
            <button
              key={m}
              onClick={() => onMetricChange(m)}
              className={`px-2.5 py-1 text-xs font-semibold rounded capitalize transition-all ${
                selectedMetric === m
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      {/* Trajectory Stats Summary */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1">
        <div className="p-2.5 bg-slate-950/40 rounded-lg border border-slate-800">
          <div className="text-[10px] text-slate-400">Latest Observation</div>
          <div className="text-sm font-bold text-slate-100 font-mono">
            {data?.latest_value?.toLocaleString()} {selectedMetric === 'yield' ? 'kg/ha' : 'k ha'}
          </div>
          <div className={`text-[10px] font-medium ${(data?.yoy_change_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
            {(data?.yoy_change_pct ?? 0) >= 0 ? '+' : ''}{data?.yoy_change_pct}% YoY
          </div>
        </div>

        <div className="p-2.5 bg-slate-950/40 rounded-lg border border-slate-800">
          <div className="text-[10px] text-slate-400">3-Yr Rolling Mean</div>
          <div className="text-sm font-bold text-cyan-300 font-mono">
            {data?.rolling_3yr_mean?.toLocaleString()}
          </div>
          <div className="text-[10px] text-slate-500">
            Z-score: {data?.rolling_3yr_zscore ?? 0.0} std
          </div>
        </div>

        <div className="p-2.5 bg-slate-950/40 rounded-lg border border-slate-800">
          <div className="text-[10px] text-slate-400">Annual Trend Slope</div>
          <div className="text-sm font-bold text-purple-300 font-mono">
            {(data?.trend_slope ?? 0) >= 0 ? '+' : ''}{data?.trend_slope} /yr
          </div>
          <div className="text-[10px] text-slate-500">
            Volatility (CV): {data?.volatility_cv}%
          </div>
        </div>

        <div className="p-2.5 bg-slate-950/40 rounded-lg border border-slate-800">
          <div className="text-[10px] text-slate-400">Historical Deviation</div>
          <div className={`text-sm font-bold font-mono ${(data?.deviation_from_historical ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
            {(data?.deviation_from_historical ?? 0) >= 0 ? '+' : ''}{data?.deviation_from_historical}%
          </div>
          <div className="text-[10px] text-slate-500">
            Mean: {data?.historical_mean?.toLocaleString()}
          </div>
        </div>
      </div>

      {/* Composed Chart */}
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={trajectory} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="year" stroke="#64748b" fontSize={11} />
            <YAxis stroke="#64748b" fontSize={11} domain={['auto', 'auto']} />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />

            <ReferenceLine
              y={data?.historical_mean}
              stroke="#a855f7"
              strokeDasharray="4 4"
              label={{ value: 'Historical Mean', fill: '#c084fc', fontSize: 10, position: 'insideTopLeft' }}
            />

            <Line
              type="monotone"
              dataKey="value"
              name="Observed Value"
              stroke="#10b981"
              strokeWidth={2.5}
              dot={{ r: 3, fill: '#10b981' }}
              activeDot={{ r: 5 }}
            />

            <Line
              type="monotone"
              dataKey="rolling_3yr"
              name="3-Yr Rolling Mean"
              stroke="#06b6d4"
              strokeWidth={1.8}
              strokeDasharray="3 3"
              dot={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </Card>
  )
}
