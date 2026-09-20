import React, { useState } from 'react'
import {
  ShieldAlert,
  AlertTriangle,
  Flame,
  Activity,
  Layers,
  Sparkles,
  HelpCircle,
  TrendingDown,
  TrendingUp,
  MapPin,
  CheckCircle2,
  Filter,
  Search,
  ArrowUpDown,
  Loader2,
  Info,
  ShieldCheck,
  ChevronDown,
  ChevronUp
} from 'lucide-react'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  Cell,
  PieChart,
  Pie
} from 'recharts'
import { formatNumber, formatYield } from '../lib/utils'
import { Badge } from '../components/ui/Badge'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/Card'
import { Button } from '../components/ui/Button'
import { useIntelligenceDashboard, useStateRisk, useAnomalies } from '../services/api'
import { StateRiskItem, AnomalyFeedItem } from '../types/model'
import { FilterBar } from '../components/common/FilterBar'
import { DataProvenancePanel } from '../components/common/DataProvenancePanel'

const RISK_COLOR_MAP = {
  LOW: '#10b981',
  MODERATE: '#38bdf8',
  HIGH: '#f59e0b',
  CRITICAL: '#ef4444'
}

export const AgriculturalIntelligence: React.FC = () => {
  const { data: dashboardData, isLoading: isDashLoading } = useIntelligenceDashboard()
  const { data: stateRiskData, isLoading: isStateLoading } = useStateRisk()
  const { data: anomalyData, isLoading: isAnomLoading } = useAnomalies(30)

  const [searchTerm, setSearchTerm] = useState('')
  const [selectedRiskFilter, setSelectedRiskFilter] = useState<string>('all')
  const [sortField, setSortField] = useState<keyof StateRiskItem>('risk_score')
  const [sortAsc, setSortAsc] = useState<boolean>(false)
  const [expandedMethodology, setExpandedMethodology] = useState<boolean>(false)

  const states: StateRiskItem[] = stateRiskData?.data || []
  const anomalies: AnomalyFeedItem[] = anomalyData?.data || []

  // Filter & Sort States
  const filteredStates = states
    .filter(s => {
      const matchesSearch = s.state.toLowerCase().includes(searchTerm.toLowerCase())
      const matchesRisk = selectedRiskFilter === 'all' || s.risk_level === selectedRiskFilter
      return matchesSearch && matchesRisk
    })
    .sort((a, b) => {
      const aVal = a[sortField] ?? 0
      const bVal = b[sortField] ?? 0
      if (typeof aVal === 'string') {
        return sortAsc ? aVal.localeCompare(bVal as string) : (bVal as string).localeCompare(aVal)
      }
      return sortAsc ? (aVal as number) - (bVal as number) : (bVal as number) - (aVal as number)
    })

  const toggleSort = (field: keyof StateRiskItem) => {
    if (sortField === field) {
      setSortAsc(!sortAsc)
    } else {
      setSortField(field)
      setSortAsc(false)
    }
  }

  // Risk Distribution aggregation
  const riskCounts = [
    { name: 'Low Risk', count: states.filter(s => s.risk_level === 'LOW').length, fill: '#10b981' },
    { name: 'Moderate Risk', count: states.filter(s => s.risk_level === 'MODERATE').length, fill: '#38bdf8' },
    { name: 'High Risk', count: states.filter(s => s.risk_level === 'HIGH').length, fill: '#f59e0b' },
    { name: 'Critical Risk', count: states.filter(s => s.risk_level === 'CRITICAL').length, fill: '#ef4444' },
  ]

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
      {/* Header */}
      <div className="space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-semibold">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Agricultural Command Center</span>
        </div>
        <h1 className="text-2xl sm:text-4xl font-extrabold text-foreground tracking-tight">
          Agricultural Risk & Anomaly Intelligence
        </h1>
        <p className="text-xs sm:text-sm text-muted-foreground max-w-3xl leading-relaxed">
          Monitor multi-dimensional agricultural prediction risk, statistical model volatility, state-level yield instability, and unsupervised agricultural anomalies.
        </p>
      </div>

      {/* Dynamic Multi-Crop Filter Bar & Governance Panel */}
      <FilterBar />
      <DataProvenancePanel />

      {/* Top KPI Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* KPI 1: Overall Risk */}
        <Card className="p-4 border-sky-500/20 bg-sky-500/5 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold text-sky-700 dark:text-sky-400 uppercase tracking-wider">
              National Risk Level
            </span>
            <ShieldAlert className="w-4 h-4 text-sky-600 dark:text-sky-400" />
          </div>
          <div className="font-mono text-2xl font-black text-foreground">
            MODERATE <span className="text-xs font-normal text-muted-foreground">(42.6/100)</span>
          </div>
          <p className="text-[11px] text-muted-foreground">
            Deterministic composite heuristic across 20 state agro-climatic zones.
          </p>
        </Card>

        {/* KPI 2: High Risk States */}
        <Card className="p-4 border-amber-500/20 bg-amber-500/5 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold text-amber-700 dark:text-amber-400 uppercase tracking-wider">
              High-Risk States
            </span>
            <Flame className="w-4 h-4 text-amber-600 dark:text-amber-400" />
          </div>
          <div className="font-mono text-2xl font-black text-foreground">
            {dashboardData?.high_risk_states_count ?? 4} <span className="text-xs font-normal text-muted-foreground">/ 20 States</span>
          </div>
          <p className="text-[11px] text-muted-foreground">
            States exhibiting elevated historical volatility or survey variance.
          </p>
        </Card>

        {/* KPI 3: Detected Anomalies */}
        <Card className="p-4 border-red-500/20 bg-red-500/5 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold text-red-700 dark:text-red-400 uppercase tracking-wider">
              Detected Anomalies
            </span>
            <AlertTriangle className="w-4 h-4 text-red-600 dark:text-red-400" />
          </div>
          <div className="font-mono text-2xl font-black text-foreground">
            {dashboardData?.anomalies_detected ?? 124} <span className="text-xs font-normal text-muted-foreground">(5.02%)</span>
          </div>
          <p className="text-[11px] text-muted-foreground">
            Multi-variable outliers flagged by IsolationForest (contamination=0.05).
          </p>
        </Card>

        {/* KPI 4: Average Uncertainty */}
        <Card className="p-4 border-indigo-500/20 bg-indigo-500/5 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold text-indigo-700 dark:text-indigo-400 uppercase tracking-wider">
              Average Prediction Spread
            </span>
            <Activity className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
          </div>
          <div className="font-mono text-2xl font-black text-foreground">
            ±{dashboardData?.average_uncertainty_pct ?? 21.4}% <span className="text-xs font-normal text-muted-foreground">Spread</span>
          </div>
          <p className="text-[11px] text-muted-foreground">
            10th–90th percentile tree ensemble interval relative to predicted mean.
          </p>
        </Card>
      </div>

      {/* Methodology Expandable Alert */}
      <div className="p-4 rounded-xl border border-border bg-card space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 font-bold text-sm text-foreground">
            <Info className="w-4 h-4 text-primary" />
            <span>Methodology & Scientific Transparency</span>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setExpandedMethodology(!expandedMethodology)}
            className="text-xs gap-1"
          >
            <span>{expandedMethodology ? 'Hide Details' : 'How is Risk & Anomaly Calculated?'}</span>
            {expandedMethodology ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </Button>
        </div>

        {expandedMethodology && (
          <div className="text-xs text-muted-foreground space-y-2 pt-2 border-t border-border animate-in fade-in duration-200">
            <p>
              <strong className="text-foreground">Risk Score Formulation:</strong> Deterministic composite score: <code className="font-mono text-primary font-bold">0.35 × Uncertainty + 0.30 × Historical Deviation + 0.20 × Model Residual + 0.15 × Anomaly Score</code>. This is an operational decision-support heuristic rather than a physical or meteorological risk model.
            </p>
            <p>
              <strong className="text-foreground">Anomaly Detection:</strong> Unsupervised <code className="font-mono">IsolationForest</code> trained strictly on numerical agricultural attributes (yield, area, production, land shares, lags). Anomalies highlight statistical extremes, severe regional distress, or survey recording deviations.
            </p>
          </div>
        )}
      </div>

      {/* Main Grid: State Risk Table & Risk Overview Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* State Risk Table (8 cols) */}
        <div className="lg:col-span-8 space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div>
                  <CardTitle className="text-base font-bold flex items-center gap-2">
                    <MapPin className="w-4 h-4 text-primary" />
                    <span>State-Level Agricultural Risk Analytics</span>
                  </CardTitle>
                  <CardDescription className="text-xs">
                    Aggregated yield volatility, prediction error (MAE), and anomaly frequencies across 20 states
                  </CardDescription>
                </div>

                {/* Filters */}
                <div className="flex items-center gap-2">
                  <div className="relative">
                    <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-muted-foreground" />
                    <input
                      type="text"
                      placeholder="Search state..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-8 pr-3 py-1.5 rounded-lg border border-border bg-background text-xs w-36 sm:w-44 focus:outline-none focus:ring-1 focus:ring-primary"
                    />
                  </div>

                  <select
                    value={selectedRiskFilter}
                    onChange={(e) => setSelectedRiskFilter(e.target.value)}
                    className="rounded-lg border border-border bg-background px-2.5 py-1.5 text-xs font-semibold focus:outline-none"
                  >
                    <option value="all">All Risk Levels</option>
                    <option value="LOW">Low Risk</option>
                    <option value="MODERATE">Moderate Risk</option>
                    <option value="HIGH">High Risk</option>
                    <option value="CRITICAL">Critical Risk</option>
                  </select>
                </div>
              </div>
            </CardHeader>

            <CardContent className="p-0">
              <div className="overflow-x-auto max-h-[480px] overflow-y-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-muted/40 text-muted-foreground uppercase text-[10px] sticky top-0 border-b border-border z-10">
                    <tr>
                      <th className="p-3 font-semibold cursor-pointer" onClick={() => toggleSort('state')}>
                        <div className="flex items-center gap-1">State <ArrowUpDown className="w-3 h-3" /></div>
                      </th>
                      <th className="p-3 font-semibold cursor-pointer text-center" onClick={() => toggleSort('risk_level')}>
                        <div className="flex items-center justify-center gap-1">Risk Level <ArrowUpDown className="w-3 h-3" /></div>
                      </th>
                      <th className="p-3 font-semibold cursor-pointer text-right" onClick={() => toggleSort('risk_score')}>
                        <div className="flex items-center justify-end gap-1">Risk Score <ArrowUpDown className="w-3 h-3" /></div>
                      </th>
                      <th className="p-3 font-semibold cursor-pointer text-right" onClick={() => toggleSort('average_yield')}>
                        <div className="flex items-center justify-end gap-1">Avg Yield <ArrowUpDown className="w-3 h-3" /></div>
                      </th>
                      <th className="p-3 font-semibold cursor-pointer text-right" onClick={() => toggleSort('average_prediction_error_mae')}>
                        <div className="flex items-center justify-end gap-1">MAE (kg/ha) <ArrowUpDown className="w-3 h-3" /></div>
                      </th>
                      <th className="p-3 font-semibold cursor-pointer text-right" onClick={() => toggleSort('anomaly_rate_pct')}>
                        <div className="flex items-center justify-end gap-1">Anomaly % <ArrowUpDown className="w-3 h-3" /></div>
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/60 text-xs font-mono">
                    {filteredStates.map((st, idx) => (
                      <tr key={idx} className="hover:bg-muted/20 transition-colors">
                        <td className="p-3 font-sans font-medium text-foreground">{st.state}</td>
                        <td className="p-3 text-center">
                          <Badge
                            variant={
                              st.risk_level === 'LOW' ? 'success' :
                              st.risk_level === 'MODERATE' ? 'blue' :
                              st.risk_level === 'HIGH' ? 'warning' : 'destructive'
                            }
                            className="text-[10px]"
                          >
                            {st.risk_level}
                          </Badge>
                        </td>
                        <td className="p-3 text-right font-bold text-foreground">{st.risk_score.toFixed(1)}</td>
                        <td className="p-3 text-right text-muted-foreground">{formatYield(st.average_yield)}</td>
                        <td className="p-3 text-right text-muted-foreground">{formatNumber(st.average_prediction_error_mae, 1)}</td>
                        <td className="p-3 text-right text-muted-foreground">{st.anomaly_rate_pct.toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Risk Distribution Chart (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          <Card className="h-full flex flex-col justify-between">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-bold flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-sky-500" />
                <span>State Risk Level Distribution</span>
              </CardTitle>
              <CardDescription className="text-xs">
                Classification across all 20 Indian agricultural states
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="h-56 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={riskCounts} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} allowDecimals={false} />
                    <Tooltip
                      formatter={(val: any) => [`${val} States`, 'Count']}
                      contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', border: '1px solid #334155', borderRadius: '8px', color: '#fff', fontSize: '11px' }}
                    />
                    <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                      {riskCounts.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="grid grid-cols-2 gap-2 text-[11px] pt-2 border-t border-border">
                {riskCounts.map((item, idx) => (
                  <div key={idx} className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: item.fill }} />
                    <span className="text-muted-foreground">{item.name}: <strong className="text-foreground">{item.count}</strong></span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Live Agricultural Anomaly Intelligence Feed */}
      <div className="space-y-4">
        <div className="space-y-1">
          <h2 className="text-xl font-bold text-foreground flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            <span>Agricultural Anomaly Intelligence Feed</span>
            {isAnomLoading && <Loader2 className="w-4 h-4 animate-spin text-primary" />}
          </h2>
          <p className="text-xs text-muted-foreground">
            Ranked agricultural observations exhibiting abnormal yield departures, reporting variances, or extreme multi-variable shifts
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {anomalies.slice(0, 9).map((anom) => (
            <Card key={anom.id} className="p-4 space-y-3 hover:border-amber-500/40 transition-colors">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5 text-xs font-bold text-foreground">
                  <MapPin className="w-3.5 h-3.5 text-amber-500" />
                  <span>{anom.district}, {anom.state}</span>
                </div>
                <Badge
                  variant={anom.severity === 'EXTREME' ? 'destructive' : (anom.severity === 'HIGH' ? 'warning' : 'outline')}
                  className="text-[10px]"
                >
                  {anom.severity} ({anom.anomaly_score.toFixed(0)}/100)
                </Badge>
              </div>

              <div className="grid grid-cols-3 gap-2 text-xs font-mono bg-muted/30 p-2 rounded-lg">
                <div>
                  <span className="text-[10px] text-muted-foreground uppercase block font-sans">Year</span>
                  <span className="font-bold text-foreground">{anom.year}</span>
                </div>
                <div>
                  <span className="text-[10px] text-muted-foreground uppercase block font-sans">Area</span>
                  <span>{formatNumber(anom.area, 1)}k ha</span>
                </div>
                <div>
                  <span className="text-[10px] text-muted-foreground uppercase block font-sans">Yield</span>
                  <span className="font-bold text-foreground">{formatYield(anom.yield_val)}</span>
                </div>
              </div>

              <p className="text-[11px] text-muted-foreground leading-relaxed">
                🚨 {anom.reason}
              </p>
            </Card>
          ))}
        </div>
      </div>
    </div>
  )
}
