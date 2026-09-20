import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import {
  ShieldAlert,
  Sparkles,
  TrendingUp,
  AlertTriangle,
  FileText,
  Activity,
  Award,
  Layers,
  MapPin,
  Bot,
  Sliders,
  CheckCircle2,
  Loader2,
  ChevronRight,
  Clock,
  Info
} from 'lucide-react'
import { useDecisionSupport, useStatesEarlyWarning, useStatesTrends } from '../services/api'
import { formatYield, formatNumber } from '../lib/utils'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { Button } from '../components/ui/Button'
import { IntelligenceReportModal } from '../components/intelligence/IntelligenceReport'
import { OutlookPanel } from '../components/temporal/OutlookPanel'

export const DecisionSupport: React.FC = () => {
  const { data, isLoading, error } = useDecisionSupport()
  const { data: warningData } = useStatesEarlyWarning()
  const { data: trendsData } = useStatesTrends()

  const [reportModalOpen, setReportModalOpen] = useState(false)
  const [modalState, setModalState] = useState<string | null>(null)

  if (isLoading) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-20 text-center space-y-4">
        <Loader2 className="w-8 h-8 animate-spin mx-auto text-sky-500" />
        <p className="text-sm font-semibold text-muted-foreground">Synthesizing Decision Intelligence Executive Dashboard...</p>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-20 text-center space-y-4">
        <AlertTriangle className="w-8 h-8 text-amber-500 mx-auto" />
        <p className="text-sm text-foreground font-bold">Failed to load Decision Support dashboard.</p>
      </div>
    )
  }

  const kpis = data.kpis
  const situation = data.regional_situation
  const signals = data.model_signals
  const anomalies = data.recent_anomalies
  const warningList = warningData?.data || []
  const topWarnings = warningList.slice(0, 5)

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="space-y-2">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-500/10 text-sky-600 dark:text-sky-400 text-xs font-semibold">
            <Award className="w-3.5 h-3.5" />
            <span>Executive Agricultural Decision Intelligence</span>
          </div>
          <h1 className="text-2xl sm:text-4xl font-extrabold text-foreground tracking-tight">
            Agricultural Decision Support Center
          </h1>
          <p className="text-xs sm:text-sm text-muted-foreground max-w-3xl leading-relaxed">
            Consolidated intelligence summary across state productivity baselines, forward forecasts, model feature signals, anomaly detections, and scenario interventions.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Button
            onClick={() => setReportModalOpen(true)}
            className="gap-2 text-xs font-bold shadow-md shadow-sky-500/20"
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Generate Intelligence Report</span>
          </Button>
        </div>
      </div>

      {/* SECTION 1: TOP KPI ROW */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="p-4 border-border bg-card/80 space-y-1">
          <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">
            Overall National Risk Rating
          </span>
          <div className="text-2xl font-black text-foreground font-mono">
            {kpis.high_risk_states_count > 0 ? 'MODERATE' : 'LOW'}
          </div>
          <p className="text-[11px] text-muted-foreground">
            {kpis.low_risk_states_count} Low Risk • {kpis.moderate_risk_states_count} Moderate • {kpis.high_risk_states_count} High Risk States
          </p>
        </Card>

        <Card className="p-4 border-amber-500/30 bg-amber-500/5 space-y-1">
          <span className="text-[10px] font-bold text-amber-700 dark:text-amber-400 uppercase tracking-wider">
            High-Risk State Regions
          </span>
          <div className="text-2xl font-black text-amber-600 dark:text-amber-400 font-mono">
            {kpis.high_risk_states_count} States
          </div>
          <p className="text-[11px] text-muted-foreground">
            Regions requiring active monitoring due to yield volatility.
          </p>
        </Card>

        <Card className="p-4 border-sky-500/30 bg-sky-500/5 space-y-1">
          <span className="text-[10px] font-bold text-sky-700 dark:text-sky-400 uppercase tracking-wider">
            Agricultural Anomalies Flagged
          </span>
          <div className="text-2xl font-black text-sky-600 dark:text-sky-400 font-mono">
            {kpis.anomalies_detected} Outliers
          </div>
          <p className="text-[11px] text-muted-foreground">
            5.02% of total panel records flagged by Isolation Forest.
          </p>
        </Card>

        <Card className="p-4 border-border bg-card/80 space-y-1">
          <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">
            Average Forecast Spread
          </span>
          <div className="text-2xl font-black text-foreground font-mono">
            ±{kpis.average_uncertainty_pct.toFixed(1)}%
          </div>
          <p className="text-[11px] text-muted-foreground">
            Ensemble tree dispersion across regional validation.
          </p>
        </Card>
      </div>

      {/* SECTION 2: AI GROUNDED EXECUTIVE INSIGHT & TEMPORAL OUTLOOK */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-8">
          <Card className="p-5 border-sky-500/30 bg-sky-500/5 space-y-2 h-full">
            <div className="flex items-center gap-2 text-sky-700 dark:text-sky-300 font-bold text-xs">
              <Bot className="w-4 h-4" />
              <span>Automated Copilot Intelligence Briefing</span>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              {data.ai_insight}
            </p>
          </Card>
        </div>

        <div className="lg:col-span-4">
          <Card className="p-4 border-amber-500/30 bg-amber-500/5 space-y-2 h-full flex flex-col justify-between">
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-amber-700 dark:text-amber-400 font-bold text-xs">
                <Clock className="w-4 h-4" />
                <span>Temporal Forecast Outlook</span>
              </div>
              <p className="text-[11px] text-muted-foreground">
                Exogenous Multi-Horizon Random Forest Forecaster active with out-of-time chronological R² = 0.7866.
              </p>
            </div>
            <Link to="/early-warning">
              <Button size="sm" variant="outline" className="w-full text-xs font-bold gap-1 mt-2">
                <span>View Full Forecasting Hub</span>
                <ChevronRight className="w-3 h-3" />
              </Button>
            </Link>
          </Card>
        </div>
      </div>

      {/* SECTION 3: EARLY WARNING & REGIONAL SITUATION MATRIX */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Regional Situation Table (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-bold text-foreground flex items-center gap-2">
              <MapPin className="w-4 h-4 text-sky-500" />
              <span>Regional Situation Matrix</span>
            </h2>
            <Link to="/early-warning" className="text-xs text-sky-600 dark:text-sky-400 hover:underline flex items-center gap-0.5">
              <span>View Temporal Hub</span>
              <ChevronRight className="w-3 h-3" />
            </Link>
          </div>

          <Card className="border-border overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-border bg-muted/40 text-muted-foreground">
                    <th className="py-2.5 px-3 font-semibold">State</th>
                    <th className="py-2.5 px-3 font-semibold">Risk Rating</th>
                    <th className="py-2.5 px-3 font-semibold">Score</th>
                    <th className="py-2.5 px-3 font-semibold">Avg Yield</th>
                    <th className="py-2.5 px-3 font-semibold">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/60">
                  {situation.map((item, idx) => (
                    <tr
                      key={idx}
                      onClick={() => setModalState(item.state)}
                      className="hover:bg-muted/20 cursor-pointer"
                    >
                      <td className="py-2.5 px-3 font-semibold text-foreground">{item.state}</td>
                      <td className="py-2.5 px-3">
                        <Badge
                          variant={item.risk_level === 'LOW' ? 'success' : item.risk_level === 'MODERATE' ? 'blue' : 'warning'}
                          className="text-[9px]"
                        >
                          {item.risk_level}
                        </Badge>
                      </td>
                      <td className="py-2.5 px-3 font-mono font-bold">{item.risk_score.toFixed(1)}</td>
                      <td className="py-2.5 px-3 font-mono">{formatYield(item.avg_yield)}</td>
                      <td className="py-2.5 px-3 text-sky-600 dark:text-sky-400 font-semibold hover:underline">
                        Inspect →
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>

        {/* Early Warning Watchlist (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-bold text-foreground flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-amber-500" />
              <span>Early Warning Priority Watchlist</span>
            </h2>
            <Link to="/early-warning" className="text-xs text-sky-600 dark:text-sky-400 hover:underline">
              All 20 States →
            </Link>
          </div>

          <Card className="border-border p-3 space-y-2">
            {topWarnings.map((w: any, idx: number) => (
              <div
                key={idx}
                onClick={() => setModalState(w.state)}
                className="p-3 rounded-lg border border-border bg-card/60 hover:bg-muted/40 cursor-pointer transition-colors space-y-1 text-xs"
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-foreground">{w.state}</span>
                  <Badge
                    variant={w.severity === 'CRITICAL' ? 'destructive' : w.severity === 'HIGH' ? 'warning' : 'blue'}
                    className="text-[9px]"
                  >
                    {w.warning_score.toFixed(1)} ({w.severity})
                  </Badge>
                </div>
                <div className="flex items-center justify-between text-[11px] text-muted-foreground font-mono">
                  <span>Trend: {w.trend_direction} ({w.trend_slope_kg_ha_yr > 0 ? `+${w.trend_slope_kg_ha_yr.toFixed(1)}` : w.trend_slope_kg_ha_yr.toFixed(1)} kg/ha/yr)</span>
                  <span>1-Yr: {formatYield(w.forecast_1yr_kg_ha)}</span>
                </div>
              </div>
            ))}
          </Card>
        </div>
      </div>

      {/* Spatial Situation GIS Banner */}
      <Card className="border-border bg-gradient-to-r from-emerald-950/30 via-slate-900/40 to-sky-950/30 p-5">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <MapPin className="w-5 h-5 text-emerald-400" />
              <h3 className="text-base font-bold text-foreground">Spatial Situation & GIS Topology</h3>
              <Badge variant="blue" className="text-[10px]">20 States • 311 Districts</Badge>
            </div>
            <p className="text-xs text-muted-foreground max-w-2xl leading-relaxed">
              Explore spatial clustering archetypes, within-state outlier departures (|z| ≥ 1.8σ), anomaly concentrations, and multi-horizon regional forecast maps in the GIS Command Center.
            </p>
          </div>
          <Link to="/geospatial">
            <Button variant="outline" className="text-xs border-emerald-500/40 text-emerald-400 hover:bg-emerald-500/10">
              <span>Open Geospatial Command Center →</span>
            </Button>
          </Link>
        </div>
      </Card>

      {/* Model Reliability & Validation Status Banner */}
      <Card className="border-border bg-gradient-to-r from-blue-950/30 via-slate-900/40 to-indigo-950/30 p-5">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <ShieldAlert className="w-5 h-5 text-blue-400" />
              <h3 className="text-base font-bold text-foreground">Model Reliability & Out-of-Time Validation</h3>
              <Badge variant="success" className="text-[10px]">Audited & Grounded</Badge>
            </div>
            <p className="text-xs text-muted-foreground max-w-2xl leading-relaxed">
              Model performance is evaluated using chronological out-of-time observations from 2016–2017 (Test R² = 0.7866, MAE = 353.0 kg/ha). Continuous monitoring tracks feature drift (PSI) and 4-pillar data quality (96.5/100).
            </p>
          </div>
          <Link to="/model-reliability">
            <Button variant="outline" className="text-xs border-blue-500/40 text-blue-400 hover:bg-blue-500/10">
              <span>Inspect Validation Diagnostics →</span>
            </Button>
          </Link>
        </div>
      </Card>

      {/* Scenario-Based Agricultural Decision Intelligence Banner */}
      <Card className="border-border bg-gradient-to-r from-purple-950/30 via-slate-900/40 to-emerald-950/30 p-5">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-purple-400" />
              <h3 className="text-base font-bold text-foreground">Scenario Simulation & Pareto Decision Optimization</h3>
              <Badge variant="purple" className="text-[10px]">Multi-Objective Solver Active</Badge>
            </div>
            <p className="text-xs text-muted-foreground max-w-2xl leading-relaxed">
              Evaluate hypothetical agricultural interventions (crop reallocation, yield shocks) against immutable baselines. Explore non-dominated Pareto alternatives balancing yield gain, risk mitigation, and resource shifts.
            </p>
          </div>
          <Link to="/scenario">
            <Button variant="outline" className="text-xs border-purple-500/40 text-purple-400 hover:bg-purple-500/10">
              <span>Explore Scenario Intelligence →</span>
            </Button>
          </Link>
        </div>
      </Card>

      {/* Scientific Disclaimer */}
      <div className="p-4 rounded-xl border border-border/80 bg-muted/20 text-xs text-muted-foreground text-center">
        <strong>Notice:</strong> Forecasts represent model-estimated future values and should not be interpreted as causal predictions or biological guarantees.
      </div>

      {/* Report Modal */}
      <IntelligenceReportModal
        isOpen={reportModalOpen}
        onClose={() => setReportModalOpen(false)}
      />

      {/* State Drilldown Modal */}
      {modalState && (
        <OutlookPanel
          stateName={modalState}
          isOpen={Boolean(modalState)}
          onClose={() => setModalState(null)}
        />
      )}
    </div>
  )
}
