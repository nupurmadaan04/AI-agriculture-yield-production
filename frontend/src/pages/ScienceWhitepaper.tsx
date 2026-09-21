import React from 'react'
import {
  ShieldCheck,
  FileText,
  Download,
  AlertTriangle,
  CheckCircle2,
  Cpu,
  BarChart3,
  ExternalLink,
  Layers,
  Sparkles,
  Loader2,
  TrendingDown,
  Activity,
  MapPin,
  Flame,
  Award,
  Zap,
  ShieldAlert,
  Sliders,
  HelpCircle,
  Clock,
  Compass,
  Target
} from 'lucide-react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid
} from 'recharts'
import { formatNumber, formatYield } from '../lib/utils'
import { Button } from '../components/ui/Button'
import { Badge } from '../components/ui/Badge'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/Card'
import { ModelLeaderboardTable } from '../components/tables/ModelLeaderboardTable'
import { useModelMetrics, useErrorAnalysis } from '../services/api'
import { ablationSummary, featureImportances } from '../data/mockModelData'

export const ScienceWhitepaper: React.FC = () => {
  const { data: metricsData, isLoading: isMetricsLoading } = useModelMetrics()
  const { data: errorData, isLoading: isErrorLoading } = useErrorAnalysis()

  const leaderboard = metricsData?.leaderboard || []
  const ablations = metricsData?.ablation_experiments?.length ? metricsData.ablation_experiments : ablationSummary

  const worstStates = errorData?.worst_performing_states || []
  const topErrors = errorData?.top_extreme_errors || []
  const yearlyStability = errorData?.yearly_error_stability || []

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-12">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-2">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 text-xs font-semibold">
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>Open Scientific Whitepaper</span>
          </div>
          <h1 className="text-2xl sm:text-4xl font-extrabold text-foreground tracking-tight">
            Empirical Validation, Risk Intelligence & Model Integrity Whitepaper
          </h1>
          <p className="text-xs sm:text-sm text-muted-foreground max-w-3xl leading-relaxed">
            Deterministic agronomic baselines, out-of-time temporal validation, geographic generalization across 20 Indian states, tree-ensemble uncertainty, and unsupervised anomaly detection.
          </p>
        </div>

        <Button
          asChild
          className="gap-2 text-xs font-bold shadow-md"
        >
          <a
            href="https://github.com/nupurmadaan04/AI-agriculture-yield-production/blob/main/docs/RESEARCH_PAPER.md"
            target="_blank"
            rel="noopener noreferrer"
          >
            <Download className="w-3.5 h-3.5" />
            <span>View Scientific Report</span>
          </a>
        </Button>
      </div>

      {/* Abstract Card */}
      <Card className="border-emerald-500/30 bg-emerald-500/5 p-6 sm:p-8 space-y-4">
        <div className="flex items-center gap-2 text-emerald-700 dark:text-emerald-300 font-bold text-sm">
          <CheckCircle2 className="w-5 h-5" />
          <span>Executive Abstract & Empirical Verdict</span>
        </div>
        <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
          In agricultural statistical reporting, crop yield is algebraically defined as <strong className="text-foreground">Yield (kg/ha) = (Production / Area) × 1,000</strong>. When machine learning models are provided post-harvest production alongside cultivated area, they achieve high apparent correlation (<strong className="text-foreground">r = 0.999746</strong>) primarily by non-linear curve fitting of a known mathematical identity.
        </p>
        <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
          The deterministic algebraic ratio evaluates with an empirical reconstruction error of <strong className="text-foreground">MAE = 4.21 kg/ha and R² = 0.9942</strong> (reflecting minor rounding in published statistical returns). For genuine pre-season forecasting, where production is strictly excluded, integrating <strong className="text-foreground">pre-season land allocation</strong> and <strong className="text-foreground">historical performance lags</strong> increases temporal holdout accuracy from <strong className="text-foreground">R² = 0.6479 to R² = 0.7785 (+20.1% R² gain, -23.7% MAE reduction)</strong> and resolves geographic generalization failures on unseen states (<strong className="text-foreground">GroupKFold R² = 0.7407 vs -0.0038</strong>).
        </p>
      </Card>

      {/* Performance Highlights Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="p-4 border-emerald-500/30 bg-emerald-500/5 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold text-emerald-700 dark:text-emerald-400 uppercase tracking-wider">
              Post-Harvest Identity
            </span>
            <Award className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
          </div>
          <div className="font-mono text-2xl font-black text-foreground">
            MAE 4.21 <span className="text-xs font-normal text-muted-foreground">kg/ha</span>
          </div>
          <p className="text-[11px] text-muted-foreground">
            Empirical formula verification across ICRISAT observations.
          </p>
        </Card>

        <Card className="p-4 border-sky-500/30 bg-sky-500/5 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold text-sky-700 dark:text-sky-400 uppercase tracking-wider">
              Advanced Pre-Season ML
            </span>
            <Zap className="w-4 h-4 text-sky-600 dark:text-sky-400" />
          </div>
          <div className="font-mono text-2xl font-black text-foreground">
            R² 0.7785 <span className="text-xs font-normal text-muted-foreground">(Temporal)</span>
          </div>
          <p className="text-[11px] text-muted-foreground">
            Out-of-time test MAE = 357.01 kg/ha with zero production leakage.
          </p>
        </Card>

        <Card className="p-4 border-indigo-500/30 bg-indigo-500/5 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold text-indigo-700 dark:text-indigo-400 uppercase tracking-wider">
              Exogenous State CV
            </span>
            <MapPin className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
          </div>
          <div className="font-mono text-2xl font-black text-foreground">
            R² 0.7407 <span className="text-xs font-normal text-muted-foreground">(GroupKFold)</span>
          </div>
          <p className="text-[11px] text-muted-foreground">
            Robust geographic generalization across 20 Indian states without look-ahead.
          </p>
        </Card>

        <Card className="p-4 border-red-500/30 bg-red-500/5 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold text-red-700 dark:text-red-400 uppercase tracking-wider">
              Isolation Forest Anomaly
            </span>
            <AlertTriangle className="w-4 h-4 text-red-600 dark:text-red-400" />
          </div>
          <div className="font-mono text-2xl font-black text-foreground">
            124 Outliers <span className="text-xs font-normal text-muted-foreground">(5.02%)</span>
          </div>
          <p className="text-[11px] text-muted-foreground">
            Unsupervised multi-variable detection of severe regional distress & survey variances.
          </p>
        </Card>
      </div>

      {/* Model Leaderboard */}
      <div className="space-y-4">
        <div className="space-y-1">
          <h2 className="text-xl font-bold text-foreground flex items-center gap-2">
            <span>1. Model Benchmark Leaderboard</span>
            {isMetricsLoading && <Loader2 className="w-4 h-4 animate-spin text-primary" />}
          </h2>
          <p className="text-xs text-muted-foreground">
            Systematic benchmark across regression algorithms comparing random holdouts, temporal splits, and state holdouts
          </p>
        </div>
        <ModelLeaderboardTable entries={leaderboard} />
      </div>

      {/* DAY 5: RISK INTELLIGENCE & METHODOLOGY */}
      <div className="space-y-4">
        <div className="space-y-1">
          <h2 className="text-xl font-bold text-foreground flex items-center gap-2">
            <ShieldAlert className="w-5 h-5 text-sky-500" />
            <span>2. Deterministic Risk Intelligence Methodology</span>
          </h2>
          <p className="text-xs text-muted-foreground">
            Transparent composite scoring heuristic combining tree ensemble dispersion, historical variance, and anomaly detection
          </p>
        </div>

        <Card className="p-6 space-y-4">
          <div className="space-y-2 text-xs leading-relaxed text-muted-foreground">
            <p>
              The system does not claim to use a black-box machine-learned risk model. Instead, it utilizes a fully transparent, documented <strong className="text-foreground">Deterministic Composite Risk Scoring Framework</strong> normalized to a 0–100 scale:
            </p>
            <div className="p-3 bg-muted/40 rounded-lg font-mono text-foreground text-[11px]">
              Risk Score = 0.35 × (Uncertainty Risk) + 0.30 × (Historical Deviation Risk) + 0.20 × (Model Residual Risk) + 0.15 × (Anomaly Risk)
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 pt-2">
              <div className="p-3 rounded-lg border border-emerald-500/20 bg-emerald-500/5">
                <span className="font-bold text-emerald-600 dark:text-emerald-400 block">LOW RISK</span>
                <span className="font-mono text-xs text-foreground">0 – 24.9</span>
                <p className="text-[10px] text-muted-foreground mt-1">Narrow prediction interval, historical stability.</p>
              </div>
              <div className="p-3 rounded-lg border border-sky-500/20 bg-sky-500/5">
                <span className="font-bold text-sky-600 dark:text-sky-400 block">MODERATE RISK</span>
                <span className="font-mono text-xs text-foreground">25.0 – 49.9</span>
                <p className="text-[10px] text-muted-foreground mt-1">Normal operational variance and typical prediction spread.</p>
              </div>
              <div className="p-3 rounded-lg border border-amber-500/20 bg-amber-500/5">
                <span className="font-bold text-amber-600 dark:text-amber-400 block">HIGH RISK</span>
                <span className="font-mono text-xs text-foreground">50.0 – 74.9</span>
                <p className="text-[10px] text-muted-foreground mt-1">Notable departure from regional mean (&gt;2 std dev) or wide spread.</p>
              </div>
              <div className="p-3 rounded-lg border border-red-500/20 bg-red-500/5">
                <span className="font-bold text-red-600 dark:text-red-400 block">CRITICAL RISK</span>
                <span className="font-mono text-xs text-foreground">75.0 – 100.0</span>
                <p className="text-[10px] text-muted-foreground mt-1">Statistical outlier flagged by Isolation Forest or severe shock.</p>
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* DAY 5: EXPLAINABILITY & ANOMALY DETECTION */}
      <div className="space-y-4">
        <div className="space-y-1">
          <h2 className="text-xl font-bold text-foreground flex items-center gap-2">
            <Sliders className="w-5 h-5 text-indigo-500" />
            <span>3. Explainable AI & Agricultural Anomaly Detection</span>
          </h2>
          <p className="text-xs text-muted-foreground">
            Feature attributions and unsupervised anomaly detection grounded in empirical agricultural properties
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card className="p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-sm text-foreground">Explainable AI (Tree Feature Attribution)</h3>
              <Badge variant="blue">Model Signals</Badge>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Tree-based feature attribution determines which pre-season attributes shifted the prediction relative to national baseline references.
            </p>
            <div className="p-3 rounded-lg bg-muted/40 text-[11px] text-muted-foreground space-y-1">
              <strong className="text-foreground block">Scientific Language Protocol:</strong>
              <p>
                We state: <em>"Historical yield was the strongest model contribution (+42.5%)"</em>.<br />
                We never state: <em>"Historical yield caused the yield outcome"</em>.
              </p>
            </div>
          </Card>

          <Card className="p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-sm text-foreground">Isolation Forest Anomaly Detection</h3>
              <Badge variant="warning">Unsupervised ML</Badge>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Trained on 7 numerical agricultural indicators (yield, area, production, land shares, lags). Outlier status is accompanied by measurable statistical facts (e.g. z-score deviations vs historical district averages).
            </p>
            <div className="p-3 rounded-lg bg-muted/40 text-[11px] text-muted-foreground space-y-1">
              <strong className="text-foreground block">Anomalies Detected in ICRISAT Panel:</strong>
              <p>
                124 observations (5.02%) identified across 2,469 total records, primarily representing small-acreage reporting variances or extreme local crop shocks.
              </p>
            </div>
          </Card>
        </div>
      </div>

      {/* DAY 7: TEMPORAL FORECASTING & EARLY WARNING METHODOLOGY */}
      <div className="space-y-6">
        <div className="space-y-1">
          <h2 className="text-xl font-bold text-foreground flex items-center gap-2">
            <Clock className="w-5 h-5 text-amber-500" />
            <span>4. Temporal Forecasting & Early Warning Methodology</span>
          </h2>
          <p className="text-xs text-muted-foreground">
            Multi-horizon autoregressive forecasting (1–3 years), robust Theil-Sen trend slope analytics, and deterministic distress scoring
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Chronological Validation Benchmark */}
          <Card className="p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-sm text-foreground">Chronological Out-of-Time Benchmark</h3>
              <Badge variant="success">Train: 2010–2015, Test: 2016–2017</Badge>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Forecasting models are evaluated strictly out-of-time (never using random train/test splits). All ML models are compared against simple statistical baselines.
            </p>

            <div className="rounded-lg border border-border overflow-hidden text-[11px]">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-muted/40 text-muted-foreground border-b border-border">
                    <th className="py-2 px-2.5 font-semibold">Model / Baseline</th>
                    <th className="py-2 px-2.5 font-semibold">MAE</th>
                    <th className="py-2 px-2.5 font-semibold">RMSE</th>
                    <th className="py-2 px-2.5 font-semibold">R²</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/60 font-mono">
                  <tr>
                    <td className="py-1.5 px-2.5 font-sans">Naive (Last Observed y_t-1)</td>
                    <td className="py-1.5 px-2.5">377.6</td>
                    <td className="py-1.5 px-2.5">575.1</td>
                    <td className="py-1.5 px-2.5">0.7319</td>
                  </tr>
                  <tr>
                    <td className="py-1.5 px-2.5 font-sans">Historical District Mean</td>
                    <td className="py-1.5 px-2.5">360.6</td>
                    <td className="py-1.5 px-2.5">530.6</td>
                    <td className="py-1.5 px-2.5">0.7718</td>
                  </tr>
                  <tr>
                    <td className="py-1.5 px-2.5 font-sans">Linear Trend Regression</td>
                    <td className="py-1.5 px-2.5">371.6</td>
                    <td className="py-1.5 px-2.5">538.2</td>
                    <td className="py-1.5 px-2.5">0.7652</td>
                  </tr>
                  <tr className="bg-emerald-500/10 font-bold text-emerald-600 dark:text-emerald-400">
                    <td className="py-1.5 px-2.5 font-sans">Random Forest Forecaster (Selected)</td>
                    <td className="py-1.5 px-2.5">353.0</td>
                    <td className="py-1.5 px-2.5">513.1</td>
                    <td className="py-1.5 px-2.5">0.7866</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </Card>

          {/* Trend & Early Warning Equations */}
          <Card className="p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-sm text-foreground">Theil-Sen & Early Warning Formulations</h3>
              <Badge variant="blue">Deterministic Math</Badge>
            </div>
            
            <div className="space-y-2 text-xs text-muted-foreground">
              <div>
                <strong className="text-foreground block font-sans">1. Theil-Sen Robust Median Slope:</strong>
                <code className="text-[11px] p-1.5 rounded bg-muted/60 block font-mono text-foreground mt-0.5">
                  Slope = median( (y_j - y_i) / (t_j - t_i) ) for all i &lt; j
                </code>
                <p className="text-[10px] mt-0.5">Immune to single-year outlier survey noise with Mann-Kendall non-parametric significance.</p>
              </div>

              <div>
                <strong className="text-foreground block font-sans">2. Composite Early Warning Score (0–100):</strong>
                <code className="text-[11px] p-1.5 rounded bg-muted/60 block font-mono text-foreground mt-0.5">
                  W = 0.30*Trend + 0.25*Forecast + 0.20*HistDev + 0.15*Anomaly + 0.10*Spread
                </code>
                <p className="text-[10px] mt-0.5">Scores &ge; 75 denote Critical, 50–74.9 High, 25–49.9 Moderate, &lt;25 Low severity.</p>
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* DAY 8: GEOSPATIAL INTELLIGENCE & CLUSTERING */}
      <div className="space-y-4">
        <div className="space-y-1">
          <h2 className="text-xl font-bold text-foreground flex items-center gap-2">
            <Compass className="w-5 h-5 text-purple-500" />
            <span>5. Geospatial Intelligence & Unsupervised Spatial Clustering</span>
          </h2>
          <p className="text-xs text-muted-foreground">
            Multi-dimensional agro-climatic clustering, within-state spatial outliers, and regional cosine similarity
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card className="p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-sm text-foreground">Spatial Feature Engineering & Outliers</h3>
              <Badge variant="blue">Empirical Scaling</Badge>
            </div>
            <div className="space-y-2 text-xs text-muted-foreground">
              <p>
                To avoid geographic confounding, districts are evaluated against their state baseline using robust within-state standard deviations:
              </p>
              <code className="text-[11px] p-2 rounded bg-muted/60 block font-mono text-foreground">
                z_state = (Yield_dist - Mean_state) / StdDev_state
              </code>
              <p className="text-[11px]">
                Districts with |z| &ge; 1.8&sigma; or volatility &gt; 2.0x state median are flagged as <strong>Spatial Outliers</strong> (statistical departures, never causal anomalies).
              </p>
            </div>
          </Card>

          <Card className="p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-sm text-foreground">KMeans Cluster Evaluation (311 Districts)</h3>
              <Badge variant="blue">Silhouette & DB Index</Badge>
            </div>
            <div className="space-y-2 text-xs text-muted-foreground">
              <div className="overflow-x-auto">
                <table className="w-full text-[11px] text-left border-collapse">
                  <thead>
                    <tr className="border-b border-border text-foreground font-mono">
                      <th className="py-1 px-2">K</th>
                      <th className="py-1 px-2">Silhouette (&uarr;)</th>
                      <th className="py-1 px-2">Calinski-Harabasz (&uarr;)</th>
                      <th className="py-1 px-2">Davies-Bouldin (&darr;)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/40 font-mono">
                    <tr><td className="py-1 px-2">K=3</td><td className="py-1 px-2">0.3098</td><td className="py-1 px-2">127.06</td><td className="py-1 px-2">1.0355</td></tr>
                    <tr className="bg-purple-500/10 font-bold text-purple-400"><td className="py-1 px-2">K=4 (Selected)</td><td className="py-1 px-2">0.2712</td><td className="py-1 px-2">120.78</td><td className="py-1 px-2">1.2358</td></tr>
                    <tr><td className="py-1 px-2">K=5</td><td className="py-1 px-2">0.2997</td><td className="py-1 px-2">128.06</td><td className="py-1 px-2">1.0539</td></tr>
                  </tbody>
                </table>
              </div>
              <p className="text-[10px] mt-1 text-slate-400">
                Clusters describe multi-dimensional productivity, volatility, slope, and outlier rates across irrigated vs rainfed zones.
              </p>
            </div>
          </Card>
        </div>
      </div>

      {/* DAY 9: MODEL VALIDATION, RELIABILITY & CONTINUOUS MONITORING */}
      <div className="space-y-4">
        <div className="space-y-1">
          <h2 className="text-xl font-bold text-foreground flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-500" />
            <span>6. Agricultural Model Reliability & Continuous Monitoring</span>
          </h2>
          <p className="text-xs text-muted-foreground">
            Strict chronological out-of-time evaluation, Population Stability Index (PSI), calibration, and data quality governance
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card className="p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-sm text-foreground">Chronological Out-of-Time Protocol</h3>
              <Badge variant="blue">Zero Leakage Split</Badge>
            </div>
            <div className="space-y-2 text-xs text-muted-foreground">
              <p>
                To avoid optimistic cross-validation bias caused by panel autocorrelation, all models are evaluated on unseen chronological test horizons:
              </p>
              <ul className="list-disc list-inside space-y-1 font-mono text-[11px]">
                <li>Training Baseline: Years &le; 2015 (1,851 records)</li>
                <li>Out-of-Time Test Set: Years 2016–2017 (618 records)</li>
                <li>Primary Model: Random Forest Forecaster (R² = 0.7866, MAE = 353.01 kg/ha)</li>
              </ul>
            </div>
          </Card>

          <Card className="p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-sm text-foreground">Feature Drift & Stability (PSI & KS)</h3>
              <Badge variant="blue">Distribution Shifts</Badge>
            </div>
            <div className="space-y-2 text-xs text-muted-foreground">
              <p>
                Distribution departures between baseline and operational data are continuously quantified:
              </p>
              <code className="text-[10px] p-2 rounded bg-muted/60 block font-mono text-foreground">
                PSI = &Sigma; (Actual% - Expected%) &times; ln(Actual% / Expected%)
              </code>
              <p className="text-[11px]">
                PSI &lt; 0.10 denotes stable distributions (NORMAL); 0.10 &le; PSI &lt; 0.25 flags WATCH status.
              </p>
            </div>
          </Card>
        </div>
      </div>

      {/* DAY 10: SCENARIO SIMULATION, SENSITIVITY & DECISION OPTIMIZATION */}
      <div className="space-y-4">
        <div className="space-y-1">
          <h2 className="text-xl font-bold text-foreground flex items-center gap-2">
            <Sliders className="w-5 h-5 text-purple-500" />
            <span>7. Scenario Simulation, Sensitivity Analysis & Decision Optimization</span>
          </h2>
          <p className="text-xs text-muted-foreground">
            Non-causal empirical what-if modeling, elasticity profiling, and multi-objective Pareto optimization
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card className="p-5 space-y-3">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-purple-400" />
              <h3 className="font-bold text-sm text-foreground">Controlled What-If Simulation</h3>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Perturbs empirical ICRISAT features (&Delta;Rice Area %, &Delta;Yield Lag %, &Delta;Rolling Mean %) relative to an immutable Baseline (&Delta;features = 0). Outputs projected response &Delta;Yield and P10–P90 ensemble dispersion.
            </p>
            <div className="text-[10px] font-mono text-purple-300 p-2 rounded bg-purple-500/10 border border-purple-500/20">
              &Delta;Yield = y&#770;<sub>scenario</sub> - y&#770;<sub>baseline</sub>
            </div>
          </Card>

          <Card className="p-5 space-y-3">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              <h3 className="font-bold text-sm text-foreground">Controlled Sensitivity Analysis</h3>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Systematic &plusmn;20% grid evaluation across supported features computing elasticity ranking:
            </p>
            <div className="text-[10px] font-mono text-cyan-300 p-2 rounded bg-cyan-500/10 border border-cyan-500/20">
              Elasticity = |(% &Delta;Yield<sub>+20%</sub> - % &Delta;Yield<sub>-20%</sub>)| / 40
            </div>
          </Card>

          <Card className="p-5 space-y-3">
            <div className="flex items-center gap-2">
              <Target className="w-4 h-4 text-emerald-400" />
              <h3 className="font-bold text-sm text-foreground">Pareto Decision Optimization</h3>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Transparent linear scalarization (0.40 Yield + 0.25 Risk + 0.20 Resource + 0.15 Reliability) with hard constraint verification and non-dominated Pareto frontier extraction.
            </p>
            <div className="text-[10px] font-mono text-emerald-300 p-2 rounded bg-emerald-500/10 border border-emerald-500/20">
              Score = &Sigma; w<sub>i</sub> &times; S<sub>i</sub>(Candidate)
            </div>
          </Card>
        </div>
      </div>

      {/* DAY 12: TEMPORAL MONITORING & EARLY WARNING METHODOLOGY */}
      <div className="space-y-4">
        <div className="space-y-1">
          <h2 className="text-xl font-bold text-foreground flex items-center gap-2">
            <Activity className="w-5 h-5 text-emerald-500" />
            <span>9. Temporal Monitoring, Change Detection & Historical Backtesting</span>
          </h2>
          <p className="text-xs text-muted-foreground">
            Multi-period rolling windows, CUSUM change detection, deterministic 5-tier alert severity, and chronological step-forward evaluation
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="p-5 space-y-3">
            <h3 className="font-bold text-sm text-foreground">Temporal Multi-Window Metrics</h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Computes 3-year, 5-year, and 8-year rolling means, standard deviations, rolling z-scores, trend slopes, and baseline deviations without lookahead bias.
            </p>
          </Card>

          <Card className="p-5 space-y-3">
            <h3 className="font-bold text-sm text-foreground">CUSUM & Regime Shift Detection</h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Tabular CUSUM control chart flags cumulative standard departures from historical reference means, strictly distinguishing statistical breaks from biological causes.
            </p>
          </Card>

          <Card className="p-5 space-y-3">
            <h3 className="font-bold text-sm text-foreground">Chronological Warning Backtesting</h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Step-forward historical validation (t &rarr; t+1) evaluating warning rule precision, recall, F1, and mean lead time across all panel districts without data leakage.
            </p>
          </Card>
        </div>
      </div>

      {/* DAY 13: EXPLAINABLE AI & DECISION TRACEABILITY */}
      <div className="space-y-4">
        <div className="space-y-1">
          <h2 className="text-xl font-bold text-foreground flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-emerald-500" />
            <span>10. Explainable AI & Decision Traceability Layer</span>
          </h2>
          <p className="text-xs text-muted-foreground">
            Multi-tiered interpretability, model-native vs permutation importance, and non-causal decision attribution
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card className="p-5 space-y-3">
            <h3 className="font-bold text-sm text-foreground">Dual Global Interpretability</h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Cross-validates <strong>Model-Native Gini Impurity</strong> against <strong>Out-of-Sample Permutation Importance</strong> evaluated on holdout evaluation sets. Disagreements in rank order are explicitly highlighted rather than suppressed.
            </p>
          </Card>

          <Card className="p-5 space-y-3">
            <h3 className="font-bold text-sm text-foreground">Marginal Reference Attribution</h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Decomposes local predictions into directional positive and negative feature contributions by substituting individual inputs into empirical median baseline reference vectors.
            </p>
          </Card>

          <Card className="p-5 space-y-3">
            <h3 className="font-bold text-sm text-foreground">Verifiable Decision Certificates</h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Generates deterministic <strong>SHA-256 audit certificates (EXP-xxxx)</strong> capturing model version (v2.1.0), dataset provenance, input features, and sensitivity limits for decision compliance.
            </p>
          </Card>
        </div>
      </div>

      {/* SCIENTIFIC LIMITATIONS */}
      <div className="space-y-4">
        <div className="space-y-1">
          <h2 className="text-xl font-bold text-foreground flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            <span>11. Scientific Limitations & Domain Boundaries</span>
          </h2>
          <p className="text-xs text-muted-foreground">
            Clear delineation of what the models can and cannot diagnose
          </p>
        </div>

        <Card className="p-6 border-amber-500/20 bg-amber-500/5 space-y-3 text-xs leading-relaxed text-muted-foreground">
          <ul className="list-disc list-inside space-y-2">
            <li>
              <strong className="text-foreground">Explanations Describe Models, Not Reality:</strong> Model attribution metrics reflect learned statistical loss minimization and decision splits within the ICRISAT training domain. They do not constitute agronomic, physical, or biological causality.
            </li>
            <li>
              <strong className="text-foreground">Early Warning &ne; Guaranteed Failure:</strong> Alerts represent empirical statistical warning signals, not guaranteed future crop collapse or drought declarations.
            </li>
            <li>
              <strong className="text-foreground">No Live Sensor/Weather Feed:</strong> Monitoring currently operates on the verified historical ICRISAT dataset and model-derived signals. No live IoT/satellite feeds are assumed.
            </li>
            <li>
              <strong className="text-foreground">Scenario &ne; Forecast:</strong> Scenario outputs represent hypothetical model simulations under modified input assumptions; they are never guaranteed future outcomes.
            </li>
            <li>
              <strong className="text-foreground">No Causal Claims:</strong> Simulated responses denote statistical associations within the trained feature space, not guaranteed agronomic or biological interventions.
            </li>
            <li>
              <strong className="text-foreground">Anomaly &ne; Error:</strong> An anomaly indicates an observation that departs statistically from historical distributions; it does not necessarily denote incorrect government survey data.
            </li>
            <li>
              <strong className="text-foreground">Statistical Association &ne; Physical Causation:</strong> High feature contribution reflects tree split importance and correlation in historical data, not agronomic causality.
            </li>
            <li>
              <strong className="text-foreground">Prediction Interval &ne; Confidence Interval:</strong> The 10th–90th percentile interval reflects the dispersion across the 150 decision trees in the random forest ensemble, not a formal frequentist confidence interval.
            </li>
            <li>
              <strong className="text-foreground">Dataset Scope:</strong> Analysis is grounded on the ICRISAT district panel covering 1966–2017. Real-time satellite imagery, high-resolution daily precipitation, and micro-fertilizer application rates are not included in this panel.
            </li>
          </ul>
        </Card>
      </div>
    </div>
  )
}


