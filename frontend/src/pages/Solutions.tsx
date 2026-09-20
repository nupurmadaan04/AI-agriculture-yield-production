import React from 'react'
import { Link } from 'react-router-dom'
import {
  TrendingUp,
  Sprout,
  ShieldAlert,
  Building2,
  GraduationCap,
  ArrowRight,
  CheckCircle2,
  Satellite,
  CloudRain,
  BarChart3,
  Globe2,
  Calculator
} from 'lucide-react'
import { Button } from '../components/ui/Button'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'

export const Solutions: React.FC = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 space-y-16">
      {/* Header */}
      <div className="max-w-3xl space-y-3">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-sky-500/10 text-sky-600 dark:text-sky-400 text-xs font-semibold">
          <Globe2 className="w-3.5 h-3.5" />
          <span>Industry Solutions</span>
        </div>
        <h1 className="text-3xl sm:text-5xl font-extrabold text-foreground tracking-tight">
          Agricultural Intelligence Built for Every Market Participant
        </h1>
        <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
          From physical grain merchandisers trading multi-million-dollar futures contracts to government ministries setting national food security policy.
        </p>
      </div>

      {/* Solution 1: Grain Traders & Commodity Desks (CropProphet style) */}
      <section id="traders" className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center p-8 rounded-3xl bg-card border border-border shadow-xs">
        <div className="lg:col-span-7 space-y-4">
          <Badge variant="blue">Trading & Commodity Risk</Badge>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-foreground">
            Grain Traders & Exporters
          </h2>
          <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
            Gain an informational edge by converting complex satellite vegetation indices and monsoon forecasts into quantitative crop yield and supply estimates before they are priced in by the broader market.
          </p>

          <div className="space-y-2 text-xs text-foreground font-medium pt-2">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-sky-500 shrink-0" />
              <span>County & District level production deviation alerts</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-sky-500 shrink-0" />
              <span>Anticipate supply shortfalls during monsoon droughts (e.g. 2015 dip)</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-sky-500 shrink-0" />
              <span>Continuous pre-harvest yield curves updated weekly</span>
            </div>
          </div>

          <div className="pt-2">
            <Link to="/calculator">
              <Button className="gap-2 text-xs font-bold">
                <span>Test Trader Calculator</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Button>
            </Link>
          </div>
        </div>

        <div className="lg:col-span-5 p-6 rounded-2xl bg-muted/40 border border-border space-y-4 text-xs font-mono">
          <div className="flex justify-between border-b border-border pb-2">
            <span className="text-muted-foreground font-sans">Crop Target:</span>
            <span className="font-bold text-foreground">Rice (Kharif Export Quality)</span>
          </div>
          <div className="flex justify-between border-b border-border pb-2">
            <span className="text-muted-foreground font-sans">Key Supply Corridor:</span>
            <span className="text-foreground">Punjab / Haryana / WB</span>
          </div>
          <div className="flex justify-between border-b border-border pb-2">
            <span className="text-muted-foreground font-sans">Forecast Lead Time:</span>
            <span className="text-emerald-600 dark:text-emerald-400 font-bold">60–90 Days Pre-Harvest</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground font-sans">Supply Shock Index:</span>
            <span className="text-sky-600 dark:text-sky-400 font-bold">Calibrated (R² 0.741)</span>
          </div>
        </div>
      </section>

      {/* Solution 2: Agribusiness & Input Suppliers (EOS DA style) */}
      <section id="agribusiness" className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center p-8 rounded-3xl bg-card border border-border shadow-xs">
        <div className="lg:col-span-7 space-y-4">
          <Badge variant="success">Agribusiness & Inputs</Badge>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-foreground">
            Input Manufacturers & Seed Cooperatives
          </h2>
          <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
            Optimize regional supply chains, forecast seed and fertilizer demand, and benchmark commercial hybrid variety performance against historical agro-climatic district baselines.
          </p>

          <div className="space-y-2 text-xs text-foreground font-medium pt-2">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
              <span>District-level yield gap analysis (potential vs actual)</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
              <span>Fertilizer responsiveness curves across varying soil textures</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
              <span>Targeted distribution strategy for high-yield zones</span>
            </div>
          </div>

          <div className="pt-2">
            <Link to="/portal">
              <Button variant="outline" className="gap-2 text-xs font-semibold">
                <span>View District APY Data</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Button>
            </Link>
          </div>
        </div>

        <div className="lg:col-span-5 p-6 rounded-2xl bg-muted/40 border border-border space-y-3 text-xs">
          <p className="font-bold text-foreground font-sans">Yield Potential Benchmarking</p>
          <p className="text-muted-foreground leading-relaxed">
            Compare fertilizer application efficiency in canal-irrigated Punjab (3,982 kg/ha) vs rainfed Bihar (1,845 kg/ha) to optimize input delivery timelines.
          </p>
        </div>
      </section>

      {/* Solution 3: Crop Insurance & Agricultural Finance */}
      <section id="insurance" className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center p-8 rounded-3xl bg-card border border-border shadow-xs">
        <div className="lg:col-span-7 space-y-4">
          <Badge variant="warning">Underwriting & Claims</Badge>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-foreground">
            Crop Insurance & Agricultural Lenders
          </h2>
          <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
            Eliminate moral hazard and streamline claim verification with audited historical yield baselines, satellite vegetation indices, and deterministic crop cutting verification.
          </p>

          <div className="space-y-2 text-xs text-foreground font-medium pt-2">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-amber-500 shrink-0" />
              <span>Parametric index insurance underwriting models</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-amber-500 shrink-0" />
              <span>Audited historical yield variance to calculate actuarial fair premiums</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-amber-500 shrink-0" />
              <span>Post-harvest claim verification using deterministic baseline formula</span>
            </div>
          </div>
        </div>

        <div className="lg:col-span-5 p-6 rounded-2xl bg-amber-500/5 border border-amber-500/20 space-y-3 text-xs">
          <p className="font-bold text-foreground">Actuarial Risk Reduction</p>
          <p className="text-muted-foreground leading-relaxed">
            Accurate distinction between micro-cultivation anomalies and true weather disaster losses prevents over-payouts in high variance districts like The Nilgiris (TN).
          </p>
        </div>
      </section>

      {/* Solution 4: Governments & Policy Planners (UPAg Aligned) */}
      <section id="policy" className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center p-8 rounded-3xl bg-card border border-border shadow-xs">
        <div className="lg:col-span-7 space-y-4">
          <Badge variant="outline">Public Sector & Policy</Badge>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-foreground">
            Agricultural Ministries & Food Security Planners
          </h2>
          <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
            Align with UPAg (Unified Portal for Agricultural Statistics) standards to generate advance crop estimates, allocate strategic grain reserves, and orchestrate targeted disaster relief funds.
          </p>

          <div className="space-y-2 text-xs text-foreground font-medium pt-2">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-purple-500 shrink-0" />
              <span>Advance harvest estimates (1st, 2nd, 3rd, and Final)</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-purple-500 shrink-0" />
              <span>Procurement target planning for Food Corporation of India (FCI)</span>
            </div>
          </div>
        </div>

        <div className="lg:col-span-5 p-6 rounded-2xl bg-purple-500/5 border border-purple-500/20 space-y-3 text-xs">
          <p className="font-bold text-foreground">Food Security Planning</p>
          <p className="text-muted-foreground leading-relaxed">
            Empirical district modeling ensures public grain buffer stocks are positioned ahead of regional monsoon deficits.
          </p>
        </div>
      </section>
    </div>
  )
}
