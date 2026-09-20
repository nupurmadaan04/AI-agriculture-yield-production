import React from 'react'
import { Link } from 'react-router-dom'
import { Calculator, LineChart, ArrowRight, ShieldAlert, CheckCircle2, FileSpreadsheet, Sparkles } from 'lucide-react'
import { Badge } from '../ui/Badge'
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card'

export const AnalyticalModes: React.FC = () => {
  return (
    <section className="py-16 md:py-24 border-b border-border/80 bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-12">
        
        {/* Section Header */}
        <div className="max-w-3xl space-y-3">
          <div className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-muted-foreground">
            <span>Methodological Integrity</span>
          </div>
          <h2 className="text-2xl sm:text-4xl font-extrabold text-foreground tracking-tight">
            Two Questions. Two Different Methods.
          </h2>
          <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
            In agricultural data systems, algebraic reporting verification and pre-season forecasting solve fundamentally different operational problems. The platform enforces strict scientific separation between derived historical identities and true out-of-sample predictions.
          </p>
        </div>

        {/* 2 Column Comparison Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-stretch">
          
          {/* Card 1: Post-Harvest Yield Verification */}
          <div className="rounded-xl border border-border bg-card p-6 sm:p-8 flex flex-col justify-between space-y-6 shadow-subtle">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="p-2 rounded-lg bg-muted text-foreground">
                    <Calculator className="w-4 h-4" />
                  </div>
                  <span className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Mode 01</span>
                </div>
                <Badge variant="outline" className="font-mono text-[11px] font-bold">
                  DERIVED / POST-HARVEST
                </Badge>
              </div>

              <div className="space-y-2">
                <h3 className="text-lg sm:text-xl font-bold text-foreground">
                  Post-Harvest Yield Verification
                </h3>
                <p className="text-xs sm:text-sm text-foreground/80 font-medium italic">
                  "Was the reported yield mathematically consistent with production and cultivated area?"
                </p>
              </div>

              <p className="text-xs text-muted-foreground leading-relaxed">
                Yield can be mathematically derived from reported production and cultivated area using the standard agricultural agronomic identity. Used by statistical auditors to detect recording transcription errors and district reporting anomalies.
              </p>

              {/* Formula Block */}
              <div className="p-4 rounded-lg bg-muted/50 border border-border/80 font-mono text-center space-y-1">
                <div className="text-[10px] uppercase font-bold text-muted-foreground tracking-wider">
                  Deterministic Agronomic Identity
                </div>
                <div className="text-xs sm:text-sm font-bold text-foreground">
                  Yield (kg/ha) = [ Production (tonnes) / Area (ha) ] × 1,000
                </div>
              </div>

              <div className="space-y-1.5 text-xs text-muted-foreground pt-1">
                <div className="flex items-start gap-2">
                  <span className="text-muted-foreground font-mono text-[10px] mt-0.5">•</span>
                  <span>Input Data: Total reported harvest volume and gross cultivated area.</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-muted-foreground font-mono text-[10px] mt-0.5">•</span>
                  <span>Analytical Role: Data hygiene, audit checks, and baseline sanity validation.</span>
                </div>
                <div className="flex items-start gap-2 text-amber-700 dark:text-amber-400 font-medium">
                  <span className="font-mono text-[10px] mt-0.5">!</span>
                  <span>Scientific Guard: Not an out-of-sample forecast. Never conflated with ML accuracy.</span>
                </div>
              </div>
            </div>

            <div className="pt-4 border-t border-border/60 flex items-center justify-between">
              <a
                href="#post-harvest-verification"
                className="text-xs font-bold text-foreground hover:text-primary transition-colors flex items-center gap-1.5"
              >
                <span>Launch Verification Formula</span>
                <ArrowRight className="w-3.5 h-3.5 text-primary" />
              </a>
              <span className="text-[10px] text-muted-foreground font-mono">Formula: P / A × 1000</span>
            </div>
          </div>

          {/* Card 2: Pre-Season Yield Forecasting */}
          <div className="rounded-xl border border-border bg-card p-6 sm:p-8 flex flex-col justify-between space-y-6 shadow-subtle">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="p-2 rounded-lg bg-primary/10 text-primary">
                    <LineChart className="w-4 h-4" />
                  </div>
                  <span className="text-xs font-bold text-primary uppercase tracking-wider">Mode 02</span>
                </div>
                <Badge variant="success" className="font-mono text-[11px] font-bold">
                  PREDICTED / PRE-SEASON
                </Badge>
              </div>

              <div className="space-y-2">
                <h3 className="text-lg sm:text-xl font-bold text-foreground">
                  Pre-Season Yield Forecasting
                </h3>
                <p className="text-xs sm:text-sm text-foreground/80 font-medium italic">
                  "Can yield be estimated before harvest using only information available at the forecast origin?"
                </p>
              </div>

              <p className="text-xs text-muted-foreground leading-relaxed">
                A genuine pre-season forecasting problem evaluated with chronological walk-forward validation (origins 2014–2017). Models use only historical lags, weather trajectories, and pre-season crop allocations without post-harvest production data.
              </p>

              {/* Validation Boundary Block */}
              <div className="p-4 rounded-lg bg-primary/5 border border-primary/20 font-mono text-center space-y-1">
                <div className="text-[10px] uppercase font-bold text-primary tracking-wider">
                  Chronological Expanding Walk-Forward
                </div>
                <div className="text-xs sm:text-sm font-bold text-foreground">
                  Train: 1966 → Origin (T-1) &nbsp;|&nbsp; Test: Horizon (T)
                </div>
              </div>

              <div className="space-y-1.5 text-xs text-muted-foreground pt-1">
                <div className="flex items-start gap-2">
                  <span className="text-primary font-mono text-[10px] mt-0.5">•</span>
                  <span>Input Data: Pre-season land allocation, soil characteristics, and climate indicators.</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-primary font-mono text-[10px] mt-0.5">•</span>
                  <span>Decision Guard: Evaluated across 14 crops against statistical persistence baselines.</span>
                </div>
                <div className="flex items-start gap-2 text-emerald-700 dark:text-emerald-400 font-medium">
                  <span className="font-mono text-[10px] mt-0.5">✓</span>
                  <span>Scientific Guard: Zero future leakage, audited feature cutoffs, certified routing.</span>
                </div>
              </div>
            </div>

            <div className="pt-4 border-t border-border/60 flex items-center justify-between">
              <Link
                to="/forecast"
                className="text-xs font-bold text-primary hover:underline transition-colors flex items-center gap-1.5"
              >
                <span>Explore Pre-Season Forecaster</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
              <span className="text-[10px] text-muted-foreground font-mono">14 Evaluated Commodities</span>
            </div>
          </div>

        </div>
      </div>
    </section>
  )
}
