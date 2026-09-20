import React from 'react'
import { Link } from 'react-router-dom'
import {
  CheckCircle2,
  GitBranch,
  ShieldCheck,
  Scale,
  Calendar,
  AlertOctagon,
  FileKey,
  TrendingUp,
  ArrowRight
} from 'lucide-react'

interface ProcessStep {
  number: string
  title: string
  description: string
  detail: string
}

const steps: ProcessStep[] = [
  {
    number: '01',
    title: 'Evaluate Crop Eligibility',
    description: 'Filter panel observations for sample sufficiency (N ≥ 100 observations, multi-district coverage).',
    detail: '14 commodities qualify for full walk-forward evaluation out of 29 canonical crops.',
  },
  {
    number: '02',
    title: 'Compare Against Baselines',
    description: 'Benchmark ML candidates against Historical District Mean and 3-Year Rolling Average baselines.',
    detail: 'ML must demonstrate statistically defensible MAE reduction to be considered.',
  },
  {
    number: '03',
    title: 'Chronological Fold Testing',
    description: 'Evaluate performance across out-of-sample expanding historical folds (2014, 2015, 2016, 2017).',
    detail: 'Fold win-rate threshold: ≥ 50% outperformance required for production consideration.',
  },
  {
    number: '04',
    title: 'Diagnose Error Regimes',
    description: 'Stress test models across extreme drought years (e.g. 2015 shock) and quantile yield tails.',
    detail: 'Flag systematic over/under prediction bias and severe regime fragility.',
  },
  {
    number: '05',
    title: 'Select Governed Strategy',
    description: 'Classify crop into Production Ready ML, Conditional Production, or Baseline Production.',
    detail: 'Baseline persistence is retained as default when ML is unstable or overfit.',
  },
  {
    number: '06',
    title: 'Attach Cryptographic Provenance',
    description: 'Generate immutable request fingerprint, model artifact hash, and operating rule reasoning.',
    detail: 'Complete auditability for every prediction delivered through the serving engine.',
  },
  {
    number: '07',
    title: 'Deliver Bounded Forecast',
    description: 'Execute inference with variance clipping, non-negative bounds, and conformal prediction intervals.',
    detail: 'Produces calibrated uncertainty ranges and risk indicators for downstream decision support.',
  },
]

export const DecisionPipeline: React.FC = () => {
  return (
    <section className="py-16 md:py-24 border-b border-border/80 bg-muted/20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-12">
        
        {/* Section Header */}
        <div className="max-w-3xl space-y-3">
          <div className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-muted-foreground">
            <span>Governance Architecture</span>
          </div>
          <h2 className="text-2xl sm:text-4xl font-extrabold text-foreground tracking-tight">
            Models Do Not Automatically Get the Final Say.
          </h2>
          <p className="text-sm sm:text-base font-medium text-foreground/90 leading-relaxed pt-1">
            "Machine learning is retained only where temporal evidence supports its use. Statistical baselines remain valid production strategies when they are more stable."
          </p>
          <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
            Rather than blindly deploying black-box algorithms across every commodity, the platform enforces a 7-stage empirical filter that prioritizes forecast reliability over algorithmic complexity.
          </p>
        </div>

        {/* 7-Step Sequential Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {steps.map((step, idx) => (
            <div
              key={step.number}
              className={`rounded-xl border border-border bg-card p-5 space-y-3 shadow-subtle hover:border-border/90 transition-all ${
                idx === steps.length - 1 ? 'md:col-span-2 lg:col-span-1 border-primary/40 bg-primary/5' : ''
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-extrabold text-primary px-2 py-0.5 rounded bg-primary/10">
                  Step {step.number}
                </span>
                <span className="text-[10px] text-muted-foreground font-mono">Stage {idx + 1}/7</span>
              </div>
              
              <h3 className="text-sm font-bold text-foreground">
                {step.title}
              </h3>
              
              <p className="text-xs text-muted-foreground leading-relaxed">
                {step.description}
              </p>

              <div className="pt-2 border-t border-border/50 text-[11px] text-foreground/75 font-medium flex items-start gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5 text-primary shrink-0 mt-0.5" />
                <span>{step.detail}</span>
              </div>
            </div>
          ))}
        </div>

        {/* Bottom Callout */}
        <div className="p-5 rounded-xl border border-border bg-card flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary/10 text-primary shrink-0">
              <Scale className="w-5 h-5" />
            </div>
            <div>
              <h4 className="text-xs sm:text-sm font-bold text-foreground">
                Documented Empirical Strategy Taxonomy
              </h4>
              <p className="text-[11px] text-muted-foreground">
                Explore the complete decision logs and fold-level error matrices for all 14 evaluated crops.
              </p>
            </div>
          </div>
          <Link
            to="/modeling-readiness"
            className="text-xs font-bold text-primary hover:underline shrink-0 flex items-center gap-1"
          >
            <span>Inspect Readiness Matrix</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>

      </div>
    </section>
  )
}
