import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import {
  ArrowRight,
  ShieldCheck,
  Activity,
  Layers,
  CheckCircle2,
  Lock,
  Sparkles,
  ChevronRight,
  Database,
  LineChart,
  Cpu,
  GitBranch,
  FileCheck
} from 'lucide-react'
import { Button } from '../ui/Button'
import { Badge } from '../ui/Badge'

interface PipelineStep {
  id: string
  label: string
  sublabel: string
  status: 'validated' | 'active' | 'governed' | 'certified'
  description: string
  icon: React.ElementType
}

const pipelineSteps: PipelineStep[] = [
  {
    id: 'data',
    label: 'DATA',
    sublabel: '71,601 Unified Records',
    status: 'validated',
    description: 'Cleaned, harmonized panel dataset across 20 states, 311 districts, and 29 verified crops (1966–2017).',
    icon: Database,
  },
  {
    id: 'validation',
    label: 'TEMPORAL VALIDATION',
    sublabel: 'Walk-Forward (2014–17)',
    status: 'validated',
    description: 'Expanding walk-forward evaluation strictly preventing future data leakage across multiple historical origins.',
    icon: Activity,
  },
  {
    id: 'models',
    label: 'MODEL / BASELINE',
    sublabel: 'ML vs Statistical Baseline',
    status: 'active',
    description: 'Rigorous empirical tournament comparing Random Forest, GBDT, and historical persistence baselines per crop.',
    icon: Cpu,
  },
  {
    id: 'strategy',
    label: 'STRATEGY',
    sublabel: 'Governed Routing',
    status: 'governed',
    description: 'Certification guard selecting ML only where win-rate and mean gain exceed baseline thresholds.',
    icon: GitBranch,
  },
  {
    id: 'forecast',
    label: 'FORECAST',
    sublabel: 'Pre-Season Estimate',
    status: 'certified',
    description: 'Certified pre-season yield estimate with conformal confidence bounds and safety clipping.',
    icon: LineChart,
  },
  {
    id: 'provenance',
    label: 'PROVENANCE',
    sublabel: 'SHA-256 Audit Trail',
    status: 'certified',
    description: 'Immutable execution signature, feature hash, and operational governance reasoning attached to every output.',
    icon: FileCheck,
  },
]

export const HeroSection: React.FC = () => {
  const [activeStep, setActiveStep] = useState<string>('strategy')
  const currentStep = pipelineSteps.find(s => s.id === activeStep) || pipelineSteps[3]

  return (
    <section className="relative pt-12 pb-16 md:pt-20 md:pb-24 border-b border-border/80 hero-glow bg-grid-pattern">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-8 items-center">
          
          {/* Left Column: Heading & Narrative */}
          <div className="lg:col-span-6 space-y-6 text-center lg:text-left">
            {/* Small Eyebrow */}
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-md bg-primary/10 border border-primary/20 text-primary text-xs font-semibold tracking-wide uppercase">
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>Agricultural Forecasting & Decision Intelligence</span>
            </div>

            {/* Main Heading */}
            <h1 className="text-3xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-foreground leading-[1.12]">
              From Agricultural Data <br className="hidden sm:inline" />
              to <span className="text-primary underline decoration-primary/30 underline-offset-8">Defensible Forecasts</span>.
            </h1>

            {/* Supporting Text */}
            <p className="text-sm sm:text-base text-muted-foreground leading-relaxed max-w-xl mx-auto lg:mx-0">
              Pre-season crop-yield forecasting, temporal validation, and governed decision strategies across 20 Indian states and 311 districts.
            </p>

            {/* CTAs */}
            <div className="flex flex-wrap items-center justify-center lg:justify-start gap-3 pt-2">
              <Link to="/forecast">
                <Button size="lg" className="gap-2 font-bold px-6 shadow-sm">
                  <span>Explore Forecasting</span>
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </Link>

              <Link to="/science">
                <Button variant="outline" size="lg" className="gap-2 font-semibold px-6">
                  <span>View Scientific Validation</span>
                </Button>
              </Link>
            </div>

            {/* Subtle third link */}
            <div className="pt-1">
              <a
                href="#post-harvest-verification"
                className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground font-medium transition-colors"
              >
                <span>Need to verify historical harvest reporting?</span>
                <span className="text-primary underline underline-offset-2">Post-Harvest Verification</span>
                <ChevronRight className="w-3 h-3 text-primary" />
              </a>
            </div>
          </div>

          {/* Right Column: Analytical System Pipeline Diagram */}
          <div className="lg:col-span-6">
            <div className="bg-card border border-border rounded-xl p-5 sm:p-6 shadow-card space-y-5">
              
              {/* Pipeline Header */}
              <div className="flex items-center justify-between border-b border-border/60 pb-3">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                  <span className="text-xs font-bold uppercase tracking-wider text-foreground">
                    Scientific Decision Architecture
                  </span>
                </div>
                <Badge variant="outline" className="text-[10px] font-mono">
                  Origin: 2014 → 2017
                </Badge>
              </div>

              {/* Pipeline Step Grid / Flow */}
              <div className="space-y-2">
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                  {pipelineSteps.map((step, idx) => {
                    const isSelected = activeStep === step.id
                    const Icon = step.icon
                    return (
                      <button
                        key={step.id}
                        onClick={() => setActiveStep(step.id)}
                        className={`text-left p-3 rounded-lg border transition-all text-xs cursor-pointer flex flex-col justify-between ${
                          isSelected
                            ? 'bg-primary/10 border-primary text-foreground shadow-xs font-semibold'
                            : 'bg-muted/40 border-border/70 hover:bg-muted/80 text-muted-foreground'
                        }`}
                      >
                        <div className="flex items-center justify-between w-full mb-2">
                          <span className="text-[10px] font-mono font-bold text-muted-foreground">0{idx + 1}</span>
                          <Icon className={`w-3.5 h-3.5 ${isSelected ? 'text-primary' : 'text-muted-foreground'}`} />
                        </div>
                        <span className="font-bold text-[11px] text-foreground block truncate">{step.label}</span>
                        <span className="text-[10px] text-muted-foreground block truncate mt-0.5">{step.sublabel}</span>
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* Active Step Details Panel */}
              <div className="p-4 rounded-lg bg-muted/40 border border-border/80 text-xs space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-foreground">{currentStep.label}</span>
                    <span className="text-[10px] text-muted-foreground font-mono">({currentStep.sublabel})</span>
                  </div>
                  <Badge variant="success" className="text-[10px] capitalize">
                    {currentStep.status}
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {currentStep.description}
                </p>
              </div>

              {/* Pipeline Footer Guarantee */}
              <div className="flex items-center justify-between text-[11px] text-muted-foreground pt-1 border-t border-border/50">
                <div className="flex items-center gap-1.5">
                  <ShieldCheck className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
                  <span>Strict Pre-Season Boundary Guarantee</span>
                </div>
                <Link to="/modeling-readiness" className="text-primary hover:underline font-semibold flex items-center gap-0.5">
                  <span>Audit Logs</span>
                  <ChevronRight className="w-3 h-3" />
                </Link>
              </div>

            </div>
          </div>

        </div>
      </div>
    </section>
  )
}
