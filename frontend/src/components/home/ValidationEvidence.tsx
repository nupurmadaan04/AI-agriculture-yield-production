import React from 'react'
import { Link } from 'react-router-dom'
import {
  ShieldCheck,
  CheckCircle2,
  Lock,
  GitCommit,
  Calendar,
  Layers,
  ArrowRight,
  FileCode2,
  FileCheck2
} from 'lucide-react'
import { Button } from '../ui/Button'
import { Badge } from '../ui/Badge'

interface ValidationPillar {
  title: string
  metric: string
  status: string
  description: string
  icon: React.ElementType
}

const pillars: ValidationPillar[] = [
  {
    title: 'Walk-Forward Validation',
    metric: '2014 → 2017',
    status: 'Chronological Folds',
    description: 'Models evaluated strictly on future unseen years with expanding historical training sets (1966–T).',
    icon: Calendar,
  },
  {
    title: 'Bitwise Reproducibility',
    metric: '100%',
    status: 'Zero Variance',
    description: 'Dual-run execution audit verifies 100% bitwise parity across model serializations and predictions.',
    icon: GitCommit,
  },
  {
    title: 'Prediction Provenance',
    metric: 'SHA-256',
    status: 'Cryptographic Trail',
    description: 'Every forecast logs request hashes, model artifact IDs, dataset versions, and strategy reasoning.',
    icon: Lock,
  },
  {
    title: 'Feature Timing Audited',
    metric: 'Zero Leakage',
    status: 'Pre-Season Boundary',
    description: 'Rigorous feature cutoffs prevent post-harvest production quantities from leaking into forecasting models.',
    icon: ShieldCheck,
  },
  {
    title: 'Model Selection Philosophy',
    metric: 'Crop-Specific',
    status: 'Tournament Governed',
    description: 'ML deployed only where temporal evidence proves genuine out-of-sample gain over statistical baselines.',
    icon: Layers,
  },
]

export const ValidationEvidence: React.FC = () => {
  return (
    <section className="py-16 md:py-24 border-b border-border/80 bg-muted/20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-12">
        
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
          <div className="max-w-3xl space-y-3">
            <div className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-muted-foreground">
              <span>Scientific Integrity</span>
            </div>
            <h2 className="text-2xl sm:text-4xl font-extrabold text-foreground tracking-tight">
              Validation Before Visualization.
            </h2>
            <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
              Every chart, metric, and forecast in this platform is backed by reproducible audit trails and strict chronological validation protocols.
            </p>
          </div>
          <Link to="/science" className="shrink-0">
            <Button variant="outline" className="gap-2 text-xs font-semibold">
              <FileCheck2 className="w-4 h-4 text-primary" />
              <span>Open Scientific Audit</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Button>
          </Link>
        </div>

        {/* 5 Pillars Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 sm:gap-6">
          {pillars.map((pillar, idx) => {
            const Icon = pillar.icon
            return (
              <div
                key={idx}
                className="rounded-xl border border-border bg-card p-5 space-y-3 flex flex-col justify-between shadow-subtle hover:border-primary/40 transition-colors"
              >
                <div className="space-y-2.5">
                  <div className="flex items-center justify-between">
                    <div className="p-2 rounded-lg bg-primary/10 text-primary">
                      <Icon className="w-4 h-4" />
                    </div>
                    <Badge variant="outline" className="text-[10px] font-mono">
                      {pillar.status}
                    </Badge>
                  </div>
                  
                  <div>
                    <span className="text-xl sm:text-2xl font-mono font-extrabold text-foreground block">
                      {pillar.metric}
                    </span>
                    <h3 className="text-xs font-bold text-foreground mt-0.5">
                      {pillar.title}
                    </h3>
                  </div>

                  <p className="text-[11px] text-muted-foreground leading-relaxed">
                    {pillar.description}
                  </p>
                </div>

                <div className="pt-2 border-t border-border/50 text-[10px] text-foreground/80 font-medium flex items-center gap-1">
                  <CheckCircle2 className="w-3 h-3 text-primary" />
                  <span>Audited & Verified</span>
                </div>
              </div>
            )
          })}
        </div>

        {/* Architecture & Artifact Verification Card */}
        <div className="p-6 rounded-xl border border-border bg-card shadow-subtle flex flex-col lg:flex-row items-center justify-between gap-6">
          <div className="space-y-1.5 text-center lg:text-left">
            <div className="flex items-center justify-center lg:justify-start gap-2">
              <FileCode2 className="w-4 h-4 text-primary" />
              <h4 className="text-sm font-bold text-foreground">
                Automated Pipeline Artifact Certification
              </h4>
            </div>
            <p className="text-xs text-muted-foreground max-w-2xl">
              Inspect model serialization manifests, walk-forward residual spreads, GroupKFold geographic generalizability tests, and full data provenance.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <Link to="/modeling-readiness">
              <Button size="sm" variant="default" className="text-xs font-bold gap-1.5">
                <span>View Modeling Readiness</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Button>
            </Link>
            <Link to="/model-reliability">
              <Button size="sm" variant="outline" className="text-xs font-semibold">
                Reliability & Drift
              </Button>
            </Link>
          </div>
        </div>

      </div>
    </section>
  )
}
