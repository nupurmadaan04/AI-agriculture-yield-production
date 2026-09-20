import React from 'react'
import {
  Clock,
  CheckCircle2,
  AlertTriangle,
  ShieldAlert,
  BarChart3
} from 'lucide-react'
import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'
import type { Alert } from '../../types/monitoring'

interface Props {
  alert?: Alert | null
}

export const SignalTimeline: React.FC<Props> = ({ alert }) => {
  const steps = [
    {
      title: 'Observation & Data Ingestion',
      desc: alert
        ? `Observed agricultural record ingested for ${alert.location} (${alert.year}). Verified with 100% completeness.`
        : 'Historical panel record processed through validation pipeline.',
      status: 'VERIFIED',
      badgeVariant: 'green' as const,
      icon: CheckCircle2,
    },
    {
      title: 'Statistical Departure Detection',
      desc: alert
        ? `Yield/feature departure evaluated against multi-window baseline. Dominant signal: ${alert.dominant_signal}.`
        : 'Deviation calculated against 3-year, 5-year, and long-term historical baseline.',
      status: 'CALCULATED',
      badgeVariant: 'cyan' as const,
      icon: BarChart3,
    },
    {
      title: 'Multi-Signal Evidence Fusion',
      desc: alert
        ? `Fused ${alert.signal_count} independent signal streams. Evidence strength evaluated as ${alert.evidence_strength}.`
        : 'Trajectory, spatial deviation, and anomaly indicators combined.',
      status: 'FUSED',
      badgeVariant: 'purple' as const,
      icon: ShieldAlert,
    },
    {
      title: 'Severity & Tier Assignment',
      desc: alert
        ? `Assigned ${alert.severity} severity tier (Composite Score: ${alert.composite_risk_score.toFixed(1)}/100).`
        : 'Assigned deterministic tier based on explicit thresholds.',
      status: alert?.severity ?? 'MONITORED',
      badgeVariant: (alert?.severity === 'CRITICAL' ? 'red' : alert?.severity === 'HIGH' ? 'rose' : alert?.severity === 'ELEVATED' ? 'amber' : 'blue') as any,
      icon: AlertTriangle,
    },
    {
      title: 'Adaptive Decision Support',
      desc: alert
        ? alert.recommended_action
        : 'Continuous tracking active. Scenario stress tests available.',
      status: 'ACTIONABLE',
      badgeVariant: 'green' as const,
      icon: Clock,
    },
  ]

  return (
    <Card className="p-5 border border-slate-800 bg-slate-900/60 backdrop-blur space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div>
          <h3 className="text-base font-semibold text-slate-100 flex items-center gap-2">
            <Clock className="w-5 h-5 text-purple-400" />
            Signal Evolution & Audit Lifecycle
          </h3>
          <p className="text-xs text-slate-400">
            Observation → Deviation → Fusion → Severity Tier → Decision Action
          </p>
        </div>
        {alert && (
          <Badge variant="purple" className="text-xs font-mono">
            {alert.alert_id}
          </Badge>
        )}
      </div>

      {/* Timeline Steps */}
      <div className="relative pl-6 space-y-5 border-l-2 border-slate-800">
        {steps.map((step, idx) => {
          const Icon = step.icon
          return (
            <div key={idx} className="relative group">
              <div className="absolute -left-[31px] top-0.5 p-1 rounded-full bg-slate-900 border border-slate-700 group-hover:border-emerald-500 transition-colors">
                <Icon className="w-3.5 h-3.5 text-slate-400 group-hover:text-emerald-400" />
              </div>
              <div className="flex items-center justify-between gap-2">
                <h4 className="text-xs font-semibold text-slate-200">{step.title}</h4>
                <Badge variant={step.badgeVariant} className="text-[10px] px-1.5 py-0.5">
                  {step.status}
                </Badge>
              </div>
              <p className="text-xs text-slate-400 mt-1 leading-relaxed">{step.desc}</p>
            </div>
          )
        })}
      </div>
    </Card>
  )
}
