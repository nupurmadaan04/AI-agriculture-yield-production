import React from 'react'
import {
  X,
  ShieldCheck,
  AlertTriangle,
  MapPin,
  FlaskConical,
  FileText
} from 'lucide-react'
import { Badge } from '../ui/Badge'
import { Button } from '../ui/Button'
import type { Alert } from '../../types/monitoring'

interface Props {
  alert: Alert | null
  onClose: () => void
  onExploreScenario?: (state: string) => void
  onViewGeospatial?: (state: string) => void
}

export const AlertDetailDrawer: React.FC<Props> = ({
  alert,
  onClose,
  onExploreScenario,
  onViewGeospatial,
}) => {
  if (!alert) return null

  const getSeverityBadgeVariant = (severity: string) => {
    switch (severity.toUpperCase()) {
      case 'CRITICAL':
        return 'red' as const
      case 'HIGH':
        return 'rose' as const
      case 'ELEVATED':
        return 'amber' as const
      case 'WATCH':
        return 'amber' as const
      default:
        return 'blue' as const
    }
  }

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-slate-950/80 backdrop-blur-sm flex justify-end animate-fadeIn">
      <div className="w-full max-w-xl bg-slate-900 border-l border-slate-800 h-full overflow-y-auto shadow-2xl p-6 flex flex-col justify-between space-y-6">
        <div className="space-y-6">
          {/* Header */}
          <div className="flex items-start justify-between border-b border-slate-800 pb-4">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="font-mono text-xs font-semibold text-emerald-400">
                  {alert.alert_id}
                </span>
                <Badge variant={getSeverityBadgeVariant(alert.severity)}>
                  {alert.severity} SEVERITY
                </Badge>
                <Badge variant="purple" className="text-[10px]">
                  EVIDENCE: {alert.evidence_strength}
                </Badge>
              </div>
              <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
                <MapPin className="w-5 h-5 text-slate-400" />
                {alert.location}
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Observation Year: <span className="font-semibold text-slate-300">{alert.year}</span> • Priority Rank:{' '}
                <span className="font-semibold text-slate-300">#{alert.priority_rank ?? 1}</span>
              </p>
            </div>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Dominant Signal & Composite Score */}
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400">Dominant Warning Signal</span>
              <span className="text-xs font-mono font-bold text-slate-200">
                Score: {alert.composite_risk_score.toFixed(1)}/100
              </span>
            </div>
            <div className="text-sm font-semibold text-amber-300 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
              {alert.dominant_signal}
            </div>
            {alert.supporting_signals.length > 0 && (
              <div className="pt-2 border-t border-slate-800/60 text-xs text-slate-400 space-y-1">
                <span className="font-medium text-slate-300">Supporting Signals:</span>
                <div className="flex flex-wrap gap-1.5 mt-1">
                  {alert.supporting_signals.map((sig, i) => (
                    <span
                      key={i}
                      className="px-2 py-0.5 rounded bg-slate-800/80 border border-slate-700/60 text-[11px] text-slate-300"
                    >
                      {sig}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Evidence Chain */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
              <FileText className="w-4 h-4 text-emerald-400" />
              Complete Evidence Chain
            </h4>
            <div className="space-y-2 border-l-2 border-slate-800 pl-3 ml-1">
              {alert.evidence_chain.map((step, idx) => (
                <div key={idx} className="relative space-y-0.5">
                  <div className="absolute -left-[19px] top-1 w-2.5 h-2.5 rounded-full bg-emerald-500/40 border border-emerald-400" />
                  <p className="text-xs text-slate-300 leading-relaxed pl-1">{step}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Model Validation & Reliability Context */}
          <div className="p-4 rounded-xl bg-emerald-950/20 border border-emerald-500/30 space-y-2.5">
            <div className="flex items-center gap-2 text-xs font-semibold text-emerald-400">
              <ShieldCheck className="w-4 h-4" />
              Underlying Model Reliability Context (Day 9 Verified)
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1">
              <div className="p-2 bg-slate-900/60 rounded border border-slate-800 text-center">
                <div className="text-[10px] text-slate-400">Model R²</div>
                <div className="text-xs font-bold font-mono text-slate-200">
                  {alert.model_validation.r2}
                </div>
              </div>
              <div className="p-2 bg-slate-900/60 rounded border border-slate-800 text-center">
                <div className="text-[10px] text-slate-400">Test MAE</div>
                <div className="text-xs font-bold font-mono text-slate-200">
                  {alert.model_validation.mae} kg/ha
                </div>
              </div>
              <div className="p-2 bg-slate-900/60 rounded border border-slate-800 text-center">
                <div className="text-[10px] text-slate-400">Feature Drift</div>
                <div className="text-xs font-bold font-mono text-emerald-400">
                  {alert.model_validation.drift_status}
                </div>
              </div>
              <div className="p-2 bg-slate-900/60 rounded border border-slate-800 text-center">
                <div className="text-[10px] text-slate-400">Data Quality</div>
                <div className="text-xs font-bold font-mono text-slate-200">
                  {alert.model_validation.data_quality_score}/100
                </div>
              </div>
            </div>
          </div>

          {/* Recommended Monitoring Action */}
          <div className="p-3.5 bg-slate-800/40 rounded-xl border border-slate-700/60 space-y-1">
            <div className="text-xs font-semibold text-slate-300">Recommended Monitoring Action</div>
            <p className="text-xs text-slate-400 leading-relaxed">{alert.recommended_action}</p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="pt-4 border-t border-slate-800 flex flex-col sm:flex-row gap-2.5">
          {onExploreScenario && (
            <Button
              variant="secondary"
              onClick={() => {
                onExploreScenario(alert.state)
                onClose()
              }}
              className="flex-1 text-xs gap-1.5"
            >
              <FlaskConical className="w-4 h-4 text-purple-400" />
              Explore Scenario
            </Button>
          )}
          {onViewGeospatial && (
            <Button
              variant="outline"
              onClick={() => {
                onViewGeospatial(alert.state)
                onClose()
              }}
              className="flex-1 text-xs gap-1.5"
            >
              <MapPin className="w-4 h-4 text-cyan-400" />
              View Spatial Context
            </Button>
          )}
          <Button variant="ghost" onClick={onClose} className="text-xs">
            Close
          </Button>
        </div>
      </div>
    </div>
  )
}
