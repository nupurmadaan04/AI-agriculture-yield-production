import React, { useState } from 'react'
import { AlertCircle, ShieldAlert, Activity, GitCommit, Award, CheckCircle } from 'lucide-react'
import { AlertExplanationResponse } from '../../types/explainability'
import { useAlertExplanation } from '../../services/api'

interface AlertExplanationProps {
  initialAlertId?: string
}

export const AlertExplanation: React.FC<AlertExplanationProps> = ({
  initialAlertId = 'ALR-000183'
}) => {
  const [alertId, setAlertId] = useState<string>(initialAlertId)
  const { data: alertExp, isLoading, isError } = useAlertExplanation(alertId)

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 mb-6">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4 mb-6">
        <div>
          <h3 className="text-base font-semibold text-white flex items-center gap-2">
            <ShieldAlert className="w-5 h-5 text-amber-400" />
            Monitoring Alert Deconstruction & Provenance
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Trace the complete multi-signal evidence chain that triggered a regional early warning alert
          </p>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-400">Alert ID:</label>
          <input
            type="text"
            value={alertId}
            onChange={(e) => setAlertId(e.target.value)}
            className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white font-mono focus:outline-none focus:border-amber-500 w-36"
          />
        </div>
      </div>

      {isLoading ? (
        <div className="p-8 text-center text-xs text-slate-400">Loading alert evidence chain...</div>
      ) : alertExp ? (
        <div>
          {/* Header Summary */}
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 mb-6">
            <div className="bg-slate-800/40 p-4 rounded-xl border border-slate-800">
              <p className="text-xs text-slate-400 mb-1">Target Location</p>
              <p className="text-base font-bold text-white">{alertExp.location}</p>
              <p className="text-[11px] text-slate-400 mt-0.5">Survey Year {alertExp.year}</p>
            </div>

            <div className="bg-slate-800/40 p-4 rounded-xl border border-slate-800">
              <p className="text-xs text-slate-400 mb-1">Alert Severity</p>
              <span
                className={`inline-block px-2.5 py-0.5 text-xs font-bold rounded ${
                  alertExp.severity === 'CRITICAL'
                    ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                    : alertExp.severity === 'HIGH'
                    ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                    : 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                }`}
              >
                {alertExp.severity}
              </span>
              <p className="text-[11px] text-slate-400 mt-1">Risk Score: {alertExp.composite_risk_score}/100</p>
            </div>

            <div className="bg-slate-800/40 p-4 rounded-xl border border-slate-800 sm:col-span-2">
              <p className="text-xs text-slate-400 mb-1">Dominant Warning Trigger</p>
              <p className="text-sm font-semibold text-amber-400">{alertExp.temporal_diagnostics.dominant_trigger}</p>
              <p className="text-[11px] text-slate-400 mt-0.5">{alertExp.temporal_diagnostics.summary}</p>
            </div>
          </div>

          {/* Multi-Signal Breakdown */}
          <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3">
            Multi-Signal Contribution Matrix
          </h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
            {alertExp.signal_breakdown.map((sig, idx) => (
              <div key={idx} className="p-3 bg-slate-800/50 rounded-lg border border-slate-800 flex items-start gap-3">
                <div className="p-1.5 bg-amber-500/10 text-amber-400 rounded mt-0.5">
                  <Activity className="w-3.5 h-3.5" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-white">{sig.type}</span>
                    <span className="text-[10px] px-1.5 py-0.2 bg-slate-700 text-slate-300 rounded font-semibold">
                      {sig.impact} IMPACT
                    </span>
                  </div>
                  <p className="text-xs text-slate-300 mt-1">{sig.description}</p>
                </div>
              </div>
            ))}
          </div>

          {/* Evidence Chain */}
          <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3">
            Sequential Evidence Audit Trail
          </h4>
          <div className="space-y-2 mb-6">
            {alertExp.evidence_chain.map((item, idx) => (
              <div key={idx} className="flex items-start gap-3 p-3 bg-slate-800/30 rounded-lg border border-slate-800">
                <span className="w-5 h-5 rounded-full bg-slate-800 text-emerald-400 text-xs font-bold flex items-center justify-center border border-slate-700 shrink-0">
                  {idx + 1}
                </span>
                <p className="text-xs text-slate-300 leading-relaxed">{item}</p>
              </div>
            ))}
          </div>

          {/* Recommended Operational Action */}
          <div className="p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-xl flex items-start gap-3 mb-4">
            <CheckCircle className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
            <div>
              <h5 className="text-xs font-bold text-emerald-300">Actionable Decision Guidance</h5>
              <p className="text-xs text-slate-300 mt-0.5">{alertExp.recommended_action}</p>
            </div>
          </div>

          <p className="text-[11px] text-slate-500 italic">{alertExp.scientific_disclaimer}</p>
        </div>
      ) : (
        <div className="p-8 text-center text-xs text-slate-500">
          Alert certificate not found. Enter a valid alert identifier (e.g. ALR-000183).
        </div>
      )}
    </div>
  )
}
