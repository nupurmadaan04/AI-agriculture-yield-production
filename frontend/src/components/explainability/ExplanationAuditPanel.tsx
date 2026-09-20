import React, { useState } from 'react'
import { FileCheck, ShieldCheck, Database, Calendar, Hash, Award, CheckCircle, Copy } from 'lucide-react'
import { ExplanationAuditResponse } from '../../types/explainability'
import { useExplanationAudit } from '../../services/api'

interface ExplanationAuditPanelProps {
  initialExplanationId?: string
}

export const ExplanationAuditPanel: React.FC<ExplanationAuditPanelProps> = ({
  initialExplanationId = 'EXP-000183'
}) => {
  const [explanationId, setExplanationId] = useState<string>(initialExplanationId)
  const [copied, setCopied] = useState<boolean>(false)

  const { data: audit, isLoading } = useExplanationAudit(explanationId)

  const handleCopy = () => {
    if (audit) {
      navigator.clipboard.writeText(JSON.stringify(audit, null, 2))
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 mb-6">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4 mb-6">
        <div>
          <h3 className="text-base font-semibold text-white flex items-center gap-2">
            <Award className="w-5 h-5 text-emerald-400" />
            Verifiable Explanation Audit Certificate
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Cryptographically reproducible audit record linking model version, input vectors, and attribution weights
          </p>
        </div>

        <div className="flex items-center gap-3">
          <input
            type="text"
            value={explanationId}
            onChange={(e) => setExplanationId(e.target.value)}
            className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white font-mono focus:outline-none focus:border-emerald-500 w-36"
            placeholder="EXP-xxxx"
          />

          <button
            onClick={handleCopy}
            disabled={!audit}
            className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium rounded-lg border border-slate-700 flex items-center gap-1.5 transition-colors disabled:opacity-50"
          >
            {copied ? <CheckCircle className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            {copied ? 'Copied JSON' : 'Copy Record'}
          </button>
        </div>
      </div>

      {isLoading ? (
        <div className="p-8 text-center text-xs text-slate-400">Loading audit certificate...</div>
      ) : audit ? (
        <div className="p-6 bg-slate-950 border border-emerald-500/30 rounded-xl relative overflow-hidden">
          {/* Certificate Badge Background */}
          <div className="absolute top-0 right-0 transform translate-x-8 -translate-y-8 w-32 h-32 bg-emerald-500/5 rounded-full blur-2xl pointer-events-none" />

          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800/80 pb-4 mb-5">
            <div>
              <span className="text-[10px] font-mono tracking-widest text-emerald-400 uppercase font-bold">
                Agricultural XAI Certificate of Decision Traceability
              </span>
              <h4 className="text-lg font-mono font-bold text-white mt-0.5">{audit.explanation_id}</h4>
            </div>

            <div className="flex items-center gap-2">
              <span className="px-2.5 py-1 text-[11px] font-mono rounded bg-slate-800 text-slate-300 border border-slate-700">
                Timestamp: {audit.timestamp.slice(0, 19).replace('T', ' ')}
              </span>
              <span className="px-2.5 py-1 text-[11px] font-semibold rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1">
                <ShieldCheck className="w-3.5 h-3.5" /> VERIFIED REPRODUCIBLE
              </span>
            </div>
          </div>

          {/* Certificate Metadata Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
            <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-500 block mb-0.5">Model Artifact</span>
              <span className="text-xs font-semibold text-slate-200">{audit.model_name}</span>
              <span className="text-[10px] font-mono text-emerald-400 block mt-0.5">v{audit.model_version}</span>
            </div>

            <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-500 block mb-0.5">Dataset Provenance</span>
              <span className="text-xs font-semibold text-slate-200">{audit.dataset_version}</span>
              <span className="text-[10px] text-blue-400 block mt-0.5">District Panel Records</span>
            </div>

            <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-500 block mb-0.5">Entity / Region</span>
              <span className="text-xs font-semibold text-white">{audit.entity}</span>
              <span className="text-[10px] text-slate-400 block mt-0.5">Survey Record</span>
            </div>

            <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-500 block mb-0.5">Attributed Yield</span>
              <span className="text-xs font-bold text-emerald-400">
                {audit.prediction_kg_ha.toLocaleString()} kg/ha
              </span>
              <span className="text-[10px] text-slate-400 block mt-0.5">
                Delta: {audit.prediction_delta_kg_ha >= 0 ? '+' : ''}
                {audit.prediction_delta_kg_ha.toFixed(1)}
              </span>
            </div>
          </div>

          {/* Scientific Limitations Accordion/Box */}
          <div className="p-3.5 bg-slate-900/60 rounded-lg border border-slate-800/80">
            <h5 className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider mb-2">
              Boundaries & Scientific Constraints
            </h5>
            <ul className="text-xs text-slate-400 space-y-1">
              {audit.limitations.map((lim, idx) => (
                <li key={idx} className="flex items-start gap-2">
                  <span className="text-emerald-400 text-xs font-bold mt-0.5">•</span>
                  <span>{lim}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      ) : (
        <div className="p-8 text-center text-xs text-slate-500">
          Certificate not found. Enter a valid explanation ID (e.g. EXP-000183).
        </div>
      )}
    </div>
  )
}
