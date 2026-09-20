import React, { useState } from 'react'
import type { DecisionAudit } from '../../types/decision'
import { ShieldCheck, Copy, Check, Download, Hash } from 'lucide-react'

interface Props {
  audit: DecisionAudit
}

export const DecisionAuditPanel: React.FC<Props> = ({ audit }) => {
  const [copied, setCopied] = useState(false)

  const copyCertificate = () => {
    navigator.clipboard.writeText(audit.decision_id)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const exportReport = () => {
    const jsonStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(audit, null, 2))
    const downloadAnchor = document.createElement('a')
    downloadAnchor.setAttribute("href", jsonStr)
    downloadAnchor.setAttribute("download", `DECISION_AUDIT_${audit.decision_id}.json`)
    document.body.appendChild(downloadAnchor)
    downloadAnchor.click()
    downloadAnchor.remove()
  }

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-emerald-400" />
          <div>
            <h3 className="text-sm font-semibold text-slate-100">Cryptographic Decision Audit Certificate</h3>
            <div className="text-[11px] text-slate-400">Deterministic SHA-256 Decision Certification</div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={copyCertificate}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-mono border border-slate-700 transition-colors"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5 text-slate-400" />}
            <span>{audit.decision_id}</span>
          </button>

          <button
            onClick={exportReport}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 rounded-lg text-xs font-medium border border-emerald-500/40 transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export Certificate</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
        <div className="bg-slate-950/60 rounded-lg p-3 border border-slate-800">
          <div className="text-slate-500 text-[11px]">Dataset Bounded</div>
          <div className="font-semibold text-slate-200 mt-0.5">{audit.dataset_version}</div>
        </div>

        <div className="bg-slate-950/60 rounded-lg p-3 border border-slate-800">
          <div className="text-slate-500 text-[11px]">Model Bounded</div>
          <div className="font-semibold text-cyan-300 mt-0.5">{audit.model_version}</div>
        </div>

        <div className="bg-slate-950/60 rounded-lg p-3 border border-slate-800">
          <div className="text-slate-500 text-[11px]">Evidence & Scenarios</div>
          <div className="font-semibold text-emerald-300 mt-0.5">{audit.evidence_count} Evidence • {audit.scenario_count} Scenarios</div>
        </div>

        <div className="bg-slate-950/60 rounded-lg p-3 border border-slate-800">
          <div className="text-slate-500 text-[11px]">Generated At</div>
          <div className="font-mono text-slate-300 mt-0.5 text-[11px]">{new Date(audit.generated_at).toLocaleString()}</div>
        </div>
      </div>

      <div className="text-[11px] text-slate-400 italic bg-slate-950/40 p-3 rounded-lg border border-slate-800/60">
        "{audit.audit_disclaimer}"
      </div>
    </div>
  )
}
