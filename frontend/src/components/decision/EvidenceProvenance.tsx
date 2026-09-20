import React from 'react'
import type { DecisionProvenance } from '../../types/decision'
import { GitBranch, Database, Cpu, FileText, ArrowRight } from 'lucide-react'

interface Props {
  provenance: DecisionProvenance
}

export const EvidenceProvenance: React.FC<Props> = ({ provenance }) => {
  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-purple-400" />
          <h3 className="text-sm font-semibold text-slate-100">Evidence Provenance & Lineage DAG</h3>
        </div>
        <div className="text-xs text-slate-400 font-mono">
          {provenance.total_nodes} Nodes • {provenance.total_edges} Edges • Dataset: {provenance.dataset_version}
        </div>
      </div>

      {/* Visual Lineage Pipeline */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs pt-2">
        <div className="bg-slate-950/80 border border-emerald-500/30 rounded-xl p-3.5 space-y-2">
          <div className="flex items-center gap-1.5 text-emerald-400 font-semibold">
            <Database className="w-4 h-4" />
            <span>1. Canonical Dataset</span>
          </div>
          <div className="text-slate-200 font-mono text-[11px]">{provenance.dataset_version}</div>
          <div className="text-[10px] text-slate-500">71,601 panel records (1966–2017) across 20 states</div>
        </div>

        <div className="bg-slate-950/80 border border-cyan-500/30 rounded-xl p-3.5 space-y-2">
          <div className="flex items-center gap-1.5 text-cyan-400 font-semibold">
            <Cpu className="w-4 h-4" />
            <span>2. Registered Models</span>
          </div>
          <div className="text-slate-200 font-mono text-[11px]">exogenous_rf_forecaster v2.1.0</div>
          <div className="text-[10px] text-slate-500">Out-of-time chronological validation (R² 0.7866)</div>
        </div>

        <div className="bg-slate-950/80 border border-purple-500/30 rounded-xl p-3.5 space-y-2">
          <div className="flex items-center gap-1.5 text-purple-400 font-semibold">
            <GitBranch className="w-4 h-4" />
            <span>3. Analytical Evidence</span>
          </div>
          <div className="text-slate-200 font-mono text-[11px]">10+ Normalized Evidence IDs</div>
          <div className="text-[10px] text-slate-500">Explicit typing (Observed, Predicted, Simulated)</div>
        </div>

        <div className="bg-slate-950/80 border border-amber-500/30 rounded-xl p-3.5 space-y-2">
          <div className="flex items-center gap-1.5 text-amber-400 font-semibold">
            <FileText className="w-4 h-4" />
            <span>4. Decision Statements</span>
          </div>
          <div className="text-slate-200 font-mono text-[11px]">Priorities & Options</div>
          <div className="text-[10px] text-slate-500">Grounded in verified multi-signal evidence</div>
        </div>
      </div>

      <div className="text-[11px] text-slate-400 bg-slate-950/40 p-3 rounded-lg border border-slate-800/60 leading-relaxed">
        <strong>Traceability Assurance:</strong> Every priority and decision recommendation is mathematically traceable back through intermediate analytical evidence items to registered model artifacts and the empirical ICRISAT dataset.
      </div>
    </div>
  )
}
