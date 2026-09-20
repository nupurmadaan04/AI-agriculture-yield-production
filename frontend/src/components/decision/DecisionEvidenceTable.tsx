import React, { useState } from 'react'
import type { EvidenceItem, EvidenceType } from '../../types/decision'
import { Filter, Layers, Database } from 'lucide-react'

interface Props {
  items: EvidenceItem[]
}

const TYPE_COLORS: Record<EvidenceType, string> = {
  OBSERVED: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
  PREDICTED: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30',
  SIMULATED: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/30',
  DERIVED: 'bg-purple-500/10 text-purple-400 border-purple-500/30',
  MODEL_ATTRIBUTION: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
  VALIDATION: 'bg-blue-500/10 text-blue-400 border-blue-500/30'
}

export const DecisionEvidenceTable: React.FC<Props> = ({ items }) => {
  const [selectedType, setSelectedType] = useState<string>('ALL')

  const filtered = selectedType === 'ALL'
    ? items
    : items.filter(e => e.evidence_type === selectedType)

  const types = ['ALL', 'OBSERVED', 'PREDICTED', 'SIMULATED', 'DERIVED', 'MODEL_ATTRIBUTION', 'VALIDATION']

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl overflow-hidden space-y-4 p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-emerald-400" />
          <h3 className="text-sm font-semibold text-slate-100">Normalized Evidence Landscape ({filtered.length} items)</h3>
        </div>

        {/* Type Filter Pills */}
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          <Filter className="w-3.5 h-3.5 text-slate-500 mr-1" />
          {types.map(t => (
            <button
              key={t}
              onClick={() => setSelectedType(t)}
              className={`px-2.5 py-1 rounded-lg transition-colors border text-[11px] ${
                selectedType === t
                  ? 'bg-slate-700 text-white border-slate-600 font-medium'
                  : 'bg-slate-900/80 text-slate-400 border-slate-800 hover:border-slate-700 hover:text-slate-200'
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs text-slate-300">
          <thead className="bg-slate-950/60 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
            <tr>
              <th className="py-2.5 px-3">Evidence ID</th>
              <th className="py-2.5 px-3">Type</th>
              <th className="py-2.5 px-3">Category</th>
              <th className="py-2.5 px-3">Statement</th>
              <th className="py-2.5 px-3">Value</th>
              <th className="py-2.5 px-3">Source & Method</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {filtered.map(ev => (
              <tr key={ev.evidence_id} className="hover:bg-slate-800/30 transition-colors">
                <td className="py-2.5 px-3 font-mono font-medium text-slate-200 whitespace-nowrap">
                  {ev.evidence_id}
                </td>
                <td className="py-2.5 px-3 whitespace-nowrap">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-semibold border ${TYPE_COLORS[ev.evidence_type] || 'bg-slate-800 text-slate-300'}`}>
                    {ev.evidence_type}
                  </span>
                </td>
                <td className="py-2.5 px-3 capitalize text-slate-400 font-medium whitespace-nowrap">
                  {ev.category}
                </td>
                <td className="py-2.5 px-3 text-slate-300 max-w-md">
                  {ev.statement}
                </td>
                <td className="py-2.5 px-3 font-mono font-medium text-emerald-300 whitespace-nowrap">
                  {ev.value} <span className="text-[10px] text-slate-500 font-normal">{ev.unit}</span>
                </td>
                <td className="py-2.5 px-3 text-slate-400 whitespace-nowrap text-[11px]">
                  <div>{ev.source_module}</div>
                  <div className="text-[10px] text-slate-500">{ev.source_method}</div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
