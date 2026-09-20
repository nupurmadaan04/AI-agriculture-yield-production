import React, { useState } from 'react'
import { Database, ChevronDown, ChevronUp, ShieldCheck, Cpu, Layers } from 'lucide-react'
import { CopilotEvidenceItem } from '../../types/intelligence'
import { Badge } from '../ui/Badge'

interface EvidencePanelProps {
  evidence: CopilotEvidenceItem[]
  recordsAnalyzed: number
  toolsUsed: string[]
  limitations: string[]
}

export const EvidencePanel: React.FC<EvidencePanelProps> = ({
  evidence,
  recordsAnalyzed,
  toolsUsed,
  limitations
}) => {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <div className="rounded-xl border border-border bg-muted/20 text-xs overflow-hidden">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-2.5 flex items-center justify-between font-semibold text-foreground hover:bg-muted/40 transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          <Database className="w-3.5 h-3.5 text-sky-500" />
          <span>Verifiable Evidence & Data Provenance</span>
          <Badge variant="outline" className="text-[10px] font-mono">
            {recordsAnalyzed} records analyzed
          </Badge>
        </div>
        <div className="flex items-center gap-1.5 text-muted-foreground text-[11px]">
          <span>{isOpen ? 'Hide Evidence' : 'Inspect Evidence'}</span>
          {isOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </div>
      </button>

      {isOpen && (
        <div className="p-4 border-t border-border space-y-4 animate-in fade-in duration-200">
          {/* Tools & Provenance Chips */}
          <div className="space-y-1.5">
            <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider block">
              Controlled Analytics Tools Executed:
            </span>
            <div className="flex flex-wrap gap-1.5">
              {toolsUsed.map((tool, idx) => (
                <span
                  key={idx}
                  className="px-2 py-0.5 rounded-md bg-sky-500/10 border border-sky-500/20 text-sky-700 dark:text-sky-300 font-mono text-[10px]"
                >
                  ✓ {tool}()
                </span>
              ))}
            </div>
          </div>

          {/* Evidence Items */}
          <div className="space-y-2">
            <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider block">
              Retrieved Dataset Records & Evidence Snippets:
            </span>
            <div className="grid grid-cols-1 gap-2">
              {evidence.map((item, idx) => (
                <div key={idx} className="p-3 rounded-lg border border-border/80 bg-card space-y-1">
                  <div className="flex items-center justify-between font-semibold">
                    <span className="text-foreground">{item.source_name}</span>
                    <span className="text-[10px] text-muted-foreground font-mono">{item.records_count} rows</span>
                  </div>
                  <p className="text-[11px] text-muted-foreground">{item.description}</p>
                  {item.data_snippet && (
                    <pre className="mt-1 p-2 rounded bg-muted/60 text-[10px] font-mono text-foreground overflow-x-auto max-h-32">
                      {JSON.stringify(item.data_snippet, null, 2)}
                    </pre>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Scientific Limitations */}
          {limitations.length > 0 && (
            <div className="p-3 rounded-lg border border-border bg-card/40 space-y-1 text-[11px] text-muted-foreground">
              <span className="font-bold text-foreground block">Model & Grounding Boundaries:</span>
              <ul className="list-disc list-inside space-y-0.5">
                {limitations.map((lim, idx) => (
                  <li key={idx}>{lim}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
