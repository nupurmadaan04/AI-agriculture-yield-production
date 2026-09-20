import React, { useState } from 'react'
import { Send, Sparkles, User, Bot, Loader2, Info } from 'lucide-react'
import { CopilotQueryResponse } from '../../types/intelligence'
import { EvidencePanel } from './EvidencePanel'
import { Button } from '../ui/Button'
import { Badge } from '../ui/Badge'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  responseDetails?: CopilotQueryResponse
  timestamp: string
}

interface CopilotChatProps {
  messages: ChatMessage[]
  isPending: boolean
  onSendMessage: (question: string) => void
  onSelectPrompt: (prompt: string) => void
}

export const CopilotChat: React.FC<CopilotChatProps> = ({
  messages,
  isPending,
  onSendMessage,
  onSelectPrompt
}) => {
  const [input, setInput] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isPending) return
    onSendMessage(input.trim())
    setInput('')
  }

  const samplePrompts = [
    "Which state had the highest rice yield?",
    "Compare Punjab and Haryana.",
    "Why is Punjab showing higher modeled risk?",
    "Show unusual yield observations and anomalies.",
    "Show Punjab's historical yield trend.",
    "Summarize model validation benchmark results."
  ]

  return (
    <div className="flex flex-col h-[650px] rounded-2xl border border-border bg-card shadow-xl overflow-hidden">
      {/* Top Copilot Bar */}
      <div className="p-4 border-b border-border bg-muted/30 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-sky-500 text-white flex items-center justify-center shadow-md shadow-sky-500/20">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h3 className="font-bold text-sm text-foreground">AgriYield AI Copilot</h3>
            <p className="text-[10px] text-muted-foreground">Evidence-grounded agricultural research assistant</p>
          </div>
        </div>
        <Badge variant="blue" className="text-[10px]">Controlled Tool Registry Active</Badge>
      </div>

      {/* Message Feed */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 text-xs">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {msg.role === 'assistant' && (
              <div className="w-7 h-7 rounded-lg bg-sky-500/10 text-sky-600 dark:text-sky-400 flex items-center justify-center shrink-0 mt-0.5">
                <Bot className="w-4 h-4" />
              </div>
            )}

            <div className={`space-y-3 max-w-2xl ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
              <div
                className={`p-4 rounded-2xl leading-relaxed whitespace-pre-line ${
                  msg.role === 'user'
                    ? 'bg-sky-500 text-white font-medium shadow-md shadow-sky-500/10'
                    : 'bg-muted/40 border border-border/80 text-foreground'
                }`}
              >
                {msg.content}
              </div>

              {/* Expandable Evidence Drawer */}
              {msg.responseDetails && (
                <EvidencePanel
                  evidence={msg.responseDetails.evidence}
                  recordsAnalyzed={msg.responseDetails.records_analyzed}
                  toolsUsed={msg.responseDetails.tools_used}
                  limitations={msg.responseDetails.limitations}
                />
              )}
            </div>

            {msg.role === 'user' && (
              <div className="w-7 h-7 rounded-lg bg-primary/10 text-primary flex items-center justify-center shrink-0 mt-0.5">
                <User className="w-4 h-4" />
              </div>
            )}
          </div>
        ))}

        {isPending && (
          <div className="flex gap-3 justify-start items-center text-muted-foreground text-xs pl-2">
            <Loader2 className="w-4 h-4 animate-spin text-sky-500" />
            <span>Executing controlled tools and retrieving grounded evidence...</span>
          </div>
        )}
      </div>

      {/* Sample Quick Questions Chips */}
      <div className="px-4 py-2 border-t border-border/60 bg-muted/10 overflow-x-auto flex gap-1.5 scrollbar-none">
        {samplePrompts.map((prompt, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => onSelectPrompt(prompt)}
            className="px-2.5 py-1 rounded-full bg-background border border-border/80 text-[11px] text-muted-foreground hover:text-foreground hover:border-sky-500/50 whitespace-nowrap transition-colors cursor-pointer"
          >
            {prompt}
          </button>
        ))}
      </div>

      {/* Input Box */}
      <form onSubmit={handleSubmit} className="p-3 border-t border-border bg-card flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask an agricultural intelligence question (e.g. 'Compare Punjab and Haryana', 'Why is Kerala high risk?')..."
          className="flex-1 rounded-xl border border-border bg-background px-4 py-2.5 text-xs font-medium focus:ring-2 focus:ring-sky-500 focus:outline-none"
        />
        <Button
          type="submit"
          disabled={isPending || !input.trim()}
          size="sm"
          className="px-4 gap-1.5 text-xs font-bold shadow-md shadow-sky-500/20"
        >
          <span>Ask</span>
          <Send className="w-3.5 h-3.5" />
        </Button>
      </form>
    </div>
  )
}
