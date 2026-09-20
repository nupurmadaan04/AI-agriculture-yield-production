import React, { useState } from 'react'
import { Sparkles, Bot, ShieldCheck, Database, Info, Layers, Award } from 'lucide-react'
import { CopilotChat } from '../components/intelligence/CopilotChat'
import { useCopilot } from '../services/api'
import { CopilotQueryResponse } from '../types/intelligence'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'

export const AgriculturalCopilot: React.FC = () => {
  const copilotMutation = useCopilot()

  const [messages, setMessages] = useState<any[]>([
    {
      id: 'init_1',
      role: 'assistant',
      content: (
        "Welcome to **AgriYield AI Copilot** — your evidence-first agricultural decision research assistant.\n\n"
        + "I reason strictly over the 2,469 district panel records of the ICRISAT dataset (2010–2017), "
        + "trained regression pipelines, deterministic risk scores, and unsupervised Isolation Forest anomaly detection.\n\n"
        + "You can ask about state comparisons, yield rankings, regional risk factors, anomalies, or model benchmarks."
      ),
      timestamp: new Date().toLocaleTimeString()
    }
  ])

  const handleSendMessage = async (question: string) => {
    const userMsg = {
      id: `user_${Date.now()}`,
      role: 'user',
      content: question,
      timestamp: new Date().toLocaleTimeString()
    }
    setMessages(prev => [...prev, userMsg])

    try {
      const res: CopilotQueryResponse = await copilotMutation.mutateAsync({ question })
      const botMsg = {
        id: `bot_${Date.now()}`,
        role: 'assistant',
        content: res.answer,
        responseDetails: res,
        timestamp: new Date().toLocaleTimeString()
      }
      setMessages(prev => [...prev, botMsg])
    } catch (err: any) {
      const errMsg = {
        id: `err_${Date.now()}`,
        role: 'assistant',
        content: `Error retrieving grounded evidence: ${err.message || 'Service unavailable.'}`,
        timestamp: new Date().toLocaleTimeString()
      }
      setMessages(prev => [...prev, errMsg])
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      {/* Top Header */}
      <div className="space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-500/10 text-sky-600 dark:text-sky-400 text-xs font-semibold">
          <Bot className="w-3.5 h-3.5" />
          <span>Evidence-First Agricultural AI Copilot</span>
        </div>
        <h1 className="text-2xl sm:text-4xl font-extrabold text-foreground tracking-tight">
          Agricultural AI Copilot
        </h1>
        <p className="text-xs sm:text-sm text-muted-foreground max-w-3xl leading-relaxed">
          Ask natural language questions about crop productivity, state rankings, multi-year yield trajectories, and anomaly signals. Answers are grounded in real data and controlled analytics tools.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Main Chat Feed (8 cols) */}
        <div className="lg:col-span-8">
          <CopilotChat
            messages={messages}
            isPending={copilotMutation.isPending}
            onSendMessage={handleSendMessage}
            onSelectPrompt={handleSendMessage}
          />
        </div>

        {/* Right Info Sidebar (4 cols) */}
        <div className="lg:col-span-4 space-y-6">
          <Card className="border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-bold flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-500" />
                <span>Scientific Grounding Architecture</span>
              </CardTitle>
              <CardDescription className="text-xs">
                Zero hallucination protocol with controlled tools
              </CardDescription>
            </CardHeader>

            <CardContent className="space-y-3 text-xs leading-relaxed text-muted-foreground">
              <p>
                The Copilot does not answer from speculative LLM memory. User questions trigger structured NLP intent parsing, which executes deterministically registered backend analytics tools.
              </p>

              <div className="p-3 rounded-lg bg-muted/40 space-y-1 text-[11px]">
                <strong className="text-foreground block font-sans">Controlled Analytics Tools:</strong>
                <ul className="list-disc list-inside space-y-0.5 font-mono">
                  <li>get_state_rankings()</li>
                  <li>get_state_risk()</li>
                  <li>get_trends()</li>
                  <li>get_records()</li>
                  <li>detect_anomaly()</li>
                  <li>predict_pre_season_advanced()</li>
                </ul>
              </div>

              <div className="p-3 rounded-lg border border-sky-500/20 bg-sky-500/5 space-y-1 text-[11px]">
                <strong className="text-sky-800 dark:text-sky-300 block font-sans">Language Constraints:</strong>
                <p>
                  Answers strictly utilize non-causal statistical terminology: <em>"model estimates"</em>, <em>"statistical association"</em>, <em>"feature contribution"</em>, and <em>"prediction spread"</em>.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
