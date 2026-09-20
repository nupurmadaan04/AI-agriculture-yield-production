import React, { useState } from 'react'
import { FileText, Download, Copy, Check, Loader2, Sparkles, X, ShieldCheck } from 'lucide-react'
import { useGenerateReport, useFilters } from '../../services/api'
import { ReportGenerateResponse } from '../../types/intelligence'
import { Button } from '../ui/Button'
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from '../ui/Card'
import { Badge } from '../ui/Badge'

interface IntelligenceReportModalProps {
  isOpen: boolean
  onClose: () => void
  initialState?: string
  initialDistrict?: string
}

export const IntelligenceReportModal: React.FC<IntelligenceReportModalProps> = ({
  isOpen,
  onClose,
  initialState = 'Punjab',
  initialDistrict = 'Ludhiana'
}) => {
  const { data: filtersData } = useFilters()
  const availableStates = filtersData?.states || ['Punjab', 'Haryana', 'Tamil Nadu', 'West Bengal', 'Uttar Pradesh']

  const [state, setState] = useState(initialState)
  const [district, setDistrict] = useState(initialDistrict)
  const [year, setYear] = useState(2017)
  const [reportType, setReportType] = useState('comprehensive')
  const [copied, setCopied] = useState(false)

  const reportMutation = useGenerateReport()
  const [reportResult, setReportResult] = useState<ReportGenerateResponse | null>(null)

  if (!isOpen) return null

  const handleGenerate = async () => {
    try {
      const res = await reportMutation.mutateAsync({
        state,
        district,
        year,
        report_type: reportType
      })
      setReportResult(res)
    } catch (err) {
      // ignore
    }
  }

  const handleCopy = () => {
    if (!reportResult) return
    navigator.clipboard.writeText(reportResult.markdown_content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleDownloadMarkdown = () => {
    if (!reportResult) return
    const blob = new Blob([reportResult.markdown_content], { type: 'text/markdown;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${reportResult.report_id}.md`
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm animate-in fade-in duration-200">
      <Card className="w-full max-w-4xl max-h-[90vh] flex flex-col border-border shadow-2xl overflow-hidden">
        {/* Modal Header */}
        <CardHeader className="p-4 border-b border-border bg-muted/30 flex flex-row items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-sky-500 text-white flex items-center justify-center">
              <FileText className="w-4 h-4" />
            </div>
            <div>
              <CardTitle className="text-sm font-bold">AI Intelligence Report Generator</CardTitle>
              <CardDescription className="text-xs">
                Export evidence-grounded markdown decision reports
              </CardDescription>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted"
          >
            <X className="w-5 h-5" />
          </button>
        </CardHeader>

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-5 text-xs">
          {/* Configuration Form */}
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 p-3.5 rounded-xl border border-border bg-card">
            <div className="space-y-1">
              <label className="font-semibold">Target State</label>
              <select
                value={state}
                onChange={(e) => setState(e.target.value)}
                className="w-full rounded-lg border border-border bg-background px-2.5 py-1.5 text-xs font-medium"
              >
                {availableStates.map(s => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>

            <div className="space-y-1">
              <label className="font-semibold">District</label>
              <input
                type="text"
                value={district}
                onChange={(e) => setDistrict(e.target.value)}
                className="w-full rounded-lg border border-border bg-background px-2.5 py-1.5 text-xs font-medium"
              />
            </div>

            <div className="space-y-1">
              <label className="font-semibold">Year</label>
              <select
                value={year}
                onChange={(e) => setYear(Number(e.target.value))}
                className="w-full rounded-lg border border-border bg-background px-2.5 py-1.5 text-xs font-mono"
              >
                {[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017].map(y => (
                  <option key={y} value={y}>{y}</option>
                ))}
              </select>
            </div>

            <div className="flex items-end">
              <Button
                onClick={handleGenerate}
                disabled={reportMutation.isPending}
                className="w-full gap-1.5 text-xs font-bold"
              >
                {reportMutation.isPending ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Sparkles className="w-3.5 h-3.5" />
                )}
                <span>Generate</span>
              </Button>
            </div>
          </div>

          {/* Generated Report Preview */}
          {reportResult ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="font-bold text-foreground text-xs uppercase tracking-wider">
                  Report Preview: {reportResult.report_id}
                </span>
                <div className="flex items-center gap-2">
                  <Button variant="outline" size="sm" onClick={handleCopy} className="gap-1 text-xs">
                    {copied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                    <span>{copied ? 'Copied' : 'Copy'}</span>
                  </Button>
                  <Button size="sm" onClick={handleDownloadMarkdown} className="gap-1 text-xs font-bold">
                    <Download className="w-3.5 h-3.5" />
                    <span>Download .md</span>
                  </Button>
                </div>
              </div>

              <div className="p-4 rounded-xl border border-border bg-muted/40 font-mono text-[11px] leading-relaxed whitespace-pre-wrap max-h-96 overflow-y-auto">
                {reportResult.markdown_content}
              </div>
            </div>
          ) : (
            <div className="py-12 text-center text-muted-foreground text-xs">
              Click <strong>Generate</strong> to create a multi-section structured intelligence report based on verified ICRISAT records.
            </div>
          )}
        </div>
      </Card>
    </div>
  )
}
