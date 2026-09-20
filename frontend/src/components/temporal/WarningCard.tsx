import React from 'react'
import { ShieldAlert, AlertTriangle, CheckCircle, Info } from 'lucide-react'
import { EarlyWarningAssessResponse } from '../../types/temporal'
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card'
import { Badge } from '../ui/Badge'

interface WarningCardProps {
  assessment: EarlyWarningAssessResponse
}

export const WarningCard: React.FC<WarningCardProps> = ({ assessment }) => {
  const { warning_score, severity, trigger_signals, components, state, district } = assessment

  const variant = severity === 'CRITICAL'
    ? 'destructive'
    : severity === 'HIGH'
    ? 'warning'
    : severity === 'MODERATE'
    ? 'blue'
    : 'success'

  return (
    <Card className="border-border bg-card overflow-hidden">
      <CardHeader className="pb-3 border-b border-border bg-muted/20">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ShieldAlert className={`w-4 h-4 ${severity === 'CRITICAL' || severity === 'HIGH' ? 'text-rose-500' : 'text-emerald-500'}`} />
            <CardTitle className="text-sm font-bold">
              {state} Early Warning Assessment
            </CardTitle>
          </div>
          <Badge variant={variant} className="text-[10px] font-bold">
            {severity} SEVERITY
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="p-4 space-y-4 text-xs">
        {/* Score and meter */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">
              Early Warning Indicator Score (0–100)
            </span>
            <span className="font-mono font-black text-base text-foreground">
              {warning_score.toFixed(1)} / 100
            </span>
          </div>
          <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
            <div
              className={`h-2 rounded-full transition-all ${
                severity === 'CRITICAL' ? 'bg-rose-500' : severity === 'HIGH' ? 'bg-amber-500' : severity === 'MODERATE' ? 'bg-sky-500' : 'bg-emerald-500'
              }`}
              style={{ width: `${Math.min(100, warning_score)}%` }}
            />
          </div>
        </div>

        {/* Trigger Signals */}
        <div className="space-y-1.5">
          <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider block">
            Trigger Signals & Factor Attribution:
          </span>
          <div className="space-y-1">
            {trigger_signals.map((sig, idx) => (
              <div key={idx} className="p-2 rounded-lg bg-muted/40 text-[11px] text-foreground flex items-start gap-2">
                <span className="text-sky-500 font-bold shrink-0 mt-0.5">•</span>
                <span>{sig}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Score Component Breakdown */}
        {components && (
          <div className="pt-2 border-t border-border grid grid-cols-2 sm:grid-cols-5 gap-2 text-[10px]">
            <div className="p-2 rounded bg-muted/20 space-y-0.5">
              <span className="text-muted-foreground block">Trend (30%)</span>
              <span className="font-mono font-bold text-foreground">{components.trend_signal_score}</span>
            </div>
            <div className="p-2 rounded bg-muted/20 space-y-0.5">
              <span className="text-muted-foreground block">Forecast (25%)</span>
              <span className="font-mono font-bold text-foreground">{components.forecast_signal_score}</span>
            </div>
            <div className="p-2 rounded bg-muted/20 space-y-0.5">
              <span className="text-muted-foreground block">Hist Dev (20%)</span>
              <span className="font-mono font-bold text-foreground">{components.historical_deviation_score}</span>
            </div>
            <div className="p-2 rounded bg-muted/20 space-y-0.5">
              <span className="text-muted-foreground block">Anomaly (15%)</span>
              <span className="font-mono font-bold text-foreground">{components.anomaly_signal_score}</span>
            </div>
            <div className="p-2 rounded bg-muted/20 space-y-0.5">
              <span className="text-muted-foreground block">Spread (10%)</span>
              <span className="font-mono font-bold text-foreground">{components.prediction_spread_score}</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
