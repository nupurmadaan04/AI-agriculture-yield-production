import React, { useState } from 'react'
import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { Button } from '../ui/Button'
import {
  RotateCcw,
  FlaskConical,
  CheckCircle2,
  ShieldCheck,
  AlertCircle
} from 'lucide-react'
import { useRunWarningBacktest } from '../../services/api'
import type { WarningBacktest } from '../../types/monitoring'

interface Props {
  initialData?: WarningBacktest
}

export const WarningBacktestPanel: React.FC<Props> = ({ initialData }) => {
  const [yieldDrop, setYieldDrop] = useState(-10.0)
  const [zScore, setZScore] = useState(1.2)
  const [leadTime, setLeadTime] = useState(1)
  const [backtestResult, setBacktestResult] = useState<WarningBacktest | undefined>(initialData)

  const { mutate: runBacktest, isPending } = useRunWarningBacktest()

  const handleExecuteBacktest = () => {
    runBacktest(
      {
        yield_drop_threshold_pct: yieldDrop,
        warning_zscore_threshold: zScore,
        lead_time_years: leadTime,
      },
      {
        onSuccess: (data) => {
          setBacktestResult(data)
        },
      }
    )
  }

  return (
    <Card className="p-5 border border-slate-800 bg-slate-900/60 backdrop-blur space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div>
          <h3 className="text-base font-semibold text-slate-100 flex items-center gap-2">
            <FlaskConical className="w-5 h-5 text-amber-400" />
            Early Warning Historical Backtesting Engine
          </h3>
          <p className="text-xs text-slate-400">
            Chronological step-forward evaluation of warning decision rules vs subsequent observed outcomes
          </p>
        </div>

        <Badge variant="cyan" className="text-xs">
          STRICT TEMPORAL INTEGRITY (t → t+{leadTime})
        </Badge>
      </div>

      {/* Interactive Parameter Controls */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 p-3.5 bg-slate-950/60 rounded-xl border border-slate-800">
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-slate-300">
            <span>Adverse Yield Drop</span>
            <span className="font-mono font-bold text-amber-300">{yieldDrop}%</span>
          </div>
          <input
            type="range"
            min="-30"
            max="-5"
            step="1"
            value={yieldDrop}
            onChange={(e) => setYieldDrop(Number(e.target.value))}
            className="w-full accent-amber-500 cursor-pointer"
          />
          <div className="text-[10px] text-slate-500">Subsequent contraction defining adverse event</div>
        </div>

        <div className="space-y-1">
          <div className="flex justify-between text-xs text-slate-300">
            <span>Warning Trigger (Z-score)</span>
            <span className="font-mono font-bold text-cyan-300">≥ {zScore} std</span>
          </div>
          <input
            type="range"
            min="0.5"
            max="3.0"
            step="0.1"
            value={zScore}
            onChange={(e) => setZScore(Number(e.target.value))}
            className="w-full accent-cyan-500 cursor-pointer"
          />
          <div className="text-[10px] text-slate-500">Historical deviation required to trigger warning</div>
        </div>

        <div className="space-y-1">
          <div className="flex justify-between text-xs text-slate-300">
            <span>Evaluation Lead Time</span>
            <span className="font-mono font-bold text-purple-300">{leadTime} Year(s)</span>
          </div>
          <div className="flex gap-2">
            {[1, 2, 3].map((yr) => (
              <button
                key={yr}
                onClick={() => setLeadTime(yr)}
                className={`flex-1 py-1 text-xs font-semibold rounded border transition-all ${
                  leadTime === yr
                    ? 'bg-purple-500/20 text-purple-300 border-purple-500/50'
                    : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                }`}
              >
                {yr} Yr
              </button>
            ))}
          </div>
          <div className="text-[10px] text-slate-500">Observation gap between t and evaluation</div>
        </div>
      </div>

      {/* Execution Button */}
      <div className="flex justify-end">
        <Button
          onClick={handleExecuteBacktest}
          disabled={isPending}
          className="text-xs gap-1.5 bg-emerald-600 hover:bg-emerald-500"
        >
          <RotateCcw className={`w-4 h-4 ${isPending ? 'animate-spin' : ''}`} />
          {isPending ? 'Executing Backtest...' : 'Execute Historical Backtest'}
        </Button>
      </div>

      {/* Backtest Results Cards */}
      {backtestResult && (
        <div className="space-y-3 pt-2">
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            <div className="p-3 bg-slate-950/40 rounded-xl border border-slate-800 text-center">
              <div className="text-[10px] text-slate-400">Precision</div>
              <div className="text-lg font-bold font-mono text-emerald-400">
                {backtestResult.precision}%
              </div>
              <div className="text-[10px] text-slate-500">TP / (TP + FP)</div>
            </div>

            <div className="p-3 bg-slate-950/40 rounded-xl border border-slate-800 text-center">
              <div className="text-[10px] text-slate-400">Recall / Sensitivity</div>
              <div className="text-lg font-bold font-mono text-cyan-400">
                {backtestResult.recall}%
              </div>
              <div className="text-[10px] text-slate-500">TP / (TP + FN)</div>
            </div>

            <div className="p-3 bg-slate-950/40 rounded-xl border border-slate-800 text-center">
              <div className="text-[10px] text-slate-400">F1-Score</div>
              <div className="text-lg font-bold font-mono text-purple-400">
                {backtestResult.f1_score}%
              </div>
              <div className="text-[10px] text-slate-500">Harmonic balance</div>
            </div>

            <div className="p-3 bg-slate-950/40 rounded-xl border border-slate-800 text-center">
              <div className="text-[10px] text-slate-400">False Positive Rate</div>
              <div className="text-lg font-bold font-mono text-amber-400">
                {backtestResult.false_positive_rate}%
              </div>
              <div className="text-[10px] text-slate-500">FP / (FP + TN)</div>
            </div>

            <div className="p-3 bg-slate-950/40 rounded-xl border border-slate-800 text-center">
              <div className="text-[10px] text-slate-400">Mean Lead Time</div>
              <div className="text-lg font-bold font-mono text-slate-100">
                {backtestResult.mean_lead_time_years} Yr
              </div>
              <div className="text-[10px] text-slate-500">{backtestResult.total_evaluations.toLocaleString()} evals</div>
            </div>
          </div>

          <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800/80 text-[11px] text-slate-400 flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <div>
              <strong className="text-slate-300">Chronological Integrity Verified ({backtestResult.evaluation_years_range}):</strong>{' '}
              {backtestResult.scientific_disclaimer}
            </div>
          </div>
        </div>
      )}
    </Card>
  )
}
