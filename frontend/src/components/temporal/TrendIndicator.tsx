import React from 'react'
import { TrendingUp, TrendingDown, Minus, ArrowUpRight, ArrowDownRight } from 'lucide-react'
import { Badge } from '../ui/Badge'

interface TrendIndicatorProps {
  direction: string
  theilSenSlope: number
  pVal?: number
  significance?: string
  compact?: boolean
}

export const TrendIndicator: React.FC<TrendIndicatorProps> = ({
  direction,
  theilSenSlope,
  pVal,
  significance,
  compact = false
}) => {
  const isUp = direction.includes('INCREASING')
  const isDown = direction.includes('DECREASING')

  const badgeVariant = isUp
    ? 'success'
    : isDown
    ? 'destructive'
    : 'blue'

  if (compact) {
    return (
      <div className="inline-flex items-center gap-1.5 font-mono text-xs">
        <Badge variant={badgeVariant} className="text-[10px] gap-1 px-1.5 py-0.5">
          {isUp && <ArrowUpRight className="w-3 h-3 text-emerald-500" />}
          {isDown && <ArrowDownRight className="w-3 h-3 text-rose-500" />}
          {!isUp && !isDown && <Minus className="w-3 h-3 text-sky-500" />}
          <span>{direction}</span>
        </Badge>
        <span className={theilSenSlope > 0 ? 'text-emerald-600 dark:text-emerald-400 font-bold' : theilSenSlope < 0 ? 'text-rose-600 dark:text-rose-400 font-bold' : 'text-muted-foreground'}>
          {theilSenSlope > 0 ? `+${theilSenSlope.toFixed(1)}` : theilSenSlope.toFixed(1)} kg/ha/yr
        </span>
      </div>
    )
  }

  return (
    <div className="p-3 rounded-xl border border-border bg-card/60 space-y-1.5 text-xs">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">
          Statistical Trend Direction
        </span>
        <Badge variant={badgeVariant} className="text-[10px]">
          {direction}
        </Badge>
      </div>

      <div className="flex items-baseline gap-2">
        <div className="text-xl font-black font-mono text-foreground">
          {theilSenSlope > 0 ? `+${theilSenSlope.toFixed(2)}` : theilSenSlope.toFixed(2)}
        </div>
        <span className="text-[11px] text-muted-foreground">kg/ha / year (Theil-Sen)</span>
      </div>

      {significance && (
        <p className="text-[10px] text-muted-foreground">
          Mann-Kendall: <strong>{significance}</strong> {pVal !== undefined ? `(p = ${pVal.toFixed(4)})` : ''}
        </p>
      )}
    </div>
  )
}
