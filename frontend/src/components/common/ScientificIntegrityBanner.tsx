import React from 'react'
import { Info, AlertTriangle } from 'lucide-react'
import { cn } from '../../lib/utils'

interface ScientificIntegrityBannerProps {
  type?: 'info' | 'warning'
  title?: string
  description?: string
  className?: string
}

export const ScientificIntegrityBanner: React.FC<ScientificIntegrityBannerProps> = ({
  type = 'info',
  title = "Scientific Integrity & Target Formulation Notice",
  description = "In agricultural panel data, crop yield is algebraically defined as (Production / Area) × 1000. When post-harvest production is included as a predictor, models perform mathematical reconstruction rather than independent pre-season forecasting. Pre-season forecasting requires bio-physical weather and soil variables.",
  className
}) => {
  const isWarning = type === 'warning'

  return (
    <div
      className={cn(
        "rounded-xl p-4 border flex items-start gap-3.5 my-4",
        isWarning
          ? "bg-amber-500/10 border-amber-500/20 text-amber-900 dark:text-amber-200"
          : "bg-sky-500/10 border-sky-500/20 text-sky-950 dark:text-sky-200",
        className
      )}
    >
      <div className={cn("p-1 rounded-md shrink-0 mt-0.5", isWarning ? "text-amber-600 dark:text-amber-400" : "text-sky-600 dark:text-sky-400")}>
        {isWarning ? <AlertTriangle className="w-5 h-5" /> : <Info className="w-5 h-5" />}
      </div>
      <div className="space-y-1 text-xs">
        <h4 className="font-semibold text-sm leading-tight">{title}</h4>
        <p className="opacity-90 leading-relaxed">{description}</p>
      </div>
    </div>
  )
}
