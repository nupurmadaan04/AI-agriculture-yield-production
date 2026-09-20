import React from 'react'
import { Card, CardContent } from '../ui/Card'
import { cn } from '../../lib/utils'
import { TrendingUp, TrendingDown } from 'lucide-react'

export interface KpiCardProps {
  title: string
  value: string | number
  unit?: string
  trend?: {
    value: number
    label: string
    isPositive?: boolean
  }
  description?: string
  icon: React.ReactNode
  variant?: 'default' | 'primary' | 'emerald' | 'amber'
}

export const KpiCard: React.FC<KpiCardProps> = ({
  title,
  value,
  unit,
  trend,
  description,
  icon,
  variant = 'default'
}) => {
  return (
    <Card className="overflow-hidden relative group">
      <CardContent className="p-5">
        <div className="flex items-start justify-between">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">{title}</p>
          <div className={cn(
            "p-2 rounded-lg transition-colors",
            variant === 'default' && "bg-muted text-foreground",
            variant === 'primary' && "bg-sky-500/10 text-sky-600 dark:text-sky-400",
            variant === 'emerald' && "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400",
            variant === 'amber' && "bg-amber-500/10 text-amber-600 dark:text-amber-400",
          )}>
            {icon}
          </div>
        </div>

        <div className="mt-3 flex items-baseline gap-1.5">
          <span className="text-2xl font-bold tracking-tight text-foreground">{value}</span>
          {unit && <span className="text-xs font-medium text-muted-foreground">{unit}</span>}
        </div>

        {(trend || description) && (
          <div className="mt-2.5 flex items-center justify-between text-xs pt-2 border-t border-border/40">
            {trend && (
              <div className={cn(
                "flex items-center font-medium gap-1",
                trend.isPositive !== false ? "text-emerald-600 dark:text-emerald-400" : "text-red-500"
              )}>
                {trend.isPositive !== false ? <TrendingUp className="w-3.5 h-3.5" /> : <TrendingDown className="w-3.5 h-3.5" />}
                <span>{trend.value > 0 ? `+${trend.value}%` : `${trend.value}%`}</span>
                <span className="text-muted-foreground font-normal ml-0.5">{trend.label}</span>
              </div>
            )}
            {description && !trend && (
              <span className="text-muted-foreground">{description}</span>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
