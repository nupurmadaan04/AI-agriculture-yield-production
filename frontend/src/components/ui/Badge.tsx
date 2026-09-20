import React from 'react'
import { cn } from '../../lib/utils'

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'secondary' | 'outline' | 'success' | 'warning' | 'destructive' | 'blue' | 'purple' | 'green' | 'amber' | 'cyan' | 'red' | 'rose'
}

export const Badge: React.FC<BadgeProps> = ({ className, variant = 'default', ...props }) => {
  return (
    <div
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
        {
          'bg-primary text-primary-foreground': variant === 'default',
          'bg-secondary text-secondary-foreground': variant === 'secondary',
          'border border-border text-foreground': variant === 'outline',
          'bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border border-emerald-500/20': variant === 'success' || variant === 'green',
          'bg-amber-500/15 text-amber-700 dark:text-amber-400 border border-amber-500/20': variant === 'warning' || variant === 'amber',
          'bg-red-500/15 text-red-700 dark:text-red-400 border border-red-500/20': variant === 'destructive' || variant === 'red',
          'bg-rose-500/15 text-rose-700 dark:text-rose-400 border border-rose-500/20': variant === 'rose',
          'bg-sky-500/15 text-sky-700 dark:text-sky-400 border border-sky-500/20': variant === 'blue',
          'bg-purple-500/15 text-purple-700 dark:text-purple-400 border border-purple-500/20': variant === 'purple',
          'bg-cyan-500/15 text-cyan-700 dark:text-cyan-400 border border-cyan-500/20': variant === 'cyan',
        },
        className
      )}
      {...props}
    />
  )
}
