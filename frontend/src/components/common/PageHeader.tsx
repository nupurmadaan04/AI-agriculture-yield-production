import React from 'react'
import { Badge } from '../ui/Badge'

interface PageHeaderProps {
  title: string
  subtitle?: string
  badge?: string
  children?: React.ReactNode
}

export const PageHeader: React.FC<PageHeaderProps> = ({ title, subtitle, badge, children }) => {
  return (
    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-border/50 mb-6">
      <div>
        <div className="flex items-center gap-2.5">
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-foreground">{title}</h1>
          {badge && <Badge variant="blue">{badge}</Badge>}
        </div>
        {subtitle && <p className="text-sm text-muted-foreground mt-1.5 max-w-3xl leading-relaxed">{subtitle}</p>}
      </div>
      {children && <div className="flex items-center gap-3 shrink-0">{children}</div>}
    </div>
  )
}
