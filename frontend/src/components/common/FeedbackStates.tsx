import React from 'react'
import { FolderSearch, AlertCircle, RefreshCw, Download, ChevronRight, Home } from 'lucide-react'
import { Button } from '../ui/Button'
import { Skeleton } from '../ui/Skeleton'
import { Link, useLocation } from 'react-router-dom'

export const EmptyState: React.FC<{
  title?: string
  description?: string
  actionLabel?: string
  onAction?: () => void
  icon?: React.ReactNode
}> = ({
  title = "No agricultural records found",
  description = "Try adjusting your filters or resetting the search criteria.",
  actionLabel,
  onAction,
  icon
}) => {
  return (
    <div className="flex flex-col items-center justify-center p-12 text-center border border-dashed border-border rounded-xl bg-card/50">
      <div className="p-3 bg-muted text-muted-foreground rounded-full mb-3">
        {icon || <FolderSearch className="w-8 h-8" />}
      </div>
      <h3 className="text-base font-semibold text-foreground">{title}</h3>
      <p className="text-sm text-muted-foreground max-w-sm mt-1 mb-4">{description}</p>
      {actionLabel && onAction && (
        <Button variant="outline" size="sm" onClick={onAction}>
          {actionLabel}
        </Button>
      )}
    </div>
  )
}

export const LoadingState: React.FC<{ message?: string }> = ({ message = "Loading agricultural intelligence..." }) => {
  return (
    <div className="space-y-4 p-6" role="status" aria-live="polite" aria-label={message}>
      <div className="flex items-center gap-3 text-sm text-muted-foreground">
        <RefreshCw className="w-4 h-4 animate-spin text-primary" aria-hidden="true" />
        <span>{message}</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Skeleton className="h-28" />
        <Skeleton className="h-28" />
        <Skeleton className="h-28" />
        <Skeleton className="h-28" />
      </div>
      <Skeleton className="h-72 w-full" />
    </div>
  )
}

export const ErrorState: React.FC<{
  title?: string
  message?: string
  onRetry?: () => void
}> = ({
  title = "Failed to load data",
  message = "An error occurred while connecting to the agricultural dataset.",
  onRetry
}) => {
  return (
    <div className="flex flex-col items-center justify-center p-8 text-center border border-destructive/20 bg-destructive/5 rounded-xl text-destructive">
      <AlertCircle className="w-8 h-8 mb-2" />
      <h3 className="font-semibold text-base">{title}</h3>
      <p className="text-xs text-muted-foreground mt-1 mb-4 max-w-md">{message}</p>
      {onRetry && (
        <Button variant="outline" size="sm" onClick={onRetry} className="gap-2">
          <RefreshCw className="w-3.5 h-3.5" />
          Retry
        </Button>
      )}
    </div>
  )
}

export const Breadcrumbs: React.FC = () => {
  const location = useLocation()
  const pathnames = location.pathname.split('/').filter(x => x)

  const formatName = (str: string) => {
    return str.charAt(0).toUpperCase() + str.slice(1).replace(/-/g, ' ')
  }

  return (
    <nav aria-label="Breadcrumb" className="flex items-center space-x-1.5 text-xs text-muted-foreground">
      <Link to="/" className="hover:text-foreground flex items-center gap-1 transition-colors">
        <Home className="w-3.5 h-3.5" aria-hidden="true" />
        <span>Intelligence</span>
      </Link>
      {pathnames.map((value, index) => {
        const to = `/${pathnames.slice(0, index + 1).join('/')}`
        const isLast = index === pathnames.length - 1

        return (
          <React.Fragment key={to}>
            <ChevronRight className="w-3 h-3 text-muted-foreground/50" aria-hidden="true" />
            {isLast ? (
              <span className="font-medium text-foreground" aria-current="page">{formatName(value)}</span>
            ) : (
              <Link to={to} className="hover:text-foreground transition-colors">
                {formatName(value)}
              </Link>
            )}
          </React.Fragment>
        )
      })}
    </nav>
  )
}

export const ExportButton: React.FC<{
  data: any[]
  filename?: string
  variant?: 'outline' | 'default'
}> = ({ data, filename = 'agricultural-data.csv', variant = 'outline' }) => {
  const handleExport = () => {
    if (!data || data.length === 0) return
    const headers = Object.keys(data[0]).join(',')
    const rows = data.map(obj => Object.values(obj).join(',')).join('\n')
    const csvContent = `data:text/csv;charset=utf-8,${headers}\n${rows}`
    const encodedUri = encodeURI(csvContent)
    const link = document.createElement('a')
    link.setAttribute('href', encodedUri)
    link.setAttribute('download', filename)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <Button variant={variant} size="sm" onClick={handleExport} className="gap-1.5 text-xs">
      <Download className="w-3.5 h-3.5" />
      Export CSV
    </Button>
  )
}
