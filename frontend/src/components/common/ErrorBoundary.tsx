import React, { Component, ErrorInfo, ReactNode } from 'react'
import { AlertCircle, RefreshCw, Home } from 'lucide-react'
import { Button } from '../ui/Button'

interface Props {
  children: ReactNode
  fallbackTitle?: string
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  }

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error caught by ErrorBoundary:', error, errorInfo)
  }

  private handleRetry = () => {
    this.setState({ hasError: false, error: null })
    window.location.reload()
  }

  private handleGoHome = () => {
    this.setState({ hasError: false, error: null })
    window.location.href = '/'
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-[400px] flex flex-col items-center justify-center p-8 text-center bg-card border border-destructive/20 rounded-2xl m-6 shadow-sm">
          <div className="p-4 bg-destructive/10 text-destructive rounded-full mb-4">
            <AlertCircle className="w-10 h-10" />
          </div>
          <h2 className="text-xl font-bold text-foreground mb-2">
            {this.props.fallbackTitle || "Something went wrong while loading this analytical module."}
          </h2>
          <p className="text-sm text-muted-foreground max-w-md mb-6">
            The platform encountered an unexpected issue while rendering this section.
            Please retry the operation or return to the main dashboard.
          </p>
          <div className="flex items-center gap-3">
            <Button variant="default" size="sm" onClick={this.handleRetry} className="gap-2">
              <RefreshCw className="w-4 h-4" />
              Retry Operation
            </Button>
            <Button variant="outline" size="sm" onClick={this.handleGoHome} className="gap-2">
              <Home className="w-4 h-4" />
              Return to Dashboard
            </Button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
