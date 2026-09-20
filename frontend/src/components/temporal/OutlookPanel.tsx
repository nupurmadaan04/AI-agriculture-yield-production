import React from 'react'
import { X, Sparkles, AlertTriangle, TrendingUp, ShieldAlert, Activity, ChevronRight, Loader2 } from 'lucide-react'
import { useForecastState, useAssessEarlyWarning } from '../../services/api'
import { ForecastChart } from './ForecastChart'
import { TrendIndicator } from './TrendIndicator'
import { WarningCard } from './WarningCard'
import { ForecastTable } from './ForecastTable'
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { Button } from '../ui/Button'
import { formatYield } from '../../lib/utils'

interface OutlookPanelProps {
  stateName: string
  isOpen: boolean
  onClose: () => void
}

export const OutlookPanel: React.FC<OutlookPanelProps> = ({
  stateName,
  isOpen,
  onClose
}) => {
  const { data: forecastData, isLoading: fcLoading } = useForecastState(stateName)
  const assessMutation = useAssessEarlyWarning()
  const [warningData, setWarningData] = React.useState<any>(null)

  React.useEffect(() => {
    if (stateName && isOpen) {
      assessMutation.mutateAsync({ state: stateName }).then(res => setWarningData(res)).catch(() => {})
    }
  }, [stateName, isOpen])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm animate-in fade-in duration-200">
      <Card className="w-full max-w-5xl max-h-[92vh] flex flex-col border-border shadow-2xl overflow-hidden">
        {/* Top Header */}
        <CardHeader className="p-4 border-b border-border bg-muted/30 flex flex-row items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-sky-500 text-white flex items-center justify-center">
              <Activity className="w-4 h-4" />
            </div>
            <div>
              <CardTitle className="text-base font-extrabold">{stateName} Temporal Drill-Down</CardTitle>
              <p className="text-xs text-muted-foreground">
                Multi-year historical trend, forward forecast trajectory, and early-warning signals
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted"
          >
            <X className="w-5 h-5" />
          </button>
        </CardHeader>

        {/* Modal Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 text-xs">
          {fcLoading || !forecastData ? (
            <div className="py-20 text-center space-y-2">
              <Loader2 className="w-6 h-6 animate-spin mx-auto text-sky-500" />
              <p className="text-xs text-muted-foreground">Loading state temporal profile...</p>
            </div>
          ) : (
            <>
              {/* Forecast & Trajectory Chart */}
              <div className="p-4 rounded-xl border border-border bg-card">
                <ForecastChart
                  historicalSeries={forecastData.historical_series}
                  forecasts={forecastData.forecasts}
                  stateName={stateName}
                />
              </div>

              {/* Multi-Horizon Table */}
              <div className="space-y-2">
                <span className="font-bold text-foreground text-xs uppercase tracking-wider block">
                  Forward Projections & Prediction Intervals (2018–2020)
                </span>
                <ForecastTable
                  forecasts={forecastData.forecasts}
                  latestObservedYield={forecastData.latest_observed_yield}
                />
              </div>

              {/* Warning Assessment Card */}
              {warningData && (
                <div className="space-y-2">
                  <span className="font-bold text-foreground text-xs uppercase tracking-wider block">
                    Early Warning Status
                  </span>
                  <WarningCard assessment={warningData} />
                </div>
              )}
            </>
          )}
        </div>
      </Card>
    </div>
  )
}
