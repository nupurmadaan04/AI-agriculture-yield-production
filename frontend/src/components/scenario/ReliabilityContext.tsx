import React from 'react'
import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { ShieldCheck, Info, Database } from 'lucide-react'
import { ValidationContext } from '../../types/scenario'

interface ReliabilityContextProps {
  context?: ValidationContext
  disclaimer?: string
}

export const ReliabilityContext: React.FC<ReliabilityContextProps> = ({
  context = {
    model_version: 'exogenous_rf_forecaster_v2.1.0',
    dataset_version: 'ICRISAT_District_Level_Data_1966_2017_Cleaned_v1.0',
    validation_r2: 0.7866,
    validation_mae: 353.01,
    validation_rmse: 513.11,
    drift_status: 'NORMAL',
    data_quality_score: 100.0,
    spread_type: 'Random Forest ensemble prediction spread (P10-P90)'
  },
  disclaimer = 'Scenario outputs represent hypothetical model simulations under modified input assumptions. They must never be presented as guaranteed future outcomes.'
}) => {
  return (
    <Card className="p-5 border-blue-500/20 bg-blue-500/5 space-y-3">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border/40 pb-3">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-blue-400" />
          <h3 className="font-bold text-sm text-foreground">Underlying Model Reliability & Governance Context</h3>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="blue">{context.model_version}</Badge>
          <Badge variant={context.drift_status === 'NORMAL' ? 'green' : 'amber'}>
            Drift: {context.drift_status}
          </Badge>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
        <div className="p-2.5 rounded bg-card/60 border border-border/50">
          <div className="text-muted-foreground text-[10px] uppercase font-mono">Out-of-Time Test R²</div>
          <div className="text-sm font-bold font-mono text-emerald-400 mt-0.5">{context.validation_r2}</div>
          <div className="text-[10px] text-muted-foreground">Unseen 2016–2017</div>
        </div>

        <div className="p-2.5 rounded bg-card/60 border border-border/50">
          <div className="text-muted-foreground text-[10px] uppercase font-mono">Test MAE</div>
          <div className="text-sm font-bold font-mono text-blue-400 mt-0.5">{context.validation_mae} kg/ha</div>
          <div className="text-[10px] text-muted-foreground">RMSE: {context.validation_rmse} kg/ha</div>
        </div>

        <div className="p-2.5 rounded bg-card/60 border border-border/50">
          <div className="text-muted-foreground text-[10px] uppercase font-mono">Data Quality</div>
          <div className="text-sm font-bold font-mono text-cyan-400 mt-0.5">{context.data_quality_score}/100</div>
          <div className="text-[10px] text-muted-foreground">4-Pillar Audit Grade</div>
        </div>

        <div className="p-2.5 rounded bg-card/60 border border-border/50">
          <div className="text-muted-foreground text-[10px] uppercase font-mono">Spread Interpretation</div>
          <div className="text-[11px] font-medium text-foreground mt-0.5 truncate">P10–P90 Ensemble</div>
          <div className="text-[10px] text-amber-400/90">Not a confidence interval</div>
        </div>
      </div>

      <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-xs text-amber-200/90 flex items-start gap-2">
        <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-amber-300">Scientific Integrity Notice: </span>
          {disclaimer}
        </div>
      </div>
    </Card>
  )
}
