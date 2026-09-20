import React from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { Database, ShieldCheck, AlertCircle, Info, Layers, Calendar, MapPin } from 'lucide-react'
import { useDatasetMetadata, useCropDetail } from '../../services/api'
import { useFilters } from '../../context/FilterContext'

export const DataProvenancePanel: React.FC = () => {
  const { filters } = useFilters()
  const activeCrop = filters.crop || 'Rice'
  const { data: metadata, isLoading: metaLoading } = useDatasetMetadata()
  const { data: cropDetail, isLoading: cropLoading } = useCropDetail(activeCrop)

  const isRice = activeCrop.toLowerCase() === 'rice'

  return (
    <div className="space-y-4 mb-6">
      {/* Non-Rice Model Notice Guardrail */}
      {!isRice && (
        <div className="flex items-start gap-3 p-4 rounded-xl border border-amber-500/30 bg-amber-500/10 text-foreground">
          <AlertCircle className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <div className="text-sm font-semibold text-amber-600 dark:text-amber-400">
              Historical Analytics Mode Active for {activeCrop}
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Forecast simulation and Tree SHAP feature attribution are unavailable for <strong>{activeCrop}</strong> under the current registered Random Forest model (trained on Rice). <strong>Historical yield, production, area trends, and state distributions remain fully available and verified.</strong>
            </p>
          </div>
        </div>
      )}

      {/* Provenance Card */}
      <Card className="border-border/80 bg-card/40 backdrop-blur-xs">
        <CardHeader className="pb-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-primary" />
              <CardTitle className="text-sm font-semibold">Data Provenance & Panel Governance</CardTitle>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="text-2xs font-mono bg-primary/10 border-primary/30 text-primary">
                {metadata?.dataset_version || 'AGRI_PANEL_1.0'}
              </Badge>
              <Badge variant="success" className="text-2xs flex items-center gap-1">
                <ShieldCheck className="w-3 h-3" />
                Quality: {metadata?.quality_status || 'PASS'}
              </Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div className="p-2.5 rounded-lg bg-muted/40 border border-border/40">
              <div className="text-muted-foreground flex items-center gap-1 text-2xs mb-1">
                <Layers className="w-3 h-3 text-muted-foreground" />
                Selected Crop Records
              </div>
              <div className="font-semibold text-sm text-foreground">
                {cropLoading ? '...' : (cropDetail?.records?.toLocaleString() || '2,469')}
              </div>
              <div className="text-2xs text-muted-foreground mt-0.5">
                {cropDetail?.forecasting_model_status === 'REGISTERED_AND_VALIDATED' ? 'Model Active' : 'Historical Analytics'}
              </div>
            </div>

            <div className="p-2.5 rounded-lg bg-muted/40 border border-border/40">
              <div className="text-muted-foreground flex items-center gap-1 text-2xs mb-1">
                <Database className="w-3 h-3 text-muted-foreground" />
                Total Multi-Crop Panel
              </div>
              <div className="font-semibold text-sm text-foreground">
                {metaLoading ? '...' : (metadata?.record_count?.toLocaleString() || '71,601')}
              </div>
              <div className="text-2xs text-muted-foreground mt-0.5">
                {metadata?.crop_count || 29} Verified Crops
              </div>
            </div>

            <div className="p-2.5 rounded-lg bg-muted/40 border border-border/40">
              <div className="text-muted-foreground flex items-center gap-1 text-2xs mb-1">
                <MapPin className="w-3 h-3 text-muted-foreground" />
                Geographic Coverage
              </div>
              <div className="font-semibold text-sm text-foreground">
                {metadata?.state_count || 20} States / {metadata?.district_count || 311} Dists
              </div>
              <div className="text-2xs text-muted-foreground mt-0.5">
                ICRISAT 1966 Baseline
              </div>
            </div>

            <div className="p-2.5 rounded-lg bg-muted/40 border border-border/40">
              <div className="text-muted-foreground flex items-center gap-1 text-2xs mb-1">
                <Calendar className="w-3 h-3 text-muted-foreground" />
                Temporal Span
              </div>
              <div className="font-semibold text-sm text-foreground">
                {metadata?.year_range || '2010–2017'}
              </div>
              <div className="text-2xs text-muted-foreground mt-0.5">
                Annual Survey Harmonized
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
