import React from 'react'
import { StateSpatialProfileResponse } from '../../types/geospatial'
import { X, MapPin, TrendingUp, AlertTriangle, ShieldCheck, ArrowRight, Activity, Calendar } from 'lucide-react'
import { Link } from 'react-router-dom'

interface RegionSideDrawerProps {
  profile: StateSpatialProfileResponse | null
  isLoading: boolean
  onClose: () => void
}

export const RegionSideDrawer: React.FC<RegionSideDrawerProps> = ({
  profile,
  isLoading,
  onClose,
}) => {
  if (!profile && !isLoading) return null

  return (
    <div className="fixed inset-y-0 right-0 w-full max-w-md bg-slate-900/95 border-l border-slate-800 shadow-2xl z-50 p-6 overflow-y-auto backdrop-blur-xl animate-in slide-in-from-right duration-300">
      <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-6">
        <div>
          <div className="flex items-center gap-2">
            <MapPin className="w-5 h-5 text-emerald-400" />
            <h2 className="text-xl font-bold text-white tracking-wide">
              {profile?.state || 'Loading Regional Profile...'}
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            {profile?.agro_zone} • {profile?.region}
          </p>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {isLoading || !profile ? (
        <div className="space-y-4 animate-pulse">
          <div className="h-24 bg-slate-800 rounded-xl" />
          <div className="h-40 bg-slate-800 rounded-xl" />
          <div className="h-40 bg-slate-800 rounded-xl" />
        </div>
      ) : (
        <div className="space-y-6">
          {/* Top KPI Cards */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl p-3.5">
              <span className="text-[11px] text-slate-400 block mb-1">State Average Yield</span>
              <div className="text-lg font-bold font-mono text-white">
                {profile.average_yield_kg_ha.toLocaleString()} <span className="text-xs font-normal text-slate-400">kg/ha</span>
              </div>
            </div>

            <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl p-3.5">
              <span className="text-[11px] text-slate-400 block mb-1">Agricultural Risk</span>
              <div className={`text-lg font-bold font-mono ${profile.risk_score >= 50 ? 'text-amber-400' : 'text-emerald-400'}`}>
                {profile.risk_score.toFixed(1)} <span className="text-xs font-normal text-slate-400">/ 100</span>
              </div>
            </div>
          </div>

          {/* Temporal Trend & Geographic Coordinates */}
          <div className="bg-slate-800/40 border border-slate-800 rounded-xl p-4 space-y-2 text-xs">
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span className="text-slate-400">Trend Direction:</span>
              <strong className="text-white">{profile.trend_direction} ({profile.theil_sen_slope > 0 ? '+' : ''}{profile.theil_sen_slope} kg/ha/yr)</strong>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span className="text-slate-400">Centroid Coordinates:</span>
              <span className="font-mono text-slate-300">{profile.lat.toFixed(4)}°N, {profile.lon.toFixed(4)}°E</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-400">Monitored Districts:</span>
              <span className="font-mono text-emerald-400 font-bold">{profile.district_count} Districts</span>
            </div>
          </div>

          {/* Multi-Horizon Forward Forecast */}
          <div>
            <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
              <Calendar className="w-3.5 h-3.5 text-sky-400" />
              Forward Yield Projections (2018–2020)
            </h4>
            <div className="space-y-2">
              {profile.forecasts.map((f) => (
                <div key={f.forecast_year} className="bg-slate-800/60 border border-slate-700/60 rounded-lg p-2.5 flex items-center justify-between text-xs">
                  <div>
                    <span className="font-bold text-white">{f.forecast_year}</span>{' '}
                    <span className="text-slate-400 text-[10px]">({f.horizon_years}-Yr Horizon)</span>
                  </div>
                  <div className="text-right">
                    <span className="font-mono font-bold text-sky-400">{f.predicted_yield.toLocaleString()} kg/ha</span>
                    <span className="text-[10px] text-slate-400 block">±{f.uncertainty_pct / 2}% spread</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Member Districts Breakdown */}
          <div>
            <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5 text-emerald-400" />
              District Yield Distribution ({profile.districts.length})
            </h4>
            <div className="max-h-56 overflow-y-auto space-y-1.5 pr-1">
              {profile.districts.slice(0, 15).map((d) => (
                <div
                  key={d.district_name}
                  className="bg-slate-800/40 hover:bg-slate-800/70 p-2 rounded-lg border border-slate-800 text-xs flex items-center justify-between transition-colors"
                >
                  <div>
                    <div className="font-medium text-white">{d.district_name}</div>
                    <div className="text-[10px] text-slate-400">
                      Volatility: ±{d.yield_volatility_pct}% • {d.trend_direction}
                    </div>
                  </div>
                  <div className="text-right font-mono">
                    <div className="font-bold text-emerald-300">{d.average_yield_kg_ha.toLocaleString()} kg/ha</div>
                    <div className={`text-[10px] ${d.district_yield_zscore_state < 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                      {d.district_yield_zscore_state > 0 ? '+' : ''}{d.district_yield_zscore_state}σ
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Quick Platform Action Buttons */}
          <div className="border-t border-slate-800 pt-4 space-y-2">
            <Link
              to={`/scenario`}
              className="w-full flex items-center justify-between px-3.5 py-2.5 rounded-lg bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/30 text-xs font-medium transition-colors"
            >
              <span>Simulate Climate Scenario for {profile.state}</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>

            <Link
              to={`/copilot`}
              className="w-full flex items-center justify-between px-3.5 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-medium transition-colors"
            >
              <span>Ask AI Copilot about {profile.state}</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>
      )}
    </div>
  )
}
