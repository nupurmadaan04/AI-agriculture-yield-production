import React, { useState } from 'react'
import {
  useGeospatialOverview,
  useGeospatialStates,
  useGeospatialClusters,
  useGeospatialStateProfile,
} from '../services/api'
import { IndiaChoroplethMap } from '../components/geospatial/IndiaChoroplethMap'
import { SpatialClusterCard } from '../components/geospatial/SpatialClusterCard'
import { SpatialOutlierTable } from '../components/geospatial/SpatialOutlierTable'
import { RegionSideDrawer } from '../components/geospatial/RegionSideDrawer'
import {
  Compass,
  MapPin,
  Layers,
  TrendingUp,
  AlertTriangle,
  ShieldCheck,
  Search,
  ArrowUpDown,
  Filter,
  BarChart3,
  Globe,
  Sliders,
} from 'lucide-react'

type MetricType = 'yield' | 'risk' | 'anomaly' | 'forecast' | 'warning' | 'trend'

export const GeospatialIntelligence: React.FC = () => {
  const [selectedMetric, setSelectedMetric] = useState<MetricType>('yield')
  const [selectedState, setSelectedState] = useState<string | null>(null)
  const [selectedClusterId, setSelectedClusterId] = useState<number | null>(null)
  const [tableSearch, setTableSearch] = useState('')
  const [sortField, setSortField] = useState<'yield' | 'risk' | 'state' | 'trend'>('yield')
  const [sortAsc, setSortAsc] = useState(false)

  // API Queries
  const { data: overview, isLoading: isOverviewLoading } = useGeospatialOverview()
  const { data: states = [], isLoading: isStatesLoading } = useGeospatialStates()
  const { data: clusters = [], isLoading: isClustersLoading } = useGeospatialClusters()
  const { data: stateProfile, isLoading: isProfileLoading } = useGeospatialStateProfile(
    selectedState || ''
  )

  // Filter and sort states for the ranking table
  const filteredStates = React.useMemo(() => {
    let list = [...states]

    if (selectedClusterId !== null) {
      list = list.filter((s) => s.cluster_id === selectedClusterId)
    }

    if (tableSearch.trim()) {
      const q = tableSearch.toLowerCase()
      list = list.filter(
        (s) =>
          s.state.toLowerCase().includes(q) ||
          s.region.toLowerCase().includes(q) ||
          s.agro_zone.toLowerCase().includes(q)
      )
    }

    list.sort((a, b) => {
      let vA = 0
      let vB = 0
      if (sortField === 'yield') {
        vA = a.average_yield_kg_ha
        vB = b.average_yield_kg_ha
      } else if (sortField === 'risk') {
        vA = a.risk_score
        vB = b.risk_score
      } else if (sortField === 'trend') {
        vA = a.theil_sen_slope
        vB = b.theil_sen_slope
      } else {
        return sortAsc ? a.state.localeCompare(b.state) : b.state.localeCompare(a.state)
      }
      return sortAsc ? vA - vB : vB - vA
    })

    return list
  }, [states, selectedClusterId, tableSearch, sortField, sortAsc])

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 md:p-8 space-y-8">
      {/* 1. Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800/80 pb-6">
        <div>
          <div className="flex items-center gap-3 mb-1.5">
            <div className="p-2.5 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
              <Compass className="w-7 h-7" />
            </div>
            <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight text-white">
              Geospatial Agricultural Intelligence
            </h1>
          </div>
          <p className="text-sm text-slate-400 max-w-3xl leading-relaxed">
            Explore the spatial distribution of yield, risk, anomalies, and multi-horizon forecast signals across India's 20 agricultural states and 311 monitored districts.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs font-mono px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">
            CRS: <strong className="text-emerald-400">WGS84</strong> • 20 States • 311 Dists
          </span>
        </div>
      </div>

      {/* 2. Top Executive KPI Row */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
          <span className="text-xs text-slate-400 block mb-1">Monitored Regions</span>
          <div className="text-2xl font-bold font-mono text-white">
            {overview?.total_states_monitored || 20} <span className="text-xs font-normal text-slate-400">States</span>
          </div>
          <span className="text-[11px] text-slate-500 font-mono mt-1 block">
            {overview?.total_districts_monitored || 311} Districts
          </span>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
          <span className="text-xs text-slate-400 block mb-1">National Mean Yield</span>
          <div className="text-2xl font-bold font-mono text-emerald-400">
            {overview?.national_average_yield_kg_ha?.toLocaleString() || '2,745'}{' '}
            <span className="text-xs font-normal text-slate-400">kg/ha</span>
          </div>
          <span className="text-[11px] text-emerald-500 font-mono mt-1 block">
            Baseline (2010–2017)
          </span>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
          <span className="text-xs text-slate-400 block mb-1">Mean Agricultural Risk</span>
          <div className="text-2xl font-bold font-mono text-amber-400">
            {overview?.national_average_risk_score || '38.5'}{' '}
            <span className="text-xs font-normal text-slate-400">/ 100</span>
          </div>
          <span className="text-[11px] text-slate-400 font-mono mt-1 block">
            {overview?.high_risk_states_count || 3} Elevated States
          </span>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
          <span className="text-xs text-slate-400 block mb-1">Spatial Outliers</span>
          <div className="text-2xl font-bold font-mono text-sky-400">
            {overview?.spatial_outliers_count || 12}
          </div>
          <span className="text-[11px] text-slate-400 font-mono mt-1 block">
            |z| ≥ 1.8σ Within-State
          </span>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 col-span-2 sm:col-span-1">
          <span className="text-xs text-slate-400 block mb-1">Spatial Clusters</span>
          <div className="text-2xl font-bold font-mono text-purple-400">
            {overview?.clusters_count || 4}
          </div>
          <span className="text-[11px] text-purple-400 font-mono mt-1 block">
            KMeans Archetypes
          </span>
        </div>
      </div>

      {/* 3. Metric Controls & Map Workspace */}
      <div className="space-y-4">
        {/* Metric Switcher Controls */}
        <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-900/70 p-3 rounded-xl border border-slate-800">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-emerald-400" />
            <span className="text-xs font-bold uppercase text-slate-300 tracking-wider">
              Choropleth Metric:
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-1.5">
            {(
              [
                { id: 'yield', label: 'Yield (kg/ha)' },
                { id: 'risk', label: 'Agricultural Risk' },
                { id: 'anomaly', label: 'Anomaly Shock' },
                { id: 'forecast', label: 'Forward Forecast' },
                { id: 'warning', label: 'Early Warning' },
                { id: 'trend', label: 'Theil-Sen Trend' },
              ] as const
            ).map((m) => (
              <button
                key={m.id}
                onClick={() => setSelectedMetric(m.id)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  selectedMetric === m.id
                    ? 'bg-emerald-600 text-white shadow-md shadow-emerald-900/30'
                    : 'bg-slate-800/80 text-slate-300 hover:bg-slate-800 hover:text-white'
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>

          {selectedClusterId !== null && (
            <button
              onClick={() => setSelectedClusterId(null)}
              className="text-xs text-emerald-400 hover:underline font-mono"
            >
              Clear Cluster Filter
            </button>
          )}
        </div>

        {/* India Choropleth Interactive Map */}
        <IndiaChoroplethMap
          states={states}
          selectedMetric={selectedMetric}
          selectedState={selectedState}
          onSelectState={(name) => setSelectedState(name)}
        />
      </div>

      {/* 4. Spatial Clustering Archetypes Section */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-purple-400" />
            <h3 className="text-lg font-bold text-white">
              Unsupervised Regional Spatial Clusters (311 Districts)
            </h3>
          </div>
          <span className="text-xs text-slate-400 font-mono">
            Evaluated by Silhouette & Davies-Bouldin Indices
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {clusters.map((c) => (
            <SpatialClusterCard
              key={c.cluster_id}
              cluster={c}
              isSelected={selectedClusterId === c.cluster_id}
              onSelect={(id) =>
                setSelectedClusterId((prev) => (prev === id ? null : id))
              }
            />
          ))}
        </div>
      </div>

      {/* 5. Spatial Outliers Section */}
      <SpatialOutlierTable
        outliers={stateProfile?.spatial_outliers || []}
        onSelectDistrict={(st, dist) => setSelectedState(st)}
      />

      {/* 6. Regional Ranking & State Matrix Table */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 backdrop-blur-md">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-4">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-emerald-400" />
            <h3 className="text-base font-bold text-white">
              State Spatial Intelligence Matrix ({filteredStates.length})
            </h3>
          </div>

          <div className="relative w-full sm:w-64">
            <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              placeholder="Search states, regions..."
              value={tableSearch}
              onChange={(e) => setTableSearch(e.target.value)}
              className="w-full bg-slate-800/80 border border-slate-700 text-xs text-white rounded-lg pl-8 pr-3 py-1.5 focus:outline-none focus:border-emerald-500"
            />
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 font-mono text-[11px]">
                <th
                  onClick={() => {
                    setSortField('state')
                    setSortAsc(!sortAsc)
                  }}
                  className="py-2.5 px-3 cursor-pointer hover:text-white"
                >
                  <div className="flex items-center gap-1">
                    State / Region <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th
                  onClick={() => {
                    setSortField('yield')
                    setSortAsc(!sortAsc)
                  }}
                  className="py-2.5 px-3 cursor-pointer hover:text-white"
                >
                  <div className="flex items-center gap-1">
                    Mean Yield <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th
                  onClick={() => {
                    setSortField('risk')
                    setSortAsc(!sortAsc)
                  }}
                  className="py-2.5 px-3 cursor-pointer hover:text-white"
                >
                  <div className="flex items-center gap-1">
                    Risk Score <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th
                  onClick={() => {
                    setSortField('trend')
                    setSortAsc(!sortAsc)
                  }}
                  className="py-2.5 px-3 cursor-pointer hover:text-white"
                >
                  <div className="flex items-center gap-1">
                    Theil-Sen Slope <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th className="py-2.5 px-3">1-Yr Forecast</th>
                <th className="py-2.5 px-3">Agro-Climatic Zone</th>
                <th className="py-2.5 px-3 text-right">Drill-Down</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {filteredStates.map((s) => (
                <tr key={s.state} className="hover:bg-slate-800/40 transition-colors">
                  <td className="py-2.5 px-3">
                    <div className="font-bold text-white">{s.state}</div>
                    <div className="text-[10px] text-slate-400">{s.region}</div>
                  </td>
                  <td className="py-2.5 px-3 font-mono font-bold text-emerald-400">
                    {s.average_yield_kg_ha.toLocaleString()} kg/ha
                  </td>
                  <td className="py-2.5 px-3 font-mono">
                    <span
                      className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                        s.risk_score >= 50
                          ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                          : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                      }`}
                    >
                      {s.risk_score.toFixed(1)} ({s.risk_level})
                    </span>
                  </td>
                  <td className="py-2.5 px-3 font-mono text-slate-300">
                    {s.theil_sen_slope > 0 ? '+' : ''}
                    {s.theil_sen_slope.toFixed(1)} kg/ha/yr
                  </td>
                  <td className="py-2.5 px-3 font-mono text-sky-400">
                    {s.forecast_1yr_kg_ha.toLocaleString()} kg/ha
                  </td>
                  <td className="py-2.5 px-3 text-slate-400 text-[11px] truncate max-w-xs">
                    {s.agro_zone}
                  </td>
                  <td className="py-2.5 px-3 text-right">
                    <button
                      onClick={() => setSelectedState(s.state)}
                      className="px-3 py-1 text-[11px] rounded-lg bg-emerald-600/20 hover:bg-emerald-600/40 text-emerald-300 font-medium transition-colors border border-emerald-500/30"
                    >
                      Inspect
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 7. Region Side Drawer for Clicked State */}
      <RegionSideDrawer
        profile={stateProfile || null}
        isLoading={isProfileLoading}
        onClose={() => setSelectedState(null)}
      />
    </div>
  )
}
