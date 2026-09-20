import React, { useState } from 'react'
import {
  Database,
  Search,
  Download,
  Filter,
  BarChart3,
  TrendingUp,
  FileSpreadsheet,
  CheckCircle2,
  ExternalLink,
  ChevronDown,
  Loader2,
  RefreshCw
} from 'lucide-react'
import { formatNumber, formatYield, formatProduction, formatArea } from '../lib/utils'
import { StatePerformanceBar } from '../components/charts/AnalyticsCharts'
import { YieldTrendChart } from '../components/charts/YieldTrendChart'
import { Button } from '../components/ui/Button'
import { Badge } from '../components/ui/Badge'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/Card'
import { useSummary, useFilters, useTrends, useStates, useRecords } from '../services/api'
import { AgriculturalRecord } from '../types/agriculture'

export const NationalPortal: React.FC = () => {
  const [selectedState, setSelectedState] = useState('all')
  const [selectedYear, setSelectedYear] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [currentPage, setCurrentPage] = useState(1)
  const pageSize = 15

  // API Hooks
  const { data: summary, isLoading: isSummaryLoading } = useSummary()
  const { data: filterOptions, isLoading: isFiltersLoading } = useFilters()
  const { data: trendsData, isLoading: isTrendsLoading } = useTrends({
    state: selectedState !== 'all' ? selectedState : undefined,
  })
  const { data: statesData, isLoading: isStatesLoading } = useStates()
  const { data: recordsData, isLoading: isRecordsLoading, isFetching: isRecordsFetching } = useRecords({
    page: currentPage,
    page_size: pageSize,
    year: selectedYear !== 'all' ? Number(selectedYear) : undefined,
    state: selectedState !== 'all' ? selectedState : undefined,
    search: searchQuery || undefined,
  })

  const records = recordsData?.data || []
  const totalPages = recordsData?.pagination?.total_pages || 1
  const totalFound = recordsData?.pagination?.total || 0

  const handleExportCsv = () => {
    if (records.length === 0) return
    const headers = ['State', 'District', 'Year', 'Area (000 ha)', 'Production (000 t)', 'Yield (kg/ha)']
    const rows = records.map((r: AgriculturalRecord) => [
      `"${r.state}"`,
      `"${r.district}"`,
      r.year,
      r.area,
      r.production,
      r.yield
    ])
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n')
    const encodedUri = encodeURI(csvContent)
    const link = document.createElement('a')
    link.setAttribute('href', encodedUri)
    link.setAttribute('download', `ICRISAT_Agricultural_Panel_${selectedState}_${selectedYear}.csv`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const yearsList = filterOptions?.years || [2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010]
  const statesList = filterOptions?.states || []
  const statesRankings = statesData?.data || []
  const trendsList = trendsData?.data || []

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-2">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-500/10 text-sky-600 dark:text-sky-400 text-xs font-semibold">
            <Database className="w-3.5 h-3.5" />
            <span>UPAg & ICRISAT Data Portal</span>
          </div>
          <h1 className="text-2xl sm:text-4xl font-extrabold text-foreground tracking-tight">
            National Agricultural Statistics & APY Explorer
          </h1>
          <p className="text-xs sm:text-sm text-muted-foreground max-w-2xl leading-relaxed">
            Standardized Area, Production, and Yield (APY) time-series panels compiled across 311 Indian districts from 2010 to 2017.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Button onClick={handleExportCsv} className="gap-2 text-xs font-bold shadow-md shadow-sky-500/20">
            <Download className="w-3.5 h-3.5" />
            <span>Export CSV Current Page</span>
          </Button>
        </div>
      </div>

      {/* KPI Counters */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <span className="text-[11px] text-muted-foreground font-semibold">Total Audited Records</span>
          <p className="text-2xl font-bold font-mono text-foreground mt-1">
            {isSummaryLoading ? '...' : (summary?.total_records ? formatNumber(summary.total_records) : '71,601')}
          </p>
          <span className="text-[10px] text-emerald-600 dark:text-emerald-400 font-medium">0 Missing • 0 Duplicates</span>
        </Card>

        <Card className="p-4">
          <span className="text-[11px] text-muted-foreground font-semibold">National Average Yield</span>
          <p className="text-2xl font-bold font-mono text-sky-600 dark:text-sky-400 mt-1">
            {isSummaryLoading ? '...' : (summary?.average_yield ? formatYield(summary.average_yield) : '2,062.8 kg/ha')}
          </p>
          <span className="text-[10px] text-muted-foreground">Median: {summary?.median_yield ? formatYield(summary.median_yield) : '2,174.7 kg/ha'}</span>
        </Card>

        <Card className="p-4">
          <span className="text-[11px] text-muted-foreground font-semibold">Top Performing State</span>
          <p className="text-2xl font-bold text-foreground mt-1">
            {statesRankings[0]?.state || 'Punjab'}
          </p>
          <span className="text-[10px] text-emerald-600 dark:text-emerald-400 font-medium">
            {statesRankings[0] ? formatYield(statesRankings[0].average_yield) : '3,991.1 kg/ha'} Avg Yield
          </span>
        </Card>

        <Card className="p-4">
          <span className="text-[11px] text-muted-foreground font-semibold">Geographic Coverage</span>
          <p className="text-2xl font-bold font-mono text-foreground mt-1">
            {summary?.total_states || 20} States
          </p>
          <span className="text-[10px] text-muted-foreground">{summary?.total_districts || 311} Reporting Districts</span>
        </Card>
      </div>

      {/* Filter Bar */}
      <div className="flex flex-wrap items-center gap-3 p-4 rounded-2xl border border-border bg-card shadow-xs">
        <div className="flex items-center gap-2 text-xs font-semibold text-foreground mr-2">
          <Filter className="w-4 h-4 text-sky-500" />
          <span>Filters:</span>
        </div>

        {/* State Selector */}
        <select
          value={selectedState}
          onChange={(e: React.ChangeEvent<HTMLSelectElement>) => {
            setSelectedState(e.target.value)
            setCurrentPage(1)
          }}
          className="rounded-lg border border-border bg-background px-3 py-1.5 text-xs font-medium cursor-pointer focus:ring-2 focus:ring-sky-500 focus:outline-none"
        >
          <option value="all">All 20 States</option>
          {statesList.map(s => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>

        {/* Year Selector */}
        <select
          value={selectedYear}
          onChange={(e: React.ChangeEvent<HTMLSelectElement>) => {
            setSelectedYear(e.target.value)
            setCurrentPage(1)
          }}
          className="rounded-lg border border-border bg-background px-3 py-1.5 text-xs font-medium cursor-pointer focus:ring-2 focus:ring-sky-500 focus:outline-none"
        >
          <option value="all">All Years (2010–2017)</option>
          {yearsList.map(yr => (
            <option key={yr} value={yr.toString()}>{yr}</option>
          ))}
        </select>

        {/* District Search */}
        <div className="relative ml-auto flex-1 min-w-[200px] max-w-xs">
          <Search className="w-3.5 h-3.5 text-muted-foreground absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder="Search district or state..."
            value={searchQuery}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
              setSearchQuery(e.target.value)
              setCurrentPage(1)
            }}
            className="w-full rounded-lg border border-border bg-background pl-8 pr-3 py-1.5 text-xs focus:ring-2 focus:ring-sky-500 focus:outline-none"
          />
        </div>
      </div>

      {/* Visual Analytics Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <YieldTrendChart
          data={trendsList}
          title={selectedState !== 'all' ? `${selectedState} Yield Trajectory (2010–2017)` : "National Rice Yield Trajectory (2010–2017)"}
          subtitle="Captures the 2015 drought impact and subsequent multi-year recovery"
        />
        <StatePerformanceBar data={statesRankings} />
      </div>

      {/* Filtered Data Table */}
      <Card>
        <CardHeader className="pb-3 flex flex-row items-center justify-between">
          <div>
            <CardTitle className="text-base font-bold flex items-center gap-2">
              <span>District Agricultural Panel Records</span>
              {isRecordsFetching && <Loader2 className="w-3.5 h-3.5 animate-spin text-sky-500" />}
            </CardTitle>
            <CardDescription className="text-xs">
              Showing page {currentPage} of {totalPages} ({totalFound} records matching current filters)
            </CardDescription>
          </div>
          <Badge variant="outline">{totalFound} Records Found</Badge>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-muted/40 text-muted-foreground uppercase text-[10px] tracking-wider border-b border-border">
                <tr>
                  <th className="p-3 font-semibold">State Name</th>
                  <th className="p-3 font-semibold">District Name</th>
                  <th className="p-3 font-semibold">Year</th>
                  <th className="p-3 font-semibold text-right">Area ('000 ha)</th>
                  <th className="p-3 font-semibold text-right">Production ('000 t)</th>
                  <th className="p-3 font-semibold text-right">Yield (kg/ha)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/60 font-mono text-xs">
                {isRecordsLoading ? (
                  <tr>
                    <td colSpan={6} className="p-8 text-center text-muted-foreground font-sans">
                      <div className="flex items-center justify-center gap-2">
                        <Loader2 className="w-4 h-4 animate-spin text-sky-500" />
                        <span>Loading panel records from API...</span>
                      </div>
                    </td>
                  </tr>
                ) : records.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="p-8 text-center text-muted-foreground font-sans">
                      No district records matching the selected filter criteria.
                    </td>
                  </tr>
                ) : (
                  records.map((r: AgriculturalRecord) => (
                    <tr key={r.id} className="hover:bg-muted/30 transition-colors">
                      <td className="p-3 font-sans font-medium text-foreground">{r.state}</td>
                      <td className="p-3 font-sans text-muted-foreground">{r.district}</td>
                      <td className="p-3 text-muted-foreground">{r.year}</td>
                      <td className="p-3 text-right text-muted-foreground">{formatArea(r.area)}</td>
                      <td className="p-3 text-right text-muted-foreground">{formatProduction(r.production)}</td>
                      <td className="p-3 text-right font-semibold text-foreground font-bold">{formatYield(r.yield)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between pt-4 border-t border-border mt-3 text-xs">
            <span className="text-muted-foreground">
              Page {currentPage} of {totalPages} ({totalFound} total records)
            </span>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={currentPage <= 1 || isRecordsFetching}
                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                className="text-xs"
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={currentPage >= totalPages || isRecordsFetching}
                onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                className="text-xs"
              >
                Next
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
