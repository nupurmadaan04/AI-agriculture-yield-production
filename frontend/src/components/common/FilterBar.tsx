import React from 'react'
import { useFilters } from '../../context/FilterContext'
import { Select } from '../ui/Select'
import { Button } from '../ui/Button'
import { RotateCcw, Filter, Sprout, Database } from 'lucide-react'
import { useAgricultureCrops, useCropDetail } from '../../services/api'

export const FilterBar: React.FC = () => {
  const { filters, setCrop, setState, setYear, resetFilters } = useFilters()

  const { data: cropsData, isLoading: cropsLoading } = useAgricultureCrops()
  const { data: cropDetail } = useCropDetail(filters.crop || 'Rice')

  const availableCrops = cropsData?.crops || [
    { crop: 'Rice', records: 2469, first_year: 2010, last_year: 2017, states: 20, districts: 311, has_area: true, has_production: true, has_yield: true, forecasting_supported: true, scenario_supported: true, xai_supported: true },
    { crop: 'Wheat', records: 2469, first_year: 2010, last_year: 2017, states: 20, districts: 311, has_area: true, has_production: true, has_yield: true, forecasting_supported: false, scenario_supported: false, xai_supported: false },
    { crop: 'Maize', records: 2469, first_year: 2010, last_year: 2017, states: 20, districts: 311, has_area: true, has_production: true, has_yield: true, forecasting_supported: false, scenario_supported: false, xai_supported: false },
    { crop: 'Cotton', records: 2469, first_year: 2010, last_year: 2017, states: 20, districts: 311, has_area: true, has_production: true, has_yield: true, forecasting_supported: false, scenario_supported: false, xai_supported: false },
  ]

  const availableStates = cropDetail?.states && cropDetail.states.length > 0
    ? cropDetail.states
    : [
        'Andhra Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Gujarat', 'Haryana',
        'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh',
        'Maharashtra', 'Odisha', 'Punjab', 'Rajasthan', 'Tamil Nadu', 'Telangana',
        'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
      ]

  const availableYears = cropDetail?.available_years && cropDetail.available_years.length > 0
    ? cropDetail.available_years
    : [2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010]

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 rounded-xl border border-border/80 bg-card/60 backdrop-blur-xs mb-6 shadow-subtle">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2 text-xs font-semibold text-foreground mr-1">
          <Filter className="w-3.5 h-3.5 text-primary" />
          <span>Analytical Scope:</span>
        </div>

        {/* Dynamic Crop Selector */}
        <div className="flex items-center gap-1.5">
          <Sprout className="w-3.5 h-3.5 text-emerald-500" />
          <span className="text-xs text-muted-foreground font-medium">Crop:</span>
          <Select
            value={filters.crop || 'Rice'}
            onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setCrop(e.target.value)}
            className="h-8 text-xs font-medium w-44 bg-background/80"
            disabled={cropsLoading}
          >
            {availableCrops.map((c) => (
              <option key={c.crop} value={c.crop}>
                {c.crop} {c.forecasting_supported ? '★ (Model Active)' : '(Historical)'}
              </option>
            ))}
          </Select>
        </div>

        {/* Dynamic State Selector */}
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-muted-foreground">State:</span>
          <Select
            value={filters.state}
            onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setState(e.target.value)}
            className="h-8 text-xs w-40 bg-background/80"
          >
            <option value="all">All {availableStates.length} States</option>
            {availableStates.map((st) => (
              <option key={st} value={st}>{st}</option>
            ))}
          </Select>
        </div>

        {/* Dynamic Year Selector */}
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-muted-foreground">Year:</span>
          <Select
            value={filters.year}
            onChange={(e: React.ChangeEvent<HTMLSelectElement>) =>
              setYear(e.target.value === 'all' ? 'all' : Number(e.target.value))
            }
            className="h-8 text-xs w-32 bg-background/80"
          >
            <option value="all">All Years ({availableYears[0]}–{availableYears[availableYears.length - 1]})</option>
            {availableYears.map((yr) => (
              <option key={yr} value={yr}>{yr}</option>
            ))}
          </Select>
        </div>
      </div>

      {/* Dataset & Reset Action */}
      <div className="flex items-center gap-3">
        <div className="hidden lg:flex items-center gap-1.5 text-2xs text-muted-foreground bg-muted/40 px-2.5 py-1 rounded-md border border-border/50">
          <Database className="w-3 h-3 text-primary" />
          <span>Panel: <strong>AGRI_PANEL_1.0</strong> (71,601 records)</span>
        </div>

        <Button
          variant="ghost"
          size="sm"
          onClick={resetFilters}
          className="h-8 text-xs gap-1.5 text-muted-foreground hover:text-foreground"
        >
          <RotateCcw className="w-3 h-3" />
          Reset
        </Button>
      </div>
    </div>
  )
}
