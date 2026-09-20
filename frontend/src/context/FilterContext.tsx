import React, { createContext, useContext, useState } from 'react'
import { FilterState } from '../types/agriculture'

interface FilterContextType {
  filters: FilterState
  setFilters: React.Dispatch<React.SetStateAction<FilterState>>
  resetFilters: () => void
  setCrop: (crop: string) => void
  setState: (state: string) => void
  setDistrict: (district: string) => void
  setYear: (year: number | 'all') => void
}

const defaultFilters: FilterState = {
  crop: 'rice',
  state: 'all',
  district: 'all',
  year: 'all'
}

const FilterContext = createContext<FilterContextType | undefined>(undefined)

export const FilterProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [filters, setFilters] = useState<FilterState>(defaultFilters)

  const resetFilters = () => setFilters(defaultFilters)
  const setCrop = (crop: string) => setFilters((prev: FilterState) => ({ ...prev, crop }))
  const setState = (state: string) => setFilters((prev: FilterState) => ({ ...prev, state, district: 'all' }))
  const setDistrict = (district: string) => setFilters((prev: FilterState) => ({ ...prev, district }))
  const setYear = (year: number | 'all') => setFilters((prev: FilterState) => ({ ...prev, year }))

  return (
    <FilterContext.Provider value={{ filters, setFilters, resetFilters, setCrop, setState, setDistrict, setYear }}>
      {children}
    </FilterContext.Provider>
  )
}

export const useFilters = () => {
  const context = useContext(FilterContext)
  if (!context) throw new Error('useFilters must be used within a FilterProvider')
  return context
}

export const useFilterContext = useFilters
