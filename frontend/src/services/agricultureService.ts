import { datasetSummary, yearlyTrends, statePerformances, sampleRecords, availableCrops } from '../data/mockAgricultureData'
import { AgriculturalRecord, DatasetSummary, StatePerformanceSummary, YearlyTrendPoint, CropInfo, FilterState } from '../types/agriculture'

export const agricultureService = {
  getSummary: async (): Promise<DatasetSummary> => {
    return Promise.resolve(datasetSummary)
  },

  getTrends: async (): Promise<YearlyTrendPoint[]> => {
    return Promise.resolve(yearlyTrends)
  },

  getStatePerformances: async (): Promise<StatePerformanceSummary[]> => {
    return Promise.resolve(statePerformances)
  },

  getCrops: async (): Promise<CropInfo[]> => {
    return Promise.resolve(availableCrops)
  },

  getRecords: async (filters?: Partial<FilterState>): Promise<AgriculturalRecord[]> => {
    let records = [...sampleRecords]
    if (filters?.year && filters.year !== 'all') {
      records = records.filter(r => r.year === Number(filters.year))
    }
    if (filters?.state && filters.state !== 'all') {
      records = records.filter(r => (r.state || r.stateName || '').toLowerCase() === filters.state?.toLowerCase())
    }
    return Promise.resolve(records)
  }
}
