import { AgriculturalRecord, DatasetSummary, StatePerformanceSummary, YearlyTrendPoint, CropInfo } from '../types/agriculture'

export const datasetSummary: DatasetSummary = {
  totalRecords: 2484,
  cleanedRecords: 2469,
  statesCovered: 20,
  districtsCovered: 311,
  yearsCovered: '2010–2017',
  avgYield: 2062.81,
  medianYield: 2174.74,
  minYield: 0.0,
  maxYield: 4816.27,
  totalProduction: 812400.5,
  totalArea: 393820.2,
  zeroYieldCount: 236,
  missingValues: 0,
  duplicateRows: 0,
  columnsCount: 80,
}

export const availableCrops: CropInfo[] = [
  { id: 'rice', name: 'Rice (Kharif/Rabi)', season: 'Kharif / Rabi', coverage: 'Nationwide (20 States)', isPrimary: true, available: true },
  { id: 'wheat', name: 'Wheat', season: 'Rabi', coverage: 'Northern & Central India', isPrimary: false, available: false },
  { id: 'maize', name: 'Maize (Corn)', season: 'Kharif', coverage: 'All India', isPrimary: false, available: false },
  { id: 'sugarcane', name: 'Sugarcane', season: 'Annual', coverage: 'Tropical & Sub-tropical', isPrimary: false, available: false },
  { id: 'cotton', name: 'Cotton', season: 'Kharif', coverage: 'Western & Southern India', isPrimary: false, available: false },
]

export const yearlyTrends: YearlyTrendPoint[] = [
  { year: 2010, avgYield: 1807.81, totalProduction: 95980.2, totalArea: 53092.1, recordCount: 308 },
  { year: 2011, avgYield: 2179.89, totalProduction: 105301.4, totalArea: 48305.8, recordCount: 310 },
  { year: 2012, avgYield: 2244.67, totalProduction: 105240.8, totalArea: 46884.2, recordCount: 309 },
  { year: 2013, avgYield: 2024.73, totalProduction: 106650.0, totalArea: 52673.5, recordCount: 311 },
  { year: 2014, avgYield: 2056.03, totalProduction: 105480.6, totalArea: 51302.9, recordCount: 310 },
  { year: 2015, avgYield: 1704.23, totalProduction: 104410.1, totalArea: 61266.3, recordCount: 303 },
  { year: 2016, avgYield: 2320.74, totalProduction: 109698.3, totalArea: 47268.4, recordCount: 309 },
  { year: 2017, avgYield: 2183.67, totalProduction: 112750.2, totalArea: 51633.2, recordCount: 309 },
]

export const statePerformances: StatePerformanceSummary[] = [
  { stateCode: 16, stateName: 'Punjab', avgYield: 3982.4, totalProduction: 118200.5, totalArea: 29680.1, districtCount: 22, mae: 102.4, rank: 1 },
  { stateCode: 7, stateName: 'Haryana', avgYield: 3410.2, totalProduction: 41200.3, totalArea: 12081.2, districtCount: 21, mae: 115.1, rank: 2 },
  { stateCode: 1, stateName: 'Andhra Pradesh', avgYield: 3120.8, totalProduction: 76400.1, totalArea: 24480.0, districtCount: 13, mae: 98.6, rank: 3 },
  { stateCode: 18, stateName: 'Tamil Nadu', avgYield: 2980.5, totalProduction: 58200.7, totalArea: 19527.3, districtCount: 32, mae: 888.6, rank: 4 },
  { stateCode: 20, stateName: 'West Bengal', avgYield: 2840.1, totalProduction: 153000.2, totalArea: 53871.2, districtCount: 19, mae: 124.3, rank: 5 },
  { stateCode: 19, stateName: 'Uttar Pradesh', avgYield: 2450.6, totalProduction: 144500.0, totalArea: 58965.1, districtCount: 75, mae: 142.1, rank: 6 },
  { stateCode: 2, stateName: 'Assam', avgYield: 2040.3, totalProduction: 51200.4, totalArea: 25094.2, districtCount: 27, mae: 54.4, rank: 7 },
  { stateCode: 14, stateName: 'Orissa', avgYield: 1890.7, totalProduction: 68400.8, totalArea: 36177.0, districtCount: 30, mae: 39.6, rank: 8 },
  { stateCode: 3, stateName: 'Bihar', avgYield: 1845.2, totalProduction: 62100.9, totalArea: 33654.5, districtCount: 38, mae: 49.8, rank: 9 },
  { stateCode: 4, stateName: 'Chhattisgarh', avgYield: 1720.9, totalProduction: 65100.0, totalArea: 37829.1, districtCount: 27, mae: 55.2, rank: 10 },
  { stateCode: 10, stateName: 'Karnataka', avgYield: 2680.4, totalProduction: 38200.2, totalArea: 14251.3, districtCount: 30, mae: 435.8, rank: 11 },
  { stateCode: 11, stateName: 'Kerala', avgYield: 2750.1, totalProduction: 5400.6, totalArea: 1963.8, districtCount: 14, mae: 514.6, rank: 12 },
  { stateCode: 12, stateName: 'Madhya Pradesh', avgYield: 1620.5, totalProduction: 34500.1, totalArea: 21289.4, districtCount: 51, mae: 360.7, rank: 13 },
  { stateCode: 17, stateName: 'Rajasthan', avgYield: 1540.2, totalProduction: 3100.4, totalArea: 2013.0, districtCount: 33, mae: 446.4, rank: 14 },
  { stateCode: 8, stateName: 'Himachal Pradesh', avgYield: 1680.1, totalProduction: 1250.2, totalArea: 744.1, districtCount: 12, mae: 706.4, rank: 15 },
  { stateCode: 9, stateName: 'Jharkhand', avgYield: 1920.0, totalProduction: 31200.0, totalArea: 16250.0, districtCount: 24, mae: 70.4, rank: 16 },
  { stateCode: 13, stateName: 'Maharashtra', avgYield: 1810.5, totalProduction: 28400.0, totalArea: 15686.0, districtCount: 36, mae: 189.2, rank: 17 },
  { stateCode: 6, stateName: 'Gujarat', avgYield: 2110.3, totalProduction: 16800.0, totalArea: 7961.0, districtCount: 33, mae: 165.4, rank: 18 },
  { stateCode: 15, stateName: 'Uttarakhand', avgYield: 2280.9, totalProduction: 5800.0, totalArea: 2542.8, districtCount: 13, mae: 140.2, rank: 19 },
  { stateCode: 5, stateName: 'Telangana', avgYield: 3250.0, totalProduction: 62000.0, totalArea: 19076.9, districtCount: 31, mae: 112.5, rank: 20 },
]

export const sampleRecords: AgriculturalRecord[] = [
  { id: 'rec-1', year: 2017, stateCode: 16, stateName: 'Punjab', distCode: 101, distName: 'Ludhiana', cropName: 'Rice', area: 258.4, production: 1072.36, yield: 4150.0 },
  { id: 'rec-2', year: 2017, stateCode: 16, stateName: 'Punjab', distCode: 102, distName: 'Amritsar', cropName: 'Rice', area: 182.1, production: 746.61, yield: 4100.0 },
  { id: 'rec-3', year: 2017, stateCode: 7, stateName: 'Haryana', distCode: 201, distName: 'Karnal', cropName: 'Rice', area: 168.5, production: 623.45, yield: 3700.0 },
  { id: 'rec-4', year: 2017, stateCode: 1, stateName: 'Andhra Pradesh', distCode: 301, distName: 'East Godavari', cropName: 'Rice', area: 412.0, production: 1359.60, yield: 3300.0 },
  { id: 'rec-5', year: 2017, stateCode: 20, stateName: 'West Bengal', distCode: 401, distName: 'Bardhaman', cropName: 'Rice', area: 540.2, production: 1674.62, yield: 3100.0 },
  { id: 'rec-6', year: 2016, stateCode: 19, stateName: 'Uttar Pradesh', distCode: 501, distName: 'Varanasi', cropName: 'Rice', area: 135.0, production: 337.50, yield: 2500.0 },
  { id: 'rec-7', year: 2016, stateCode: 14, stateName: 'Orissa', distCode: 601, distName: 'Cuttack', cropName: 'Rice', area: 180.4, production: 360.80, yield: 2000.0 },
  { id: 'rec-8', year: 2016, stateCode: 2, stateName: 'Assam', distCode: 701, distName: 'Kamrup', cropName: 'Rice', area: 110.2, production: 231.42, yield: 2100.0 },
  { id: 'rec-9', year: 2015, stateCode: 3, stateName: 'Bihar', distCode: 801, distName: 'Patna', cropName: 'Rice', area: 125.6, production: 238.64, yield: 1900.0 },
  { id: 'rec-10', year: 2015, stateCode: 4, stateName: 'Chhattisgarh', distCode: 901, distName: 'Raipur', cropName: 'Rice', area: 245.0, production: 416.50, yield: 1700.0 },
  { id: 'rec-11', year: 2015, stateCode: 10, stateName: 'Karnataka', distCode: 1001, distName: 'Mandya', cropName: 'Rice', area: 95.0, production: 285.00, yield: 3000.0 },
  { id: 'rec-12', year: 2014, stateCode: 18, stateName: 'Tamil Nadu', distCode: 1101, distName: 'Thanjavur', cropName: 'Rice', area: 175.2, production: 578.16, yield: 3300.0 },
  { id: 'rec-13', year: 2014, stateCode: 12, stateName: 'Madhya Pradesh', distCode: 1201, distName: 'Balaghat', cropName: 'Rice', area: 260.0, production: 442.00, yield: 1700.0 },
  { id: 'rec-14', year: 2013, stateCode: 17, stateName: 'Rajasthan', distCode: 1301, distName: 'Kota', cropName: 'Rice', area: 42.0, production: 126.00, yield: 3000.0 },
  { id: 'rec-15', year: 2013, stateCode: 8, stateName: 'Himachal Pradesh', distCode: 1401, distName: 'Kangra', cropName: 'Rice', area: 38.0, production: 64.60, yield: 1700.0 },
  { id: 'rec-16', year: 2012, stateCode: 11, stateName: 'Kerala', distCode: 1501, distName: 'Palakkad', cropName: 'Rice', area: 82.5, production: 231.00, yield: 2800.0 },
  { id: 'rec-17', year: 2011, stateCode: 9, stateName: 'Jharkhand', distCode: 1601, distName: 'Ranchi', cropName: 'Rice', area: 98.0, production: 186.20, yield: 1900.0 },
  { id: 'rec-18', year: 2010, stateCode: 13, stateName: 'Maharashtra', distCode: 1701, distName: 'Bhandara', cropName: 'Rice', area: 175.0, production: 315.00, yield: 1800.0 },
  { id: 'rec-19', year: 2010, stateCode: 6, stateName: 'Gujarat', distCode: 1801, distName: 'Kheda', cropName: 'Rice', area: 112.0, production: 246.40, yield: 2200.0 },
  { id: 'rec-20', year: 2010, stateCode: 15, stateName: 'Uttarakhand', distCode: 1901, distName: 'Udham Singh Nagar', cropName: 'Rice', area: 104.0, production: 312.00, yield: 3000.0 }
]
