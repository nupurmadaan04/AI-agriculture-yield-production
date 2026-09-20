export interface ReportItem {
  id: string
  title: string
  category: 'scientific' | 'benchmark' | 'dataset' | 'trends'
  summary: string
  generatedDate: string
  fileFormat: 'PDF' | 'CSV' | 'MD'
  status: 'available' | 'coming-soon'
  downloadUrl?: string
  fileSize?: string
}

export type ExportFormat = 'csv' | 'json' | 'pdf'
