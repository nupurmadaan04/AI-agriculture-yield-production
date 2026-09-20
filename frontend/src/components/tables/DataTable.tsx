import React, { useState } from 'react'
import { AgriculturalRecord } from '../../types/agriculture'
import { formatNumber, formatYield, formatProduction, formatArea } from '../../lib/utils'
import { ExportButton } from '../common/FeedbackStates'
import { Search, ChevronLeft, ChevronRight, ArrowUpDown } from 'lucide-react'
import { Input } from '../ui/Input'
import { Button } from '../ui/Button'
import { Badge } from '../ui/Badge'

interface DataTableProps {
  records: AgriculturalRecord[]
  title?: string
}

export const DataTable: React.FC<DataTableProps> = ({ records, title = "Agricultural Dataset Explorer" }) => {
  const [searchTerm, setSearchTerm] = useState('')
  const [sortField, setSortField] = useState<keyof AgriculturalRecord>('year')
  const [sortAsc, setSortAsc] = useState(false)
  const [page, setPage] = useState(1)
  const pageSize = 10

  const filtered = records.filter(r => {
    const sName = (r.state || r.stateName || '').toLowerCase()
    const dName = (r.district || r.distName || '').toLowerCase()
    const term = searchTerm.toLowerCase()
    return sName.includes(term) || dName.includes(term) || r.year.toString().includes(term)
  })

  const sorted = [...filtered].sort((a, b) => {
    const valA = a[sortField] ?? ''
    const valB = b[sortField] ?? ''
    if (valA < valB) return sortAsc ? -1 : 1
    if (valA > valB) return sortAsc ? 1 : -1
    return 0
  })

  const totalPages = Math.ceil(sorted.length / pageSize) || 1
  const paginated = sorted.slice((page - 1) * pageSize, page * pageSize)

  const handleSort = (field: keyof AgriculturalRecord) => {
    if (sortField === field) {
      setSortAsc(!sortAsc)
    } else {
      setSortField(field)
      setSortAsc(true)
    }
  }

  return (
    <div className="rounded-xl border border-border bg-card shadow-subtle overflow-hidden">
      {/* Table Header Controls */}
      <div className="p-4 border-b border-border flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <h3 className="font-semibold text-sm text-foreground">{title}</h3>
          <Badge variant="secondary">{records.length} records</Badge>
        </div>
        <div className="flex items-center gap-2">
          <div className="relative w-48 sm:w-64">
            <Search className="w-3.5 h-3.5 absolute left-3 top-3 text-muted-foreground" />
            <Input
              placeholder="Search district, state..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value)
                setPage(1)
              }}
              className="pl-8 h-9 text-xs"
            />
          </div>
          <ExportButton data={records} filename="icrisat-rice-dataset.csv" />
        </div>
      </div>

      {/* Table Content */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="bg-muted/40 text-muted-foreground uppercase text-[10px] tracking-wider border-b border-border">
            <tr>
              <th className="p-3 font-semibold cursor-pointer hover:text-foreground" onClick={() => handleSort('year')}>
                <div className="flex items-center gap-1">Year <ArrowUpDown className="w-3 h-3" /></div>
              </th>
              <th className="p-3 font-semibold cursor-pointer hover:text-foreground" onClick={() => handleSort('state')}>
                <div className="flex items-center gap-1">State <ArrowUpDown className="w-3 h-3" /></div>
              </th>
              <th className="p-3 font-semibold cursor-pointer hover:text-foreground" onClick={() => handleSort('district')}>
                <div className="flex items-center gap-1">District <ArrowUpDown className="w-3 h-3" /></div>
              </th>
              <th className="p-3 font-semibold cursor-pointer hover:text-foreground text-right" onClick={() => handleSort('area')}>
                <div className="flex items-center justify-end gap-1">Area ('000 ha) <ArrowUpDown className="w-3 h-3" /></div>
              </th>
              <th className="p-3 font-semibold cursor-pointer hover:text-foreground text-right" onClick={() => handleSort('production')}>
                <div className="flex items-center justify-end gap-1">Production ('000 t) <ArrowUpDown className="w-3 h-3" /></div>
              </th>
              <th className="p-3 font-semibold cursor-pointer hover:text-foreground text-right" onClick={() => handleSort('yield')}>
                <div className="flex items-center justify-end gap-1">Yield (kg/ha) <ArrowUpDown className="w-3 h-3" /></div>
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/60">
            {paginated.length === 0 ? (
              <tr>
                <td colSpan={6} className="p-8 text-center text-muted-foreground text-xs">
                  No records match the current filter.
                </td>
              </tr>
            ) : (
              paginated.map((r) => (
                <tr key={r.id} className="hover:bg-muted/30 transition-colors">
                  <td className="p-3 font-medium text-foreground">{r.year}</td>
                  <td className="p-3 text-foreground">{r.state || r.stateName}</td>
                  <td className="p-3 text-foreground font-medium">{r.district || r.distName}</td>
                  <td className="p-3 text-right text-muted-foreground font-mono">{formatArea(r.area)}</td>
                  <td className="p-3 text-right text-muted-foreground font-mono">{formatProduction(r.production)}</td>
                  <td className="p-3 text-right font-semibold text-foreground font-mono">
                    {r.yield === 0 ? (
                      <span className="text-amber-500">0.0 kg/ha</span>
                    ) : (
                      formatYield(r.yield)
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Table Pagination */}
      <div className="p-3 border-t border-border flex items-center justify-between text-xs text-muted-foreground">
        <span>
          Showing {Math.min(sorted.length, (page - 1) * pageSize + 1)} to {Math.min(sorted.length, page * pageSize)} of {sorted.length} records
        </span>
        <div className="flex items-center gap-1.5">
          <Button
            variant="outline"
            size="icon"
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
            className="h-8 w-8"
          >
            <ChevronLeft className="w-4 h-4" />
          </Button>
          <span className="text-xs px-2 font-medium text-foreground">
            Page {page} of {totalPages}
          </span>
          <Button
            variant="outline"
            size="icon"
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            className="h-8 w-8"
          >
            <ChevronRight className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  )
}
