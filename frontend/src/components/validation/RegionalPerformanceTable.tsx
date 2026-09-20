import React, { useState } from 'react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { Input } from '../ui/Input'
import { Search } from 'lucide-react'
import { StatePerformanceItem } from '../../types/validation'

interface RegionalPerformanceTableProps {
  states: StatePerformanceItem[]
}

export const RegionalPerformanceTable: React.FC<RegionalPerformanceTableProps> = ({ states }) => {
  const [search, setSearch] = useState('')

  const filtered = states.filter((s) => s.state.toLowerCase().includes(search.toLowerCase()))

  return (
    <Card className="border-border/80">
      <CardHeader className="pb-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <CardTitle className="text-base font-bold text-foreground">State-Level Performance Slices</CardTitle>
            <CardDescription className="text-xs">
              Out-of-time evaluation sliced across all 20 rice-producing states
            </CardDescription>
          </div>
          <div className="relative w-full sm:w-56">
            <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-muted-foreground" />
            <Input
              type="text"
              placeholder="Filter by state..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-8 text-xs h-8"
            />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto max-h-80 overflow-y-auto">
          <table className="w-full text-left text-xs">
            <thead className="sticky top-0 bg-muted/90 backdrop-blur border-b border-border/80 text-muted-foreground uppercase font-semibold">
              <tr>
                <th className="py-2.5 px-3">State</th>
                <th className="py-2.5 px-3">Samples</th>
                <th className="py-2.5 px-3">Obs. Mean (kg/ha)</th>
                <th className="py-2.5 px-3">MAE (kg/ha)</th>
                <th className="py-2.5 px-3">RMSE (kg/ha)</th>
                <th className="py-2.5 px-3">R² Score</th>
                <th className="py-2.5 px-3 text-right">Bias Direction</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60">
              {filtered.map((s, idx) => (
                <tr key={idx} className="hover:bg-muted/30">
                  <td className="py-2.5 px-3 font-semibold text-foreground">{s.state}</td>
                  <td className="py-2.5 px-3 font-mono">{s.sample_count}</td>
                  <td className="py-2.5 px-3 font-mono">{s.mean_observed_yield.toFixed(1)}</td>
                  <td className="py-2.5 px-3 font-mono font-bold text-foreground">{s.mae.toFixed(1)}</td>
                  <td className="py-2.5 px-3 font-mono">{s.rmse.toFixed(1)}</td>
                  <td className="py-2.5 px-3 font-mono">{s.r2.toFixed(3)}</td>
                  <td className="py-2.5 px-3 text-right">
                    <Badge
                      variant={
                        s.bias_direction === 'BALANCED'
                          ? 'outline'
                          : s.bias_direction === 'UNDERPREDICTING'
                          ? 'warning'
                          : 'destructive'
                      }
                      className="text-[10px]"
                    >
                      {s.bias_direction}
                    </Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  )
}
