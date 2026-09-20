import React from 'react'
import { formatNumber } from '../../lib/utils'

interface MetricItem {
  value: string
  label: string
  descriptor: string
}

const verifiedMetrics: MetricItem[] = [
  {
    value: '71,601',
    label: 'Unified Agricultural Records',
    descriptor: 'Unified agricultural panel records (1966–2017)',
  },
  {
    value: '29',
    label: 'Verified Crops',
    descriptor: 'Verified crops in canonical schema',
  },
  {
    value: '20',
    label: 'Indian States',
    descriptor: 'Indian states represented',
  },
  {
    value: '311',
    label: 'Districts',
    descriptor: 'Districts represented across pan-India',
  },
  {
    value: '14',
    label: 'Model-Ready Crops',
    descriptor: 'Crops eligible for forecasting evaluation',
  },
]

export const PlatformMetrics: React.FC = () => {
  return (
    <section className="py-10 border-b border-border/80 bg-muted/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-6 sm:gap-8">
          {verifiedMetrics.map((m, idx) => (
            <div
              key={idx}
              className={`space-y-1 ${
                idx === verifiedMetrics.length - 1 ? 'col-span-2 md:col-span-1 text-center md:text-left' : 'text-left'
              }`}
            >
              <div className="text-2xl sm:text-3xl font-extrabold text-foreground font-mono tracking-tight">
                {m.value}
              </div>
              <div className="text-xs font-bold text-foreground">
                {m.label}
              </div>
              <p className="text-[11px] text-muted-foreground leading-snug">
                {m.descriptor}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
