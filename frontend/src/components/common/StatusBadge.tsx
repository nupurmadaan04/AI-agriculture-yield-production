import React from 'react'
import { Badge } from '../ui/Badge'

interface StatusBadgeProps {
  status: 'production' | 'benchmark' | 'baseline' | 'experimental' | 'coming-soon' | 'requires-backend'
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status }) => {
  switch (status) {
    case 'production':
      return <Badge variant="success">Validated Model</Badge>
    case 'benchmark':
      return <Badge variant="secondary">Evaluated Benchmark</Badge>
    case 'baseline':
      return <Badge variant="blue">Exact Statistical Baseline</Badge>
    case 'experimental':
      return <Badge variant="warning">Experimental</Badge>
    case 'coming-soon':
      return <Badge variant="outline">Coming Soon</Badge>
    case 'requires-backend':
      return <Badge variant="outline">Requires Backend API</Badge>
    default:
      return <Badge variant="secondary">{status}</Badge>
  }
}
