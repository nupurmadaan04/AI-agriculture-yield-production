import React from 'react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { ModelRegistryItem } from '../../types/validation'

interface ModelRegistryTableProps {
  models: ModelRegistryItem[]
}

export const ModelRegistryTable: React.FC<ModelRegistryTableProps> = ({ models }) => {
  return (
    <Card className="border-border/80">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base font-bold text-foreground">Agricultural Model Governance Registry</CardTitle>
            <CardDescription className="text-xs">
              Version-controlled repository of active production pipelines, training temporal bounds, and evaluation artifacts
            </CardDescription>
          </div>
          <Badge variant="blue" className="text-xs">{models.length} Active Pipelines</Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-muted/50 border-b border-border/80 text-muted-foreground uppercase font-semibold">
              <tr>
                <th className="py-2.5 px-3">Pipeline Name</th>
                <th className="py-2.5 px-3">Version</th>
                <th className="py-2.5 px-3">Task & Target</th>
                <th className="py-2.5 px-3">Training Window</th>
                <th className="py-2.5 px-3">Evaluation Set</th>
                <th className="py-2.5 px-3">Artifact Path</th>
                <th className="py-2.5 px-3 text-right">Deployment Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60">
              {models.map((m, idx) => (
                <tr key={idx} className="hover:bg-muted/30">
                  <td className="py-2.5 px-3">
                    <div className="font-bold text-foreground">{m.model_name}</div>
                    <div className="text-[10px] text-muted-foreground">{m.model_type}</div>
                  </td>
                  <td className="py-2.5 px-3 font-mono font-semibold">{m.version}</td>
                  <td className="py-2.5 px-3">
                    <div className="text-foreground">{m.task}</div>
                    <div className="text-[10px] text-muted-foreground">Target: {m.target}</div>
                  </td>
                  <td className="py-2.5 px-3 text-muted-foreground">{m.training_period}</td>
                  <td className="py-2.5 px-3 text-muted-foreground">{m.evaluation_period}</td>
                  <td className="py-2.5 px-3 font-mono text-[11px] text-muted-foreground">{m.artifact_path}</td>
                  <td className="py-2.5 px-3 text-right">
                    <Badge variant={m.is_primary ? 'blue' : 'success'} className="text-[10px]">
                      {m.status}
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
