/**
 * Day 32 Decision Workspace API Client & Hooks
 */

import { useQuery } from '@tanstack/react-query'
import type {
  WorkspaceAnalyzeRequest,
  DecisionWorkspaceResponse,
  WorkspaceTemplatesResponse,
  ScenarioItem
} from '../types/workspace'

const API_BASE = import.meta.env.VITE_API_BASE_URL
  ? `${import.meta.env.VITE_API_BASE_URL}/workspace`
  : '/api/workspace'

export const workspaceService = {
  async analyzeWorkspace(req: WorkspaceAnalyzeRequest): Promise<DecisionWorkspaceResponse> {
    const res = await fetch(`${API_BASE}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req)
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.detail || `Workspace analysis failed (${res.status})`)
    }
    return res.json()
  },

  async getTemplates(): Promise<WorkspaceTemplatesResponse> {
    const res = await fetch(`${API_BASE}/templates`)
    if (!res.ok) {
      throw new Error(`Failed to load scenario templates (${res.status})`)
    }
    return res.json()
  },

  async simulateScenario(params: {
    crop: string
    state: string
    district?: string
    forecast_year: number
    scenario_type: string
    modifications?: Record<string, number>
  }): Promise<ScenarioItem> {
    const res = await fetch(`${API_BASE}/scenarios/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.detail || 'Scenario simulation failed')
    }
    return res.json()
  }
}

export function useWorkspaceQuery(req: WorkspaceAnalyzeRequest, enabled = true) {
  return useQuery({
    queryKey: ['decision-workspace', req.crop, req.state, req.district, req.forecast_year, req.custom_modifications],
    queryFn: () => workspaceService.analyzeWorkspace(req),
    enabled,
    staleTime: 1000 * 60 * 5,
  })
}

export function useWorkspaceTemplatesQuery() {
  return useQuery({
    queryKey: ['workspace-templates'],
    queryFn: () => workspaceService.getTemplates(),
    staleTime: 1000 * 60 * 30,
  })
}
