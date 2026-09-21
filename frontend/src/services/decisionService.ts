/**
 * API client and React Query hooks for Day 14 Decision Intelligence.
 */

import { useQuery, useMutation } from '@tanstack/react-query'
import type {
  DecisionAnalyzeRequest,
  DecisionAnalyzeResponse,
  DecisionBrief,
  DecisionOptionsResponse,
  DecisionRobustnessResponse,
  DecisionHistoryResponse,
  DecisionAudit,
  DecisionProvenance
} from '../types/decision'

const API_BASE = import.meta.env.VITE_API_BASE_URL ? `${import.meta.env.VITE_API_BASE_URL}/decision` : '/api/decision'

async function handleResponse<T>(res: Response, fallbackError: string): Promise<T> {
  if (!res.ok) {
    let msg = `${fallbackError}: HTTP ${res.status}`
    try {
      const err = await res.json()
      if (err.error?.message) msg = err.error.message
      else if (err.detail) msg = typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail)
    } catch {
      // ignore
    }
    throw new Error(msg)
  }
  return res.json()
}

export const decisionService = {
  analyzeDecision: async (params: DecisionAnalyzeRequest): Promise<DecisionAnalyzeResponse> => {
    const res = await fetch(`${API_BASE}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    })
    return handleResponse<DecisionAnalyzeResponse>(res, 'Decision analysis failed')
  },

  generateBrief: async (params: DecisionAnalyzeRequest): Promise<DecisionBrief> => {
    const res = await fetch(`${API_BASE}/brief`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    })
    return handleResponse<DecisionBrief>(res, 'Failed to generate decision brief')
  },

  getOptions: async (params: DecisionAnalyzeRequest): Promise<DecisionOptionsResponse> => {
    const res = await fetch(`${API_BASE}/options`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    })
    return handleResponse<DecisionOptionsResponse>(res, 'Failed to fetch decision options')
  },

  getRobustness: async (params: DecisionAnalyzeRequest): Promise<DecisionRobustnessResponse> => {
    const res = await fetch(`${API_BASE}/robustness`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    })
    return handleResponse<DecisionRobustnessResponse>(res, 'Failed to evaluate robustness')
  },

  getDecisionById: async (decisionId: string): Promise<DecisionAnalyzeResponse> => {
    const res = await fetch(`${API_BASE}/${decisionId}`)
    return handleResponse<DecisionAnalyzeResponse>(res, `Failed to fetch decision ${decisionId}`)
  },

  getAudit: async (decisionId: string): Promise<DecisionAudit> => {
    const res = await fetch(`${API_BASE}/${decisionId}/audit`)
    return handleResponse<DecisionAudit>(res, 'Failed to fetch audit certificate')
  },

  getProvenance: async (decisionId: string): Promise<DecisionProvenance> => {
    const res = await fetch(`${API_BASE}/${decisionId}/provenance`)
    return handleResponse<DecisionProvenance>(res, 'Failed to fetch provenance')
  },

  getHistory: async (limit: number = 50): Promise<DecisionHistoryResponse> => {
    const res = await fetch(`${API_BASE}/history?limit=${limit}`)
    return handleResponse<DecisionHistoryResponse>(res, 'Failed to fetch decision history')
  },

  getMethodology: async (): Promise<any> => {
    const res = await fetch(`${API_BASE}/methodology`)
    return handleResponse<any>(res, 'Failed to fetch methodology')
  },

  getBriefQuery: async (params: { crop: string; state: string; district?: string; year?: number; horizon?: string }): Promise<DecisionBrief> => {
    const q = new URLSearchParams({
      crop: params.crop,
      state: params.state,
      year: String(params.year || 2017),
      decision_horizon: params.horizon || 'next_season'
    })
    if (params.district) q.set('district', params.district)
    const res = await fetch(`${API_BASE}/brief?${q.toString()}`)
    return handleResponse<DecisionBrief>(res, 'Failed to fetch decision brief')
  },

  getCropEvidence: async (crop: string, state: string = 'Punjab', district?: string, year: number = 2017): Promise<any> => {
    const q = new URLSearchParams({ state, year: String(year) })
    if (district) q.set('district', district)
    const res = await fetch(`${API_BASE}/evidence/${encodeURIComponent(crop)}?${q.toString()}`)
    return handleResponse<any>(res, `Failed to fetch evidence for ${crop}`)
  }
}

export const useDecisionBriefQuery = (params: { crop: string; state: string; district?: string; year?: number; horizon?: string }) => {
  return useQuery({
    queryKey: ['decision', 'brief', params.crop, params.state, params.district, params.year, params.horizon],
    queryFn: () => decisionService.getBriefQuery(params),
    staleTime: 60 * 1000
  })
}


// React Query Hooks
export const useDecisionAnalysis = () => {
  return useMutation({
    mutationFn: (params: DecisionAnalyzeRequest) => decisionService.analyzeDecision(params)
  })
}

export const useDecisionHistory = (limit: number = 50) => {
  return useQuery({
    queryKey: ['decision', 'history', limit],
    queryFn: () => decisionService.getHistory(limit),
    staleTime: 60 * 1000
  })
}

export const useDecisionDetail = (decisionId?: string) => {
  return useQuery({
    queryKey: ['decision', 'detail', decisionId],
    queryFn: () => decisionService.getDecisionById(decisionId!),
    enabled: !!decisionId,
    staleTime: 5 * 60 * 1000
  })
}

export const useDecisionMethodology = () => {
  return useQuery({
    queryKey: ['decision', 'methodology'],
    queryFn: () => decisionService.getMethodology(),
    staleTime: 60 * 60 * 1000
  })
}
