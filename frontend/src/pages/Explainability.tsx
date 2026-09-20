import React, { useState } from 'react'
import { Sparkles, BarChart3, Sliders, ShieldAlert, GitCompare, Award, BookOpen, ShieldCheck, Info } from 'lucide-react'
import { ReliabilityContext } from '../components/explainability/ReliabilityContext'
import { GlobalFeatureImportance } from '../components/explainability/GlobalFeatureImportance'
import { PredictionExplanation } from '../components/explainability/PredictionExplanation'
import { SensitivityAnalysis } from '../components/explainability/SensitivityAnalysis'
import { AlertExplanation } from '../components/explainability/AlertExplanation'
import { ScenarioExplanation } from '../components/explainability/ScenarioExplanation'
import { ExplanationAuditPanel } from '../components/explainability/ExplanationAuditPanel'
import { ExplanationMethodology } from '../components/explainability/ExplanationMethodology'
import { useGlobalImportance, useStates } from '../services/api'

export const Explainability: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'global' | 'prediction' | 'sensitivity' | 'alert' | 'scenario' | 'audit' | 'methodology'>('global')

  const { data: globalImportance, isLoading: isGlobalLoading } = useGlobalImportance()
  const { data: statesData } = useStates()
  const stateNames = statesData?.data?.map((s) => s.state) || ['Punjab', 'Haryana', 'Andhra Pradesh', 'Uttar Pradesh', 'West Bengal', 'Tamil Nadu']

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 lg:p-8 space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800/80 pb-6">
        <div>
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-emerald-500/10 text-emerald-400 rounded-xl border border-emerald-500/20">
              <Sparkles className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight">
                Explainable Agricultural AI & Decision Traceability
              </h1>
              <p className="text-xs text-slate-400 mt-1">
                Understand how registered models generate agricultural predictions, early warnings, and scenario responses.
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1.5">
            <ShieldCheck className="w-4 h-4" /> Non-Causal XAI Layer
          </span>
        </div>
      </div>

      {/* Mandatory Scientific Integrity Banner */}
      <div className="p-4 bg-slate-900 border-l-4 border-l-emerald-500 border-y border-r border-slate-800 rounded-r-xl flex items-start gap-3 shadow-sm">
        <Info className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
        <div className="text-xs text-slate-300 leading-relaxed">
          <strong className="text-emerald-300 font-semibold">Scientific Integrity Directive:</strong> Explanations describe the machine
          learning model's internal feature attribution and mathematical loss sensitivity curves within the historical ICRISAT panel.
          They <strong className="text-white">do not assert physical, agronomic, or causal intervention efficacy</strong> in real-world agricultural environments.
        </div>
      </div>

      {/* Model Reliability & Version Metadata Context */}
      <ReliabilityContext />

      {/* Tab Navigation */}
      <div className="flex overflow-x-auto gap-2 border-b border-slate-800 pb-2">
        <button
          onClick={() => setActiveTab('global')}
          className={`px-4 py-2 text-xs font-semibold rounded-lg flex items-center gap-2 transition-colors whitespace-nowrap ${
            activeTab === 'global'
              ? 'bg-emerald-500 text-slate-950 shadow-sm'
              : 'bg-slate-900 text-slate-400 hover:text-white border border-slate-800'
          }`}
        >
          <BarChart3 className="w-4 h-4" /> Global Feature Importance
        </button>

        <button
          onClick={() => setActiveTab('prediction')}
          className={`px-4 py-2 text-xs font-semibold rounded-lg flex items-center gap-2 transition-colors whitespace-nowrap ${
            activeTab === 'prediction'
              ? 'bg-emerald-500 text-slate-950 shadow-sm'
              : 'bg-slate-900 text-slate-400 hover:text-white border border-slate-800'
          }`}
        >
          <Sparkles className="w-4 h-4" /> Explain Prediction
        </button>

        <button
          onClick={() => setActiveTab('sensitivity')}
          className={`px-4 py-2 text-xs font-semibold rounded-lg flex items-center gap-2 transition-colors whitespace-nowrap ${
            activeTab === 'sensitivity'
              ? 'bg-emerald-500 text-slate-950 shadow-sm'
              : 'bg-slate-900 text-slate-400 hover:text-white border border-slate-800'
          }`}
        >
          <Sliders className="w-4 h-4" /> Model Sensitivity
        </button>

        <button
          onClick={() => setActiveTab('alert')}
          className={`px-4 py-2 text-xs font-semibold rounded-lg flex items-center gap-2 transition-colors whitespace-nowrap ${
            activeTab === 'alert'
              ? 'bg-emerald-500 text-slate-950 shadow-sm'
              : 'bg-slate-900 text-slate-400 hover:text-white border border-slate-800'
          }`}
        >
          <ShieldAlert className="w-4 h-4" /> Alert Provenance
        </button>

        <button
          onClick={() => setActiveTab('scenario')}
          className={`px-4 py-2 text-xs font-semibold rounded-lg flex items-center gap-2 transition-colors whitespace-nowrap ${
            activeTab === 'scenario'
              ? 'bg-emerald-500 text-slate-950 shadow-sm'
              : 'bg-slate-900 text-slate-400 hover:text-white border border-slate-800'
          }`}
        >
          <GitCompare className="w-4 h-4" /> Scenario Attribution
        </button>

        <button
          onClick={() => setActiveTab('audit')}
          className={`px-4 py-2 text-xs font-semibold rounded-lg flex items-center gap-2 transition-colors whitespace-nowrap ${
            activeTab === 'audit'
              ? 'bg-emerald-500 text-slate-950 shadow-sm'
              : 'bg-slate-900 text-slate-400 hover:text-white border border-slate-800'
          }`}
        >
          <Award className="w-4 h-4" /> Audit Certificates
        </button>

        <button
          onClick={() => setActiveTab('methodology')}
          className={`px-4 py-2 text-xs font-semibold rounded-lg flex items-center gap-2 transition-colors whitespace-nowrap ${
            activeTab === 'methodology'
              ? 'bg-emerald-500 text-slate-950 shadow-sm'
              : 'bg-slate-900 text-slate-400 hover:text-white border border-slate-800'
          }`}
        >
          <BookOpen className="w-4 h-4" /> Methodology
        </button>
      </div>

      {/* Tab Panels */}
      <div>
        {activeTab === 'global' && (
          <GlobalFeatureImportance data={globalImportance} isLoading={isGlobalLoading} />
        )}

        {activeTab === 'prediction' && <PredictionExplanation states={stateNames} />}

        {activeTab === 'sensitivity' && <SensitivityAnalysis states={stateNames} />}

        {activeTab === 'alert' && <AlertExplanation />}

        {activeTab === 'scenario' && <ScenarioExplanation states={stateNames} />}

        {activeTab === 'audit' && <ExplanationAuditPanel />}

        {activeTab === 'methodology' && <ExplanationMethodology />}
      </div>
    </div>
  )
}
export default Explainability
