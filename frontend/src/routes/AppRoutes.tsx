import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { WebShell } from '../components/layout/WebShell'
import { Home } from '../pages/Home'
import { YieldCalculator } from '../pages/YieldCalculator'
import { AgriculturalIntelligence } from '../pages/AgriculturalIntelligence'
import { ScenarioIntelligence } from '../pages/ScenarioIntelligence'
import { AgriculturalCopilot } from '../pages/AgriculturalCopilot'
import { DecisionSupport } from '../pages/DecisionSupport'
import { EarlyWarning } from '../pages/EarlyWarning'
import { GeospatialIntelligence } from '../pages/GeospatialIntelligence'
import { ModelReliability } from '../pages/ModelReliability'
import { Solutions } from '../pages/Solutions'
import { NationalPortal } from '../pages/NationalPortal'
import { ScienceWhitepaper } from '../pages/ScienceWhitepaper'
import { AgriculturalMonitoring } from '../pages/AgriculturalMonitoring'
import { Explainability } from '../pages/Explainability'
import { DecisionIntelligence } from '../pages/DecisionIntelligence'
import { ModelingReadiness } from '../pages/ModelingReadiness'
import { ForecastIntelligence } from '../pages/ForecastIntelligence'
import { ObservabilityCenter } from '../pages/ObservabilityCenter'
import { PredictionExplorer } from '../pages/PredictionExplorer'
import { Pricing } from '../pages/Pricing'
import { Contact } from '../pages/Contact'
import { NotFound } from '../pages/NotFound'

export const AppRoutes: React.FC = () => {
  return (
    <Routes>
      <Route element={<WebShell />}>
        <Route path="/" element={<Home />} />
        <Route path="/prediction-explorer" element={<PredictionExplorer />} />
        <Route path="/forecast" element={<ForecastIntelligence />} />
        <Route path="/forecast-intelligence" element={<ForecastIntelligence />} />
        <Route path="/observability" element={<ObservabilityCenter />} />
        <Route path="/decision-intelligence" element={<DecisionIntelligence />} />
        <Route path="/modeling-readiness" element={<ModelingReadiness />} />
        <Route path="/monitoring" element={<AgriculturalMonitoring />} />
        <Route path="/explainability" element={<Explainability />} />
        <Route path="/calculator" element={<YieldCalculator />} />
        <Route path="/intelligence" element={<AgriculturalIntelligence />} />
        <Route path="/scenario" element={<ScenarioIntelligence />} />
        <Route path="/copilot" element={<AgriculturalCopilot />} />
        <Route path="/early-warning" element={<EarlyWarning />} />
        <Route path="/geospatial" element={<GeospatialIntelligence />} />
        <Route path="/model-reliability" element={<ModelReliability />} />
        <Route path="/decision-support" element={<DecisionSupport />} />
        <Route path="/solutions" element={<Solutions />} />
        <Route path="/portal" element={<NationalPortal />} />
        <Route path="/science" element={<ScienceWhitepaper />} />
        <Route path="/pricing" element={<Pricing />} />
        <Route path="/contact" element={<Contact />} />
        
        {/* Convenience Redirects */}
        <Route path="/explorer" element={<Navigate to="/prediction-explorer" replace />} />
        <Route path="/explain" element={<Navigate to="/prediction-explorer" replace />} />
        <Route path="/ops" element={<Navigate to="/observability" replace />} />
        <Route path="/telemetry" element={<Navigate to="/observability" replace />} />
        <Route path="/readiness" element={<Navigate to="/modeling-readiness" replace />} />
        <Route path="/multi-crop" element={<Navigate to="/modeling-readiness" replace />} />
        <Route path="/decision" element={<Navigate to="/decision-intelligence" replace />} />
        <Route path="/xai" element={<Navigate to="/explainability" replace />} />
        <Route path="/forecasting" element={<Navigate to="/forecast" replace />} />
        <Route path="/trends" element={<Navigate to="/early-warning" replace />} />
        <Route path="/what-if" element={<Navigate to="/scenario" replace />} />
        <Route path="/ai-copilot" element={<Navigate to="/copilot" replace />} />
        <Route path="/chat" element={<Navigate to="/copilot" replace />} />
        <Route path="/executive" element={<Navigate to="/decision-support" replace />} />
        <Route path="/scenario-lab" element={<Navigate to="/scenario" replace />} />
        <Route path="/science-whitepaper" element={<Navigate to="/science" replace />} />
        <Route path="/yield-calculator" element={<Navigate to="/calculator" replace />} />
        <Route path="/risk" element={<Navigate to="/intelligence" replace />} />
        <Route path="/anomalies" element={<Navigate to="/intelligence" replace />} />
        <Route path="/yield-estimator" element={<Navigate to="/calculator" replace />} />
        <Route path="/data-explorer" element={<Navigate to="/portal" replace />} />
        <Route path="/model-analytics" element={<Navigate to="/science" replace />} />
        <Route path="/geographic" element={<Navigate to="/geospatial" replace />} />
        <Route path="/gis" element={<Navigate to="/geospatial" replace />} />
        <Route path="/crop-comparison" element={<Navigate to="/calculator" replace />} />
        
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>

  )
}
