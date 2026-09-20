import React from 'react'
import { HeroSection } from '../components/home/HeroSection'
import { PlatformMetrics } from '../components/home/PlatformMetrics'
import { AnalyticalModes } from '../components/home/AnalyticalModes'
import { DecisionPipeline } from '../components/home/DecisionPipeline'
import { StrategyOverview } from '../components/home/StrategyOverview'
import { ValidationEvidence } from '../components/home/ValidationEvidence'
import { YieldVerificationSection } from '../components/home/YieldVerificationSection'
import { FinalCTASection } from '../components/home/FinalCTASection'

export const Home: React.FC = () => {
  return (
    <div className="flex flex-col w-full overflow-hidden">
      {/* 1. HERO SECTION */}
      <HeroSection />

      {/* 2. VERIFIED PLATFORM SCALE */}
      <PlatformMetrics />

      {/* 3. TWO ANALYTICAL MODES */}
      <AnalyticalModes />

      {/* 4. HOW THE PLATFORM DECIDES */}
      <DecisionPipeline />

      {/* 5. MODEL GOVERNANCE */}
      <StrategyOverview />

      {/* 6. SCIENTIFIC VALIDATION */}
      <ValidationEvidence />

      {/* 7. POST-HARVEST CALCULATOR */}
      <YieldVerificationSection />

      {/* 8. FINAL CTA */}
      <FinalCTASection />
    </div>
  )
}
