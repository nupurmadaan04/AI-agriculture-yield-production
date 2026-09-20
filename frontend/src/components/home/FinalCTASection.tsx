import React from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, LineChart, ShieldCheck, Sparkles, BookOpen, Activity } from 'lucide-react'
import { Button } from '../ui/Button'

export const FinalCTASection: React.FC = () => {
  return (
    <section className="py-16 md:py-24 bg-card border-b border-border/80">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 text-center space-y-8">
        
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-md bg-primary/10 border border-primary/20 text-primary text-xs font-semibold">
          <ShieldCheck className="w-3.5 h-3.5" />
          <span>Transparent & Audited Agricultural Intelligence</span>
        </div>

        <div className="space-y-3">
          <h2 className="text-2xl sm:text-4xl lg:text-5xl font-extrabold text-foreground tracking-tight">
            Explore the Evidence Behind the Forecast.
          </h2>
          <p className="text-xs sm:text-base text-muted-foreground max-w-2xl mx-auto leading-relaxed">
            Inspect walk-forward validation logs, review explainability attributions, or query governed production forecasts across 20 Indian states.
          </p>
        </div>

        {/* 4 Action Buttons Grid */}
        <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
          <Link to="/forecast">
            <Button size="lg" className="gap-2 font-bold px-6 shadow-sm">
              <LineChart className="w-4 h-4" />
              <span>Explore Forecasting</span>
              <ArrowRight className="w-4 h-4" />
            </Button>
          </Link>

          <Link to="/model-reliability">
            <Button variant="outline" size="lg" className="gap-2 font-semibold px-5">
              <Activity className="w-4 h-4 text-primary" />
              <span>Open Reliability</span>
            </Button>
          </Link>

          <Link to="/explainability">
            <Button variant="outline" size="lg" className="gap-2 font-semibold px-5">
              <Sparkles className="w-4 h-4 text-primary" />
              <span>View Explainability</span>
            </Button>
          </Link>

          <Link to="/science">
            <Button variant="outline" size="lg" className="gap-2 font-semibold px-5">
              <BookOpen className="w-4 h-4 text-primary" />
              <span>Read Scientific Validation</span>
            </Button>
          </Link>
        </div>

        <div className="pt-6 border-t border-border/60 max-w-xl mx-auto text-[11px] text-muted-foreground flex flex-wrap items-center justify-center gap-4">
          <span>20 States Covered</span>
          <span>•</span>
          <span>311 Reporting Districts</span>
          <span>•</span>
          <span>14 Evaluated Commodities</span>
          <span>•</span>
          <span>Zero Data Leakage Guarantee</span>
        </div>

      </div>
    </section>
  )
}
