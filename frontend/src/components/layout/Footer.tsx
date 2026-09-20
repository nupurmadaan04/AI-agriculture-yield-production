import React from 'react'
import { Link } from 'react-router-dom'
import { Sprout, ShieldCheck, ArrowRight, ExternalLink, LineChart, FileText } from 'lucide-react'

export const Footer: React.FC = () => {
  return (
    <footer className="border-t border-border bg-card/60 text-muted-foreground mt-auto">
      {/* Main Footer Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-8">
          
          {/* Brand Col */}
          <div className="lg:col-span-2 space-y-4">
            <Link to="/" className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-primary text-primary-foreground flex items-center justify-center shadow-xs">
                <Sprout className="w-4 h-4" />
              </div>
              <span className="text-base sm:text-lg font-extrabold tracking-tight text-foreground">
                AgriYield<span className="text-primary">.ai</span>
              </span>
            </Link>
            <p className="text-xs leading-relaxed max-w-sm text-muted-foreground">
              Agricultural yield forecasting, temporal walk-forward validation, and governed decision intelligence across 20 Indian states and 311 districts.
            </p>
            <div className="space-y-1 text-xs text-muted-foreground pt-1">
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-3.5 h-3.5 text-primary" />
                <span>71,601 Unified Agricultural Panel Records</span>
              </div>
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-3.5 h-3.5 text-primary" />
                <span>100% Bitwise Reproducible Validation Artifacts</span>
              </div>
            </div>
          </div>

          {/* Forecast & Decision Col */}
          <div className="space-y-3 text-xs">
            <h4 className="font-bold text-foreground text-sm">Forecasting</h4>
            <ul className="space-y-2">
              <li>
                <Link to="/forecast" className="hover:text-foreground transition-colors flex items-center gap-1">
                  <span>Forecast Service</span>
                </Link>
              </li>
              <li>
                <Link to="/decision-intelligence" className="hover:text-foreground transition-colors">
                  Decision Intelligence
                </Link>
              </li>
              <li>
                <Link to="/scenario" className="hover:text-foreground transition-colors">
                  Scenario Simulation Lab
                </Link>
              </li>
              <li>
                <Link to="/early-warning" className="hover:text-foreground transition-colors">
                  Early Warning & Trends
                </Link>
              </li>
              <li>
                <Link to="/modeling-readiness" className="hover:text-foreground transition-colors">
                  Strategy Classifications
                </Link>
              </li>
            </ul>
          </div>

          {/* Science & Verification Col */}
          <div className="space-y-3 text-xs">
            <h4 className="font-bold text-foreground text-sm">Science & Audit</h4>
            <ul className="space-y-2">
              <li>
                <Link to="/science" className="hover:text-foreground transition-colors">
                  Scientific Validation Report
                </Link>
              </li>
              <li>
                <Link to="/model-reliability" className="hover:text-foreground transition-colors">
                  Reliability & Error Drift
                </Link>
              </li>
              <li>
                <Link to="/explainability" className="hover:text-foreground transition-colors">
                  Explainable AI (SHAP)
                </Link>
              </li>
              <li>
                <Link to="/calculator" className="hover:text-foreground transition-colors">
                  Post-Harvest Yield Verification
                </Link>
              </li>
              <li>
                <Link to="/geospatial" className="hover:text-foreground transition-colors">
                  Geospatial District Analysis
                </Link>
              </li>
            </ul>
          </div>

          {/* Data References Col */}
          <div className="space-y-3 text-xs">
            <h4 className="font-bold text-foreground text-sm">Data Provenance</h4>
            <ul className="space-y-2">
              <li>
                <Link to="/portal" className="hover:text-foreground transition-colors">
                  National APY Data Portal
                </Link>
              </li>
              <li>
                <a
                  href="https://upag.gov.in"
                  target="_blank"
                  rel="noreferrer"
                  className="hover:text-foreground transition-colors flex items-center gap-1"
                >
                  <span>UPAg National Portal</span>
                  <ExternalLink className="w-3 h-3 opacity-60" />
                </a>
              </li>
              <li>
                <a
                  href="http://data.icrisat.org"
                  target="_blank"
                  rel="noreferrer"
                  className="hover:text-foreground transition-colors flex items-center gap-1"
                >
                  <span>ICRISAT Panel Datasets</span>
                  <ExternalLink className="w-3 h-3 opacity-60" />
                </a>
              </li>
              <li>
                <Link to="/copilot" className="hover:text-foreground transition-colors">
                  Agricultural AI Copilot
                </Link>
              </li>
              <li>
                <Link to="/contact" className="hover:text-foreground transition-colors">
                  Research Inquiries
                </Link>
              </li>
            </ul>
          </div>

        </div>

        {/* Bottom copyright line */}
        <div className="pt-8 mt-8 border-t border-border/50 flex flex-col sm:flex-row items-center justify-between gap-4 text-[11px]">
          <p>© {new Date().getFullYear()} Agricultural Intelligence Platform. Designed for Empirical Research & Decision Governance.</p>
          <div className="flex items-center gap-4">
            <Link to="/science" className="hover:underline">Scientific Methodology</Link>
            <span>•</span>
            <Link to="/portal" className="hover:underline">Data Provenance</Link>
            <span>•</span>
            <Link to="/modeling-readiness" className="hover:underline">Model Governance</Link>
            <span>•</span>
            <Link to="/forecast" className="hover:underline">Forecast Provenance</Link>
          </div>
        </div>
      </div>
    </footer>
  )
}
