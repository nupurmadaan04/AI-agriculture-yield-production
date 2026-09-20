import React, { useState, useRef, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import {
  Sprout,
  ChevronDown,
  Menu,
  X,
  Sun,
  Moon,
  LineChart,
  ShieldCheck,
  Activity,
  Layers,
  Sparkles,
  Bot,
  Clock,
  Compass,
  Database,
  BookOpen,
  Calculator,
  Sliders,
  AlertCircle,
  FileCheck2,
  BarChart3,
  Globe2,
  ArrowRight
} from 'lucide-react'
import { useTheme } from '../../context/ThemeContext'
import { Button } from '../ui/Button'

interface NavGroup {
  name: string
  items: {
    name: string
    path: string
    description: string
    badge?: string
    icon: React.ElementType
  }[]
}

const navGroups: NavGroup[] = [
  {
    name: 'Forecast',
    items: [
      {
        name: 'Forecast Service',
        path: '/forecast',
        description: 'Governed pre-season multi-crop yield prediction engine.',
        badge: 'Production',
        icon: LineChart,
      },
      {
        name: 'Forecast Audit & Provenance',
        path: '/forecast',
        description: 'Cryptographic execution logs, model artifacts, and audit trails.',
        icon: ShieldCheck,
      },
    ],
  },
  {
    name: 'Decision Intelligence',
    items: [
      {
        name: 'Decision Brief',
        path: '/decision-intelligence',
        description: 'Auditable policy recommendations, agronomic trade-offs, and supply shocks.',
        badge: 'Executive',
        icon: Compass,
      },
      {
        name: 'Scenario Lab',
        path: '/scenario',
        description: 'Simulate climate shocks, irrigation shifts, and fertilizer changes.',
        icon: Sliders,
      },
      {
        name: 'Early Warning Engine',
        path: '/early-warning',
        description: 'Theil-Sen trend diagnostics, drought alerts, and structural shifts.',
        icon: Clock,
      },
    ],
  },
  {
    name: 'Analytics',
    items: [
      {
        name: 'Reliability & Drift',
        path: '/model-reliability',
        description: 'Calibration curves, covariate drift tracking, and error regimes.',
        icon: Activity,
      },
      {
        name: 'Explainable AI (XAI)',
        path: '/explainability',
        description: 'SHAP values, Permutation feature importance, and partial dependence.',
        icon: Sparkles,
      },
      {
        name: 'Geospatial Intelligence',
        path: '/geospatial',
        description: 'Pan-India district maps, agro-climatic clusters, and spatial outliers.',
        icon: Globe2,
      },
      {
        name: 'Temporal Monitoring',
        path: '/monitoring',
        description: 'Continuous signal surveillance, change detection, and health probes.',
        icon: BarChart3,
      },
    ],
  },
  {
    name: 'Data & Science',
    items: [
      {
        name: 'Agricultural Data Portal',
        path: '/portal',
        description: '71,601 unified records across 20 Indian states and 311 districts.',
        icon: Database,
      },
      {
        name: 'Scientific Validation',
        path: '/science',
        description: 'Formal whitepaper on expanding walk-forward methodology & leak prevention.',
        badge: 'Audit',
        icon: BookOpen,
      },
      {
        name: 'Modeling Readiness',
        path: '/modeling-readiness',
        description: '14-crop tournament evaluations and certified strategy registries.',
        icon: FileCheck2,
      },
    ],
  },
  {
    name: 'Tools',
    items: [
      {
        name: 'Yield Verification',
        path: '/calculator',
        description: 'Post-harvest mathematical identity consistency check.',
        icon: Calculator,
      },
      {
        name: 'Agricultural AI Copilot',
        path: '/copilot',
        description: 'Evidence-grounded agronomic assistant for query reasoning.',
        badge: 'AI',
        icon: Bot,
      },
      {
        name: 'Risk Analytics',
        path: '/intelligence',
        description: 'Historical yield anomalies and empirical state risk profiles.',
        icon: AlertCircle,
      },
      {
        name: 'Decision Support',
        path: '/decision-support',
        description: 'Automated executive summaries and PDF brief generation.',
        icon: Layers,
      },
    ],
  },
]

export const Navbar: React.FC = () => {
  const location = useLocation()
  const { theme, setTheme } = useTheme()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [openDropdown, setOpenDropdown] = useState<string | null>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)

  const toggleTheme = () => setTheme(theme === 'dark' ? 'light' : 'dark')

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setOpenDropdown(null)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  // Close menus on route change
  useEffect(() => {
    setOpenDropdown(null)
    setMobileMenuOpen(false)
  }, [location.pathname])

  const isGroupActive = (group: NavGroup) => {
    return group.items.some(item => location.pathname === item.path)
  }

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/80 bg-background/95 backdrop-blur-md">
      
      {/* Top Status Strip */}
      <div className="bg-muted/50 border-b border-border/60 px-4 py-1 text-[11px] text-muted-foreground">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="flex h-2 w-2 rounded-full bg-emerald-600 dark:text-emerald-400 animate-pulse" />
            <span className="font-semibold text-foreground">Agricultural Decision Intelligence Platform:</span>
            <span className="hidden sm:inline">20 States • 311 Districts • 14 Evaluated Crops • Walk-Forward Governed</span>
          </div>
          <div className="flex items-center gap-3 font-medium">
            <Link to="/forecast" className="hover:text-foreground transition-colors flex items-center gap-1">
              <span>Pre-Season Forecasts</span>
            </Link>
            <span className="opacity-40">|</span>
            <Link to="/science" className="hover:text-foreground transition-colors flex items-center gap-1">
              <span>Scientific Audit</span>
            </Link>
          </div>
        </div>
      </div>

      {/* Main Navigation Bar */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8" ref={dropdownRef}>
        <div className="flex items-center justify-between h-16">
          
          {/* Brand Logo */}
          <Link to="/" className="flex items-center gap-2.5 group">
            <div className="w-9 h-9 rounded-lg bg-primary text-primary-foreground flex items-center justify-center shadow-xs group-hover:scale-105 transition-transform">
              <Sprout className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="text-base sm:text-lg font-extrabold tracking-tight text-foreground font-sans">
                  AgriYield<span className="text-primary">.ai</span>
                </span>
                <span className="text-[9px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-bold uppercase tracking-wider">
                  Intelligence
                </span>
              </div>
              <p className="text-[10px] text-muted-foreground hidden sm:block -mt-0.5">
                Forecasting & Decision Governance
              </p>
            </div>
          </Link>

          {/* Desktop Grouped Navigation Links */}
          <nav className="hidden lg:flex items-center gap-1">
            {navGroups.map((group) => {
              const isOpen = openDropdown === group.name
              const active = isGroupActive(group)

              return (
                <div key={group.name} className="relative">
                  <button
                    onClick={() => setOpenDropdown(isOpen ? null : group.name)}
                    className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all flex items-center gap-1 cursor-pointer ${
                      active || isOpen
                        ? 'text-primary bg-primary/10 font-bold'
                        : 'text-muted-foreground hover:text-foreground hover:bg-muted/60'
                    }`}
                  >
                    <span>{group.name}</span>
                    <ChevronDown className={`w-3.5 h-3.5 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
                  </button>

                  {/* Dropdown Menu */}
                  {isOpen && (
                    <div className="absolute top-full left-0 mt-1 w-72 rounded-xl border border-border bg-card p-2 shadow-elevated animate-in fade-in slide-in-from-top-1 z-50">
                      <div className="space-y-1">
                        {group.items.map((item) => {
                          const Icon = item.icon
                          const isItemActive = location.pathname === item.path
                          return (
                            <Link
                              key={item.name}
                              to={item.path}
                              className={`p-2.5 rounded-lg flex items-start gap-2.5 transition-colors ${
                                isItemActive
                                  ? 'bg-primary/10 text-primary font-semibold'
                                  : 'hover:bg-muted/70 text-foreground'
                              }`}
                            >
                              <div className="p-1.5 rounded-md bg-muted text-foreground shrink-0 mt-0.5">
                                <Icon className="w-3.5 h-3.5 text-primary" />
                              </div>
                              <div className="space-y-0.5 min-w-0 flex-1">
                                <div className="flex items-center justify-between">
                                  <span className="text-xs font-bold truncate">{item.name}</span>
                                  {item.badge && (
                                    <span className="text-[9px] font-semibold px-1.5 py-0.2 rounded-full bg-primary/15 text-primary">
                                      {item.badge}
                                    </span>
                                  )}
                                </div>
                                <p className="text-[10px] text-muted-foreground leading-tight line-clamp-2">
                                  {item.description}
                                </p>
                              </div>
                            </Link>
                          )
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </nav>

          {/* Right Controls */}
          <div className="hidden lg:flex items-center gap-3">
            {/* Theme Toggle */}
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors cursor-pointer"
              title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            >
              {theme === 'dark' ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4" />}
            </button>

            <Link to="/forecast">
              <Button size="sm" className="gap-1.5 text-xs font-bold shadow-xs">
                <span>Explore Forecasting</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Button>
            </Link>
          </div>

          {/* Mobile Menu Toggle */}
          <div className="flex lg:hidden items-center gap-2">
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg text-muted-foreground hover:text-foreground"
            >
              {theme === 'dark' ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4" />}
            </button>
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>

        </div>
      </div>

      {/* Mobile Drawer Menu */}
      {mobileMenuOpen && (
        <div className="lg:hidden border-t border-border bg-card/95 backdrop-blur-md px-4 pt-3 pb-8 space-y-4 shadow-elevated max-h-[85vh] overflow-y-auto">
          {navGroups.map((group) => (
            <div key={group.name} className="space-y-1.5">
              <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground px-2">
                {group.name}
              </span>
              <div className="grid grid-cols-1 gap-1">
                {group.items.map((item) => {
                  const Icon = item.icon
                  const isItemActive = location.pathname === item.path
                  return (
                    <Link
                      key={item.name}
                      to={item.path}
                      onClick={() => setMobileMenuOpen(false)}
                      className={`p-2.5 rounded-lg flex items-center justify-between text-xs font-semibold ${
                        isItemActive
                          ? 'bg-primary/10 text-primary font-bold'
                          : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                      }`}
                    >
                      <div className="flex items-center gap-2.5">
                        <Icon className="w-4 h-4 text-primary" />
                        <span>{item.name}</span>
                      </div>
                      {item.badge && (
                        <span className="text-[9px] px-1.5 py-0.2 rounded-full bg-primary/10 text-primary">
                          {item.badge}
                        </span>
                      )}
                    </Link>
                  )
                })}
              </div>
            </div>
          ))}

          <div className="pt-3 border-t border-border flex flex-col gap-2">
            <Link to="/forecast" onClick={() => setMobileMenuOpen(false)}>
              <Button className="w-full justify-center text-xs font-bold gap-2">
                <span>Explore Forecasting</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Button>
            </Link>
          </div>
        </div>
      )}

    </header>
  )
}
