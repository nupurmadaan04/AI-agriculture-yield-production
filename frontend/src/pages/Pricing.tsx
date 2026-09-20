import React from 'react'
import { Link } from 'react-router-dom'
import { Check, Sparkles, HelpCircle, ArrowRight, ShieldCheck } from 'lucide-react'
import { Button } from '../components/ui/Button'
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'

export const Pricing: React.FC = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 space-y-16">
      {/* Header */}
      <div className="max-w-3xl mx-auto text-center space-y-3">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-sky-500/10 text-sky-600 dark:text-sky-400 text-xs font-semibold">
          <Sparkles className="w-3.5 h-3.5" />
          <span>SaaS Pricing & Subscription Plans</span>
        </div>
        <h1 className="text-3xl sm:text-5xl font-extrabold text-foreground tracking-tight">
          Transparent Pricing for Every Agricultural Scale
        </h1>
        <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
          From open-access academic research to high-volume grain trading desks and national agribusiness cooperatives.
        </p>
      </div>

      {/* Pricing Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 items-stretch">
        {/* Tier 1: Free Open Research */}
        <Card className="flex flex-col justify-between hover:border-border/80 transition-all">
          <CardHeader className="space-y-2">
            <Badge variant="outline" className="w-fit">Open Research</Badge>
            <CardTitle className="text-xl font-bold">Academic & Public</CardTitle>
            <CardDescription className="text-xs">
              Essential agronomic calculators and public agricultural panel explorer.
            </CardDescription>
            <div className="pt-4">
              <span className="text-4xl font-extrabold text-foreground font-mono">$0</span>
              <span className="text-xs text-muted-foreground"> / forever</span>
            </div>
          </CardHeader>

          <CardContent className="space-y-3 text-xs flex-1">
            <div className="space-y-2 text-muted-foreground">
              <div className="flex items-center gap-2 text-foreground font-medium">
                <Check className="w-4 h-4 text-emerald-500 shrink-0" />
                <span>Interactive Yield Calculator Tool</span>
              </div>
              <div className="flex items-center gap-2 text-foreground font-medium">
                <Check className="w-4 h-4 text-emerald-500 shrink-0" />
                <span>71,601 Unified Agricultural Records</span>
              </div>
              <div className="flex items-center gap-2">
                <Check className="w-4 h-4 text-emerald-500 shrink-0" />
                <span>National APY Explorer (20 States)</span>
              </div>
              <div className="flex items-center gap-2">
                <Check className="w-4 h-4 text-emerald-500 shrink-0" />
                <span>Scientific Whitepaper Access</span>
              </div>
            </div>
          </CardContent>

          <CardFooter className="pt-4 border-t border-border">
            <Link to="/calculator" className="w-full">
              <Button variant="outline" className="w-full text-xs font-semibold">
                Launch Free Calculator
              </Button>
            </Link>
          </CardFooter>
        </Card>

        {/* Tier 2: Agronomist Pro (Popular) */}
        <Card className="flex flex-col justify-between border-sky-500 ring-2 ring-sky-500/20 shadow-xl shadow-sky-500/10 relative">
          <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-0.5 rounded-full bg-sky-500 text-white text-[10px] font-bold uppercase tracking-wider">
            Most Popular
          </div>
          <CardHeader className="space-y-2 pt-6">
            <Badge variant="blue" className="w-fit">Professional</Badge>
            <CardTitle className="text-xl font-bold">Agronomist Pro</CardTitle>
            <CardDescription className="text-xs">
              For agricultural consultants, input retailers, and precision farm managers.
            </CardDescription>
            <div className="pt-4">
              <span className="text-4xl font-extrabold text-foreground font-mono">$49</span>
              <span className="text-xs text-muted-foreground"> / month</span>
            </div>
          </CardHeader>

          <CardContent className="space-y-3 text-xs flex-1">
            <div className="space-y-2 text-muted-foreground">
              <div className="flex items-center gap-2 text-foreground font-semibold">
                <Check className="w-4 h-4 text-sky-500 shrink-0" />
                <span>Everything in Open Research</span>
              </div>
              <div className="flex items-center gap-2 text-foreground font-medium">
                <Check className="w-4 h-4 text-sky-500 shrink-0" />
                <span>Multi-Crop Simulation (Rice, Wheat, Maize)</span>
              </div>
              <div className="flex items-center gap-2 text-foreground font-medium">
                <Check className="w-4 h-4 text-sky-500 shrink-0" />
                <span>95% Confidence Bounds & Drought Risk Score</span>
              </div>
              <div className="flex items-center gap-2 text-foreground font-medium">
                <Check className="w-4 h-4 text-sky-500 shrink-0" />
                <span>Automated PDF Executive Brief Export</span>
              </div>
              <div className="flex items-center gap-2">
                <Check className="w-4 h-4 text-sky-500 shrink-0" />
                <span>Climate Anomaly Sensitivity Modeling</span>
              </div>
            </div>
          </CardContent>

          <CardFooter className="pt-4 border-t border-border">
            <Link to="/contact" className="w-full">
              <Button className="w-full text-xs font-bold gap-2">
                <span>Start 14-Day Pro Trial</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Button>
            </Link>
          </CardFooter>
        </Card>

        {/* Tier 3: Enterprise Agribusiness & Trading Desk */}
        <Card className="flex flex-col justify-between hover:border-border/80 transition-all">
          <CardHeader className="space-y-2">
            <Badge variant="secondary" className="w-fit">Enterprise</Badge>
            <CardTitle className="text-xl font-bold">Agribusiness & Traders</CardTitle>
            <CardDescription className="text-xs">
              For commodity trading desks, crop insurers, and government ministries.
            </CardDescription>
            <div className="pt-4">
              <span className="text-4xl font-extrabold text-foreground font-mono">$499</span>
              <span className="text-xs text-muted-foreground"> / month (or custom)</span>
            </div>
          </CardHeader>

          <CardContent className="space-y-3 text-xs flex-1">
            <div className="space-y-2 text-muted-foreground">
              <div className="flex items-center gap-2 text-foreground font-semibold">
                <Check className="w-4 h-4 text-purple-500 shrink-0" />
                <span>Everything in Agronomist Pro</span>
              </div>
              <div className="flex items-center gap-2 text-foreground font-medium">
                <Check className="w-4 h-4 text-purple-500 shrink-0" />
                <span>Sentinel-2 Satellite Remote Sensing Ingestion</span>
              </div>
              <div className="flex items-center gap-2 text-foreground font-medium">
                <Check className="w-4 h-4 text-purple-500 shrink-0" />
                <span>Pre-Harvest Supply Shock & Deficit Alerts</span>
              </div>
              <div className="flex items-center gap-2 text-foreground font-medium">
                <Check className="w-4 h-4 text-purple-500 shrink-0" />
                <span>RESTful JSON API & Webhooks Access</span>
              </div>
              <div className="flex items-center gap-2">
                <Check className="w-4 h-4 text-purple-500 shrink-0" />
                <span>Dedicated Agronomist & ML Engineer SLA</span>
              </div>
            </div>
          </CardContent>

          <CardFooter className="pt-4 border-t border-border">
            <Link to="/contact" className="w-full">
              <Button variant="secondary" className="w-full text-xs font-bold">
                Contact Enterprise Sales
              </Button>
            </Link>
          </CardFooter>
        </Card>
      </div>

      {/* Money Back Guarantee & Security */}
      <div className="p-6 rounded-2xl bg-card border border-border text-center space-y-2 max-w-2xl mx-auto">
        <ShieldCheck className="w-6 h-6 text-emerald-500 mx-auto" />
        <h4 className="font-bold text-sm text-foreground">100% Scientific Transparency Guarantee</h4>
        <p className="text-xs text-muted-foreground leading-relaxed">
          All baseline statistical figures are derived from verified open-access ICRISAT and government datasets. Cancel anytime.
        </p>
      </div>
    </div>
  )
}
