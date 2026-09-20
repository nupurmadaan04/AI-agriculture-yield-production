import React, { useState } from 'react'
import {
  Mail,
  Phone,
  MapPin,
  Send,
  CheckCircle2,
  Sparkles,
  Building,
  Globe2
} from 'lucide-react'
import { Button } from '../components/ui/Button'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'

export const Contact: React.FC = () => {
  const [submitted, setSubmitted] = useState(false)
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    org: '',
    role: 'Grain Trader & Merchandiser',
    interest: 'API & Satellite Ingestion',
    message: ''
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitted(true)
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 space-y-12">
      {/* Header */}
      <div className="max-w-3xl space-y-3">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-sky-500/10 text-sky-600 dark:text-sky-400 text-xs font-semibold">
          <Mail className="w-3.5 h-3.5" />
          <span>Connect with Our Team</span>
        </div>
        <h1 className="text-3xl sm:text-5xl font-extrabold text-foreground tracking-tight">
          Schedule an Agricultural Intelligence Consultation
        </h1>
        <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
          Request a personalized walkthrough, discuss custom district data integrations, or explore enterprise API access.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Form Col (7 cols) */}
        <Card className="lg:col-span-7">
          <CardHeader>
            <CardTitle className="text-lg font-bold">Request a Live Demonstration</CardTitle>
            <CardDescription className="text-xs">
              Fill in your details and an agronomic data engineer will respond within 24 hours.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {submitted ? (
              <div className="p-8 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 text-center space-y-3 animate-in fade-in-50">
                <CheckCircle2 className="w-10 h-10 text-emerald-500 mx-auto" />
                <h3 className="text-base font-bold text-foreground">Demonstration Request Received!</h3>
                <p className="text-xs text-muted-foreground max-w-md mx-auto leading-relaxed">
                  Thank you for reaching out, <strong className="text-foreground">{formData.name}</strong>. Our agricultural data team will contact you at <strong className="text-foreground">{formData.email}</strong> shortly.
                </p>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setSubmitted(false)}
                  className="text-xs mt-2"
                >
                  Send Another Inquiry
                </Button>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="space-y-4 text-xs">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="font-semibold text-foreground">Full Name</label>
                    <input
                      required
                      type="text"
                      placeholder="e.g. Dr. Rajesh Kumar"
                      value={formData.name}
                      onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                      className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs"
                    />
                  </div>

                  <div className="space-y-1.5">
                    <label className="font-semibold text-foreground">Work Email</label>
                    <input
                      required
                      type="email"
                      placeholder="rajesh@agricorp.com"
                      value={formData.email}
                      onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                      className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="font-semibold text-foreground">Organization / Company</label>
                    <input
                      type="text"
                      placeholder="e.g. National Grain Traders Federation"
                      value={formData.org}
                      onChange={(e) => setFormData({ ...formData, org: e.target.value })}
                      className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs"
                    />
                  </div>

                  <div className="space-y-1.5">
                    <label className="font-semibold text-foreground">Industry Role</label>
                    <select
                      value={formData.role}
                      onChange={(e) => setFormData({ ...formData, role: e.target.value })}
                      className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs cursor-pointer"
                    >
                      <option value="Grain Trader & Merchandiser">Grain Trader & Merchandiser</option>
                      <option value="Agribusiness & Seed Manufacturer">Agribusiness & Seed Manufacturer</option>
                      <option value="Crop Insurance Underwriter">Crop Insurance Underwriter</option>
                      <option value="Academic Researcher / Agronomist">Academic Researcher / Agronomist</option>
                      <option value="Government & Policy Analyst">Government & Policy Analyst</option>
                    </select>
                  </div>
                </div>

                <div className="space-y-1.5">
                  <label className="font-semibold text-foreground">Inquiry Details</label>
                  <textarea
                    rows={4}
                    placeholder="Describe your agricultural use-case, target regions, or specific dataset requirements..."
                    value={formData.message}
                    onChange={(e) => setFormData({ ...formData, message: e.target.value })}
                    className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs"
                  />
                </div>

                <Button type="submit" className="w-full gap-2 text-xs font-bold py-2.5">
                  <Send className="w-3.5 h-3.5" />
                  <span>Submit Demo Request</span>
                </Button>
              </form>
            )}
          </CardContent>
        </Card>

        {/* Info Col (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          <Card className="p-6 space-y-4 text-xs text-muted-foreground">
            <h3 className="font-bold text-sm text-foreground">AgriYield AI Research Hub</h3>
            <p className="leading-relaxed">
              Our modeling team works closely with agricultural economists, crop modelers, and remote sensing specialists across India.
            </p>

            <div className="space-y-3 pt-2">
              <div className="flex items-center gap-3 text-foreground">
                <MapPin className="w-4 h-4 text-sky-500 shrink-0" />
                <span>ICRISAT Agricultural Corridor, Hyderabad / New Delhi, India</span>
              </div>
              <div className="flex items-center gap-3 text-foreground">
                <Mail className="w-4 h-4 text-sky-500 shrink-0" />
                <span>research@agriyield.ai</span>
              </div>
              <div className="flex items-center gap-3 text-foreground">
                <Globe2 className="w-4 h-4 text-sky-500 shrink-0" />
                <span>UPAg & ICRISAT Open Statistical Partner</span>
              </div>
            </div>
          </Card>

          <div className="p-5 rounded-2xl bg-sky-500/10 border border-sky-500/20 text-xs text-sky-950 dark:text-sky-200 space-y-2">
            <p className="font-bold flex items-center gap-1.5">
              <Sparkles className="w-4 h-4 text-sky-500" />
              <span>Looking for Quick Calculations?</span>
            </p>
            <p className="opacity-90 leading-relaxed">
              You do not need an account to use our public Yield Estimator tool. Try it directly on the calculator page.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
