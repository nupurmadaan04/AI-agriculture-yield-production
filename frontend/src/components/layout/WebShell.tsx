import React from 'react'
import { Outlet, ScrollRestoration } from 'react-router-dom'
import { Navbar } from './Navbar'
import { Footer } from './Footer'

export const WebShell: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground relative selection:bg-sky-500/20 selection:text-sky-600">
      {/* Skip-to-content link — visually hidden until focused (WCAG 2.4.1) */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-[100] focus:px-4 focus:py-2 focus:rounded-md focus:bg-primary focus:text-primary-foreground focus:font-semibold focus:text-sm focus:shadow-lg"
      >
        Skip to main content
      </a>
      <Navbar />
      <main id="main-content" className="flex-1 flex flex-col" tabIndex={-1}>
        <Outlet />
      </main>
      <Footer />
    </div>
  )
}
