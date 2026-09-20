import React from 'react'
import { Outlet, ScrollRestoration } from 'react-router-dom'
import { Navbar } from './Navbar'
import { Footer } from './Footer'

export const WebShell: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground relative selection:bg-sky-500/20 selection:text-sky-600">
      <Navbar />
      <main className="flex-1 flex flex-col">
        <Outlet />
      </main>
      <Footer />
    </div>
  )
}
