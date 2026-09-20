import React from 'react'
import { Link } from 'react-router-dom'
import { Sprout, ArrowLeft, Home as HomeIcon } from 'lucide-react'
import { Button } from '../components/ui/Button'

export const NotFound: React.FC = () => {
  return (
    <div className="flex-1 flex items-center justify-center py-20 px-4">
      <div className="max-w-md w-full text-center space-y-6">
        <div className="w-16 h-16 rounded-2xl bg-sky-500/10 text-sky-600 dark:text-sky-400 mx-auto flex items-center justify-center">
          <Sprout className="w-8 h-8" />
        </div>
        <div className="space-y-2">
          <p className="text-sm font-bold text-sky-600 dark:text-sky-400 uppercase tracking-wider font-mono">404 Error</p>
          <h1 className="text-3xl font-extrabold text-foreground tracking-tight">Agricultural Page Not Found</h1>
          <p className="text-xs text-muted-foreground leading-relaxed">
            The page or crop model you requested does not exist or has been relocated to another agronomic section.
          </p>
        </div>

        <div className="flex items-center justify-center gap-3">
          <Link to="/">
            <Button className="gap-2 text-xs font-bold">
              <HomeIcon className="w-3.5 h-3.5" />
              <span>Return Home</span>
            </Button>
          </Link>
          <Link to="/calculator">
            <Button variant="outline" className="text-xs font-semibold">
              Open Yield Calculator
            </Button>
          </Link>
        </div>
      </div>
    </div>
  )
}
