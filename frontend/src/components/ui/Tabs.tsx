import React from 'react'
import { cn } from '../../lib/utils'

interface TabsProps {
  value: string
  onValueChange: (value: string) => void
  children: React.ReactNode
  className?: string
}

export const Tabs: React.FC<TabsProps> = ({ value, onValueChange, children, className }) => {
  return (
    <div className={cn("space-y-4", className)}>
      {React.Children.map(children, child => {
        if (React.isValidElement(child)) {
          return React.cloneElement(child, { currentValue: value, onValueChange } as any)
        }
        return child
      })}
    </div>
  )
}

export const TabsList: React.FC<{ children: React.ReactNode; className?: string; currentValue?: string; onValueChange?: (val: string) => void }> = ({
  children,
  className,
  currentValue,
  onValueChange
}) => {
  return (
    <div className={cn("inline-flex h-10 items-center justify-center rounded-lg bg-muted p-1 text-muted-foreground", className)}>
      {React.Children.map(children, child => {
        if (React.isValidElement(child)) {
          return React.cloneElement(child, { currentValue, onValueChange } as any)
        }
        return child
      })}
    </div>
  )
}

export const TabsTrigger: React.FC<{
  value: string
  children: React.ReactNode
  className?: string
  currentValue?: string
  onValueChange?: (val: string) => void
}> = ({ value, children, className, currentValue, onValueChange }) => {
  const isActive = currentValue === value
  return (
    <button
      type="button"
      onClick={() => onValueChange?.(value)}
      className={cn(
        "inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 cursor-pointer",
        isActive
          ? "bg-card text-foreground shadow-xs"
          : "text-muted-foreground hover:text-foreground hover:bg-card/40",
        className
      )}
    >
      {children}
    </button>
  )
}

export const TabsContent: React.FC<{
  value: string
  children: React.ReactNode
  className?: string
  currentValue?: string
}> = ({ value, children, className, currentValue }) => {
  if (currentValue !== value) return null
  return <div className={cn("focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2", className)}>{children}</div>
}
