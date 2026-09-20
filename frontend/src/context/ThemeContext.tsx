import React, { createContext, useContext, useEffect, useState } from 'react'
import { ThemeMode, AccentColor } from '../types/settings'

interface ThemeContextType {
  theme: ThemeMode
  setTheme: (mode: ThemeMode) => void
  accentColor: AccentColor
  setAccentColor: (accent: AccentColor) => void
  resolvedTheme: 'light' | 'dark'
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [theme, setThemeState] = useState<ThemeMode>(() => {
    const saved = localStorage.getItem('agri_theme_mode')
    return (saved as ThemeMode) || 'system'
  })

  const [accentColor, setAccentColorState] = useState<AccentColor>(() => {
    const saved = localStorage.getItem('agri_accent_color')
    return (saved as AccentColor) || 'blue'
  })

  const [resolvedTheme, setResolvedTheme] = useState<'light' | 'dark'>('light')

  useEffect(() => {
    const root = document.documentElement
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')

    const applyTheme = () => {
      let isDark = false
      if (theme === 'dark') {
        isDark = true
      } else if (theme === 'light') {
        isDark = false
      } else {
        isDark = mediaQuery.matches
      }

      setResolvedTheme(isDark ? 'dark' : 'light')
      if (isDark) {
        root.classList.add('dark')
      } else {
        root.classList.remove('dark')
      }
    }

    applyTheme()
    mediaQuery.addEventListener('change', applyTheme)
    return () => mediaQuery.removeEventListener('change', applyTheme)
  }, [theme])

  const setTheme = (mode: ThemeMode) => {
    setThemeState(mode)
    localStorage.setItem('agri_theme_mode', mode)
  }

  const setAccentColor = (accent: AccentColor) => {
    setAccentColorState(accent)
    localStorage.setItem('agri_accent_color', accent)
  }

  return (
    <ThemeContext.Provider value={{ theme, setTheme, accentColor, setAccentColor, resolvedTheme }}>
      {children}
    </ThemeContext.Provider>
  )
}

export const useTheme = () => {
  const context = useContext(ThemeContext)
  if (!context) throw new Error('useTheme must be used within a ThemeProvider')
  return context
}
