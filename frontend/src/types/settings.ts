export type ThemeMode = 'light' | 'dark' | 'system'
export type AccentColor = 'blue' | 'green' | 'slate'

export interface AppSettings {
  theme: ThemeMode
  accentColor: AccentColor
  compactMode: boolean
  sidebarDefaultCollapsed: boolean
  enableAnimations: boolean
  defaultCrop: string
  defaultYear: number
  defaultState: string
  dataAlerts: boolean
  modelAlerts: boolean
  reportNotifications: boolean
  localProcessingOnly: boolean
}
