import React from 'react'

interface ScenarioSliderProps {
  label: string
  unit: string
  baselineValue: number
  currentValue: number
  min: number
  max: number
  step?: number
  onChange: (value: number) => void
  onReset: () => void
}

export const ScenarioSlider: React.FC<ScenarioSliderProps> = ({
  label,
  unit,
  baselineValue,
  currentValue,
  min,
  max,
  step = 1,
  onChange,
  onReset
}) => {
  const delta = currentValue - baselineValue
  const pctChange = baselineValue !== 0 ? (delta / baselineValue) * 100 : 0
  const hasChanged = Math.abs(delta) > 1e-4

  return (
    <div className="p-3.5 rounded-xl border border-border bg-card/60 space-y-2 text-xs">
      <div className="flex items-center justify-between font-semibold">
        <span className="text-foreground">{label}</span>
        <div className="flex items-center gap-2">
          {hasChanged && (
            <span
              className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold ${
                pctChange > 0
                  ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
                  : 'bg-amber-500/10 text-amber-600 dark:text-amber-400'
              }`}
            >
              {pctChange > 0 ? '+' : ''}{pctChange.toFixed(1)}%
            </span>
          )}
          <span className="font-mono text-foreground font-bold">
            {currentValue.toFixed(1)} <span className="text-muted-foreground font-normal">{unit}</span>
          </span>
        </div>
      </div>

      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={currentValue}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 bg-muted rounded-lg appearance-none cursor-pointer accent-sky-500"
      />

      <div className="flex items-center justify-between text-[11px] text-muted-foreground pt-0.5">
        <span>Baseline: <strong className="font-mono text-foreground">{baselineValue.toFixed(1)} {unit}</strong></span>
        {hasChanged ? (
          <button
            type="button"
            onClick={onReset}
            className="text-sky-600 dark:text-sky-400 hover:underline font-semibold cursor-pointer"
          >
            Reset
          </button>
        ) : (
          <span className="italic opacity-60">Unmodified</span>
        )}
      </div>
    </div>
  )
}
