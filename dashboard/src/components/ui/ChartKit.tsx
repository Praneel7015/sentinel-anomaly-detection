import type { ReactNode } from 'react'

/**
 * Shared Recharts styling — reads CSS variables so values respect the
 * active theme (light / dark) without any JS re-renders.
 *
 * Call `getChartTheme()` inside a component render to pick up the current
 * CSS variable values from the document root.
 */
function cssVar(name: string, fallback: string): string {
  if (typeof window === 'undefined') return fallback
  const val = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return val ? `rgb(${val})` : fallback
}

export function getChartTheme() {
  return {
    grid: cssVar('--chart-grid', '#e2e8f0'),
    text: cssVar('--chart-text', '#64748b'),
    axis: cssVar('--edge', '#e2e8f0'),
  }
}

export const AXIS = {
  fontSize: 10,
  tickLine: false,
} as const

export const GRID = {
  strokeDasharray: '2 4',
} as const

export interface TooltipEntry {
  name?: string | number
  value?: string | number
  color?: string
  dataKey?: string | number
  payload?: Record<string, unknown>
}

export interface ChartTooltipProps {
  active?: boolean
  payload?: TooltipEntry[]
  label?: string | number
  render?: (entries: TooltipEntry[], label: string | number | undefined) => ReactNode
  labelFormatter?: (label: string | number | undefined) => ReactNode
  valueFormatter?: (value: string | number | undefined, entry: TooltipEntry) => ReactNode
}

export function ChartTooltip({ active, payload, label, render, labelFormatter, valueFormatter }: ChartTooltipProps) {
  if (!active || !payload || payload.length === 0) return null
  return (
    <div className="pointer-events-none max-w-xs rounded border border-edge-strong bg-surface-0 px-2.5 py-2 shadow-lg backdrop-blur">
      {label !== undefined && (
        <div className="mb-1 text-2xs font-semibold uppercase tracking-wider text-ink-faint">
          {labelFormatter ? labelFormatter(label) : label}
        </div>
      )}
      {render ? (
        render(payload, label)
      ) : (
        <ul className="space-y-0.5">
          {payload.map((entry, i) => (
            <li key={i} className="flex items-center gap-2 text-xs">
              <span className="h-2 w-2 shrink-0 rounded-[2px]" style={{ background: entry.color ?? 'rgb(var(--accent))' }} />
              <span className="text-ink-dim">{entry.name}</span>
              <span className="tnum ml-auto font-mono text-ink">
                {valueFormatter ? valueFormatter(entry.value, entry) : entry.value}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
