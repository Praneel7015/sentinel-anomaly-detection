import type { ReactNode } from 'react'

/** Shared Recharts styling so every chart in the console reads the same. */
export const AXIS = {
  stroke: '#5d6879',
  fontSize: 10,
  tickLine: false,
  axisLine: { stroke: '#232936' },
} as const

export const GRID = {
  stroke: '#1a1f29',
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
  /** Override the rendered body entirely. */
  render?: (entries: TooltipEntry[], label: string | number | undefined) => ReactNode
  labelFormatter?: (label: string | number | undefined) => ReactNode
  valueFormatter?: (value: string | number | undefined, entry: TooltipEntry) => ReactNode
}

export function ChartTooltip({ active, payload, label, render, labelFormatter, valueFormatter }: ChartTooltipProps) {
  if (!active || !payload || payload.length === 0) return null
  return (
    <div className="pointer-events-none max-w-xs rounded border border-edge-strong bg-surface-0/95 px-2.5 py-2 shadow-xl shadow-black/60 backdrop-blur">
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
              <span className="h-2 w-2 shrink-0 rounded-[2px]" style={{ background: entry.color ?? '#22d3ee' }} />
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
