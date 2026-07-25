import clsx from 'clsx'
import type { ReactNode } from 'react'

const TONE_VALUE_CLASS: Record<string, string> = {
  default: 'text-ink',
  good:    'text-risk-low',
  warn:    'text-risk-medium',
  bad:     'text-risk-critical',
  accent:  'text-accent',
}

const TONE_BORDER_CLASS: Record<string, string> = {
  default: 'border-l-edge-strong',
  good:    'border-l-risk-low',
  warn:    'border-l-risk-medium',
  bad:     'border-l-risk-critical',
  accent:  'border-l-accent',
}

export function StatTile({
  label,
  value,
  unit,
  hint,
  tone = 'default',
  icon,
  className,
}: {
  label: string
  value: ReactNode
  unit?: string
  hint?: ReactNode
  tone?: 'default' | 'good' | 'warn' | 'bad' | 'accent'
  icon?: ReactNode
  className?: string
}) {
  return (
    <div
      className={clsx(
        'flex flex-col gap-1.5 border border-edge bg-surface-1 border-l-4 px-4 py-3',
        TONE_BORDER_CLASS[tone],
        className,
      )}
    >
      {/* Label row */}
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-ink-faint">
          {label}
        </span>
        {icon && <span className="text-ink-faint">{icon}</span>}
      </div>

      {/* Value row */}
      <div className="flex items-baseline gap-1.5">
        <span className={clsx('tnum font-mono text-[32px] font-bold leading-none', TONE_VALUE_CLASS[tone])}>
          {value}
        </span>
        {unit && (
          <span className="font-mono text-[11px] font-medium text-ink-faint">{unit}</span>
        )}
      </div>

      {/* Hint */}
      {hint && (
        <p className="text-[11px] leading-snug text-ink-faint">{hint}</p>
      )}
    </div>
  )
}
