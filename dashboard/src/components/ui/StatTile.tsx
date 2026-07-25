import clsx from 'clsx'
import type { ReactNode } from 'react'

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
  const tones: Record<string, string> = {
    default: 'text-ink',
    good: 'text-risk-low',
    warn: 'text-risk-medium',
    bad: 'text-risk-critical',
    accent: 'text-accent',
  }
  const borderTones: Record<string, string> = {
    default: 'border-l-edge-strong',
    good: 'border-l-risk-low',
    warn: 'border-l-risk-medium',
    bad: 'border-l-risk-critical',
    accent: 'border-l-accent',
  }
  return (
    <div className={clsx('panel flex flex-col gap-1 border-l-2 p-4', borderTones[tone], className)}>
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-semibold uppercase tracking-widest text-ink-faint">{label}</span>
        {icon && <span className="text-ink-faint">{icon}</span>}
      </div>
      <div className="flex items-baseline gap-1">
        <span className={clsx('tnum font-mono text-3xl font-bold leading-none', tones[tone])}>{value}</span>
        {unit && <span className="text-xs text-ink-faint">{unit}</span>}
      </div>
      {hint && <p className="text-xs leading-snug text-ink-faint">{hint}</p>}
    </div>
  )
}
