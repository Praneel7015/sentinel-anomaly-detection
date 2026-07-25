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
  return (
    <div className={clsx('panel flex flex-col gap-1 p-3', className)}>
      <div className="flex items-center justify-between gap-2">
        <span className="text-2xs font-semibold uppercase tracking-[0.12em] text-ink-faint">{label}</span>
        {icon && <span className="text-ink-faint">{icon}</span>}
      </div>
      <div className="flex items-baseline gap-1">
        <span className={clsx('tnum font-mono text-2xl font-semibold leading-none', tones[tone])}>{value}</span>
        {unit && <span className="text-xs text-ink-faint">{unit}</span>}
      </div>
      {hint && <p className="text-2xs leading-snug text-ink-faint">{hint}</p>}
    </div>
  )
}
