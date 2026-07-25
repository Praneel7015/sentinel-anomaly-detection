import clsx from 'clsx'
import type { ReactNode } from 'react'
import { attackMeta } from '../../lib/domain'

export function Chip({
  children,
  className,
  title,
}: {
  children: ReactNode
  className?: string
  title?: string
}) {
  return (
    <span
      className={clsx('chip', className ?? 'border-edge-strong bg-surface-2 text-ink-dim')}
      title={title}
    >
      {children}
    </span>
  )
}

export function AttackChip({ type, className }: { type: string; className?: string }) {
  const meta = attackMeta(type)
  return (
    <span
      className={clsx('chip', meta.chip, className)}
      title={`${meta.label} — ${meta.description}`}
    >
      {meta.short}
    </span>
  )
}
