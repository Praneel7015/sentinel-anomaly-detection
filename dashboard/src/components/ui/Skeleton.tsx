import clsx from 'clsx'
import type { CSSProperties } from 'react'

export function Skeleton({ className, style }: { className?: string; style?: CSSProperties }) {
  return (
    <div className={clsx('relative overflow-hidden rounded bg-surface-2', className)} style={style}>
      <div className="absolute inset-0 -translate-x-full animate-shimmer bg-gradient-to-r from-transparent via-white/[0.045] to-transparent" />
    </div>
  )
}

export function SkeletonRows({ rows = 8, className }: { rows?: number; className?: string }) {
  return (
    <div className={clsx('space-y-px', className)}>
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="flex items-center gap-3 px-3 py-2.5">
          <Skeleton className="h-7 w-9" />
          <Skeleton className="h-1.5 flex-1" />
          <Skeleton className="h-4 w-16" />
          <Skeleton className="h-4 w-28" />
          <Skeleton className="h-4 w-10" />
        </div>
      ))}
    </div>
  )
}

const BAR_HEIGHTS = ['30%', '55%', '42%', '78%', '61%', '88%', '47%', '70%', '35%', '92%', '58%', '44%', '81%', '52%']

export function SkeletonChart({ className }: { className?: string }) {
  return (
    <div className={clsx('flex items-end gap-1.5', className)}>
      {BAR_HEIGHTS.map((height, i) => (
        <Skeleton key={i} className="flex-1" style={{ height }} />
      ))}
    </div>
  )
}
