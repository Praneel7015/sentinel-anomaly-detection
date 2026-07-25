import { AlertTriangle, Inbox, RefreshCw } from 'lucide-react'
import type { ReactNode } from 'react'

export function EmptyState({
  icon,
  title,
  hint,
  action,
}: {
  icon?: ReactNode
  title: string
  hint?: string
  action?: ReactNode
}) {
  return (
    <div className="flex h-full min-h-[180px] flex-col items-center justify-center gap-2 px-6 py-10 text-center">
      <div className="text-ink-faint">{icon ?? <Inbox size={26} strokeWidth={1.4} />}</div>
      <p className="text-sm font-medium text-ink-dim">{title}</p>
      {hint && <p className="max-w-sm text-xs leading-relaxed text-ink-faint">{hint}</p>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  )
}

export function ErrorState({ error, onRetry }: { error: Error; onRetry?: () => void }) {
  return (
    <div className="flex h-full min-h-[180px] flex-col items-center justify-center gap-2 px-6 py-10 text-center">
      <AlertTriangle size={26} strokeWidth={1.4} className="text-risk-critical" />
      <p className="text-sm font-medium text-ink">Request failed</p>
      <p className="max-w-md break-words font-mono text-2xs text-ink-faint">{error.message}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-2 inline-flex items-center gap-1.5 rounded border border-edge-strong bg-surface-2 px-2.5 py-1 text-xs text-ink-dim transition hover:border-accent/50 hover:text-ink"
        >
          <RefreshCw size={12} />
          Retry
        </button>
      )}
    </div>
  )
}
