import { AlertTriangle, Inbox, RefreshCw, SearchX, WifiOff } from 'lucide-react'
import type { ReactNode } from 'react'
import { ApiError } from '../../api/real'

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
  const is404 = error instanceof ApiError && error.status === 404
  const isOffline = error instanceof TypeError && error.message.includes('fetch')

  const icon = is404
    ? <SearchX size={26} strokeWidth={1.4} className="text-ink-faint" />
    : isOffline
    ? <WifiOff size={26} strokeWidth={1.4} className="text-risk-medium" />
    : <AlertTriangle size={26} strokeWidth={1.4} className="text-risk-critical" />

  const title = is404
    ? 'Not found'
    : isOffline
    ? 'Cannot reach the backend'
    : 'Request failed'

  const hint = is404
    ? 'This record doesn\'t exist yet — it will appear once events are streamed through the system.'
    : isOffline
    ? 'Check that the API server is running and reachable.'
    : error.message

  return (
    <div className="flex h-full min-h-[180px] flex-col items-center justify-center gap-2 px-6 py-10 text-center">
      {icon}
      <p className="text-sm font-medium text-ink">{title}</p>
      <p className="max-w-md text-xs leading-relaxed text-ink-faint">{hint}</p>
      {!is404 && onRetry && (
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
