import clsx from 'clsx'
import { useEffect, useRef } from 'react'
import { SearchX } from 'lucide-react'
import type { ScoredEvent, Verdict } from '../../api/types'
import { SkeletonRows } from '../ui/Skeleton'
import { EmptyState, ErrorState } from '../ui/States'
import { AlertRow } from './AlertRow'

interface AlertQueueProps {
  alerts: ScoredEvent[]
  loading: boolean
  error: Error | null
  onRetry: () => void
  selectedId: string | null
  onSelect: (eventId: string) => void
  newIds: Set<string>
  verdicts: Record<string, Verdict>
  threshold: number
}

/**
 * The ranked queue. Rows above the budget threshold are alerts; everything below
 * is shown greyed so the analyst can see exactly what the budget is excluding.
 */
export function AlertQueue({
  alerts,
  loading,
  error,
  onRetry,
  selectedId,
  onSelect,
  newIds,
  verdicts,
  threshold,
}: AlertQueueProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  // Keep the keyboard-driven selection inside the viewport.
  useEffect(() => {
    if (!selectedId || !scrollRef.current) return
    const node = scrollRef.current.querySelector<HTMLElement>(`[data-event-id="${CSS.escape(selectedId)}"]`)
    node?.scrollIntoView({ block: 'nearest' })
  }, [selectedId])

  if (error) return <ErrorState error={error} onRetry={onRetry} />
  if (loading && alerts.length === 0) return <SkeletonRows rows={14} />
  if (alerts.length === 0) {
    return (
      <EmptyState
        icon={<SearchX size={26} strokeWidth={1.4} />}
        title="No events match these filters"
        hint="Widen the risk floor, clear the attack-type filter, or raise the alert budget to bring more of the ranked stream into view."
      />
    )
  }

  const firstBelow = alerts.findIndex((a) => a.risk_score < threshold)

  return (
    <div ref={scrollRef} className="h-full overflow-y-auto">
      {alerts.map((alert, index) => (
        <div key={alert.event_id}>
          {index === firstBelow && index > 0 && <BudgetDivider threshold={threshold} count={index} />}
          <AlertRow
            alert={alert}
            selected={alert.event_id === selectedId}
            isNew={newIds.has(alert.event_id)}
            verdict={verdicts[alert.event_id]}
            onSelect={onSelect}
          />
        </div>
      ))}
    </div>
  )
}

function BudgetDivider({ threshold, count }: { threshold: number; count: number }) {
  return (
    <div className={clsx('flex items-center gap-2 bg-surface-0/60 px-3 py-1')}>
      <span className="h-px flex-1 bg-gradient-to-r from-transparent via-accent/40 to-accent/40" />
      <span className="shrink-0 text-[0.625rem] font-semibold uppercase tracking-[0.14em] text-accent/80">
        alert budget line · {count} alerts · risk ≥ {threshold.toFixed(1)}
      </span>
      <span className="h-px flex-1 bg-gradient-to-l from-transparent via-accent/40 to-accent/40" />
    </div>
  )
}
