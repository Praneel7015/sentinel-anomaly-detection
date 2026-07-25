import clsx from 'clsx'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { thresholdForBudget } from '../../api/mockDataset'
import type { AlertSort, ScoredEvent, Verdict } from '../../api/types'
import { useAlertDetail, useAlerts } from '../../hooks/useAlerts'
import { useStream } from '../../hooks/useStream'
import { AlertDetail } from '../alertDetail/AlertDetail'
import { AlertFilters } from './AlertFilters'
import type { TriageFilters } from './AlertFilters'
import { AlertQueue } from './AlertQueue'
import { StreamControls } from './StreamControls'

interface TriageViewProps {
  budgetPct: number
  onBudgetChange: (pct: number) => void
  onOpenEntity: (entityId: string) => void
  focusEventId: string | null
  onFocusConsumed: () => void
}

const DEFAULT_FILTERS: TriageFilters = {
  search: '',
  attackType: 'all',
  entityType: 'all',
  minRisk: 0,
  sort: 'risk_desc',
}

export function TriageView({
  budgetPct,
  onBudgetChange,
  onOpenEntity,
  focusEventId,
  onFocusConsumed,
}: TriageViewProps) {
  const [filters, setFilters] = useState<TriageFilters>(DEFAULT_FILTERS)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [verdicts, setVerdicts] = useState<Record<string, Verdict>>({})
  const [newIds, setNewIds] = useState<Set<string>>(new Set())
  const [heldAlerts, setHeldAlerts] = useState<ScoredEvent[]>([])
  const [follow, setFollow] = useState(true)
  const [sessionReceived, setSessionReceived] = useState(0)
  const { pending, consumePending } = useStream()
  const prevPendingRef = useRef<Set<string>>(new Set())

  const query = useMemo(
    () => ({
      limit: 250,
      offset: 0,
      min_risk: filters.minRisk || undefined,
      attack_type: filters.attackType !== 'all' ? filters.attackType : undefined,
      budget_pct: budgetPct,
      sort: filters.sort as AlertSort,
    }),
    [filters.minRisk, filters.attackType, budgetPct, filters.sort],
  )

  const { data, loading, error, reload } = useAlerts(query)

  const threshold = useMemo(() => thresholdForBudget(budgetPct), [budgetPct])

  const filteredAlerts = useMemo(() => {
    const alerts = data?.alerts ?? []
    const q = filters.search.toLowerCase()
    if (!q && filters.entityType === 'all') return alerts
    return alerts.filter((a) => {
      if (filters.entityType !== 'all' && a.entity_type !== filters.entityType) return false
      if (q) {
        const raw = String(a.event.source_ip ?? '')
        const resource = String(a.event.resource ?? '')
        if (
          !a.entity_id.toLowerCase().includes(q) &&
          !a.event_id.toLowerCase().includes(q) &&
          !raw.toLowerCase().includes(q) &&
          !resource.toLowerCase().includes(q) &&
          !a.predicted_attack_type.includes(q)
        )
          return false
      }
      return true
    })
  }, [data, filters.search, filters.entityType])

  useEffect(() => {
    if (focusEventId) {
      setSelectedId(focusEventId)
      onFocusConsumed()
    }
  }, [focusEventId, onFocusConsumed])

  useEffect(() => {
    if (pending.length === 0) return
    const currentIds = new Set(pending.map((e) => e.event_id))
    const brandNew = pending.filter((e) => !prevPendingRef.current.has(e.event_id))
    prevPendingRef.current = currentIds
    if (brandNew.length === 0) return

    setSessionReceived((n) => n + brandNew.length)

    if (follow) {
      const drained = consumePending()
      setNewIds((prev) => {
        const next = new Set(prev)
        for (const e of drained) next.add(e.event_id)
        return next
      })
      setTimeout(() => setNewIds(new Set()), 4000)
      reload()
    } else {
      setHeldAlerts((prev) => [...brandNew, ...prev].slice(0, 60))
    }
  }, [pending, follow, consumePending, reload])

  const handleFlush = useCallback(() => {
    const drained = consumePending()
    setNewIds((prev) => {
      const next = new Set(prev)
      for (const e of drained) next.add(e.event_id)
      return next
    })
    setHeldAlerts([])
    setTimeout(() => setNewIds(new Set()), 4000)
    reload()
    setFollow(true)
  }, [consumePending, reload])

  const handleVerdict = useCallback((eventId: string, verdict: Verdict) => {
    setVerdicts((prev) => ({ ...prev, [eventId]: verdict }))
  }, [])

  useEffect(() => {
    const handle = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'j' || e.key === 'ArrowDown') {
        e.preventDefault()
        setSelectedId((cur) => {
          const idx = filteredAlerts.findIndex((a) => a.event_id === cur)
          const next = filteredAlerts[Math.min(filteredAlerts.length - 1, idx + 1)]
          return next?.event_id ?? cur
        })
      }
      if (e.key === 'k' || e.key === 'ArrowUp') {
        e.preventDefault()
        setSelectedId((cur) => {
          const idx = filteredAlerts.findIndex((a) => a.event_id === cur)
          const prev = filteredAlerts[Math.max(0, idx - 1)]
          return prev?.event_id ?? cur
        })
      }
      if (e.key === 'Escape') setSelectedId(null)
    }
    window.addEventListener('keydown', handle)
    return () => window.removeEventListener('keydown', handle)
  }, [filteredAlerts])

  const alertCount = filteredAlerts.filter((a) => a.risk_score >= threshold).length
  const hasDetail = selectedId !== null

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 items-center justify-between gap-3 border-b border-edge bg-surface-1 px-3 py-1.5">
        <span className="text-2xs text-ink-faint">
          <kbd className="kbd">j</kbd>/<kbd className="kbd">k</kbd> navigate &nbsp;·&nbsp;
          <kbd className="kbd">1</kbd>–<kbd className="kbd">3</kbd> triage &nbsp;·&nbsp;
          <kbd className="kbd">Esc</kbd> close
        </span>
        <StreamControls
          heldCount={heldAlerts.length}
          follow={follow}
          onToggleFollow={() => setFollow((f) => !f)}
          onFlush={handleFlush}
          received={sessionReceived}
        />
      </div>

      <AlertFilters
        filters={filters}
        onChange={setFilters}
        budgetPct={budgetPct}
        onBudgetChange={onBudgetChange}
        alertCount={alertCount}
        totalCount={filteredAlerts.length}
        threshold={threshold}
      />

      <div className="flex min-h-0 flex-1">
        <div
          className={clsx(
            'min-h-0 border-r border-edge transition-[width]',
            hasDetail ? 'w-[42%] shrink-0' : 'flex-1',
          )}
        >
          <AlertQueue
            alerts={filteredAlerts}
            loading={loading}
            error={error}
            onRetry={reload}
            selectedId={selectedId}
            onSelect={setSelectedId}
            newIds={newIds}
            verdicts={verdicts}
            threshold={threshold}
          />
        </div>

        {hasDetail && (
          <div className="min-h-0 flex-1 overflow-hidden">
            <AlertDetailPanel
              eventId={selectedId}
              onClose={() => setSelectedId(null)}
              onOpenEntity={onOpenEntity}
              onVerdict={handleVerdict}
              verdict={verdicts[selectedId] ?? null}
            />
          </div>
        )}

        {!hasDetail && (
          <div className="hidden flex-1 items-center justify-center text-ink-faint lg:flex">
            <div className="text-center">
              <p className="text-sm">Select an alert to inspect</p>
              <p className="mt-1 text-xs">or press <kbd className="kbd">j</kbd>/<kbd className="kbd">k</kbd> to navigate</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function AlertDetailPanel({
  eventId,
  onClose,
  onOpenEntity,
  onVerdict,
  verdict,
}: {
  eventId: string
  onClose: () => void
  onOpenEntity: (id: string) => void
  onVerdict: (eventId: string, verdict: Verdict) => void
  verdict: Verdict | null
}) {
  const { data, loading, error, reload } = useAlertDetail(eventId)

  useEffect(() => {
    const handle = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === '1') onVerdict(eventId, 'true_positive')
      if (e.key === '2') onVerdict(eventId, 'false_positive')
      if (e.key === '3') onVerdict(eventId, 'escalate')
    }
    window.addEventListener('keydown', handle)
    return () => window.removeEventListener('keydown', handle)
  }, [eventId, onVerdict])

  return (
    <AlertDetail
      eventId={eventId}
      detail={data}
      loading={loading}
      error={error}
      onRetry={reload}
      onClose={onClose}
      onOpenEntity={onOpenEntity}
      onVerdict={onVerdict}
      verdict={verdict}
    />
  )
}
