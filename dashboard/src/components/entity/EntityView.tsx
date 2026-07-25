import clsx from 'clsx'
import { RefreshCw, SearchX } from 'lucide-react'
import { useState } from 'react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { useAlerts } from '../../hooks/useAlerts'
import { useEntities } from '../../hooks/useEntities'
import { useEntity } from '../../hooks/useEntity'
import { riskColor } from '../../lib/domain'
import { absolute, dayHour } from '../../lib/time'
import { AttackChip } from '../ui/Chip'
import { AXIS, ChartTooltip, GRID } from '../ui/ChartKit'
import { EntityIcon } from '../ui/EntityIcon'
import { Panel } from '../ui/Panel'
import { RiskScore } from '../ui/Risk'
import { SkeletonChart } from '../ui/Skeleton'
import { EmptyState, ErrorState } from '../ui/States'

interface EntityViewProps {
  entityId: string | null
  onSelectEntity: (id: string | null) => void
  onOpenAlert: (eventId: string) => void
}

/* ---------------------------------------------------------------- sidebar */

function EntitySidebar({
  selectedId,
  onSelect,
}: {
  selectedId: string | null
  onSelect: (id: string) => void
}) {
  const [search, setSearch] = useState('')
  const [sort, setSort] = useState('risk_desc')
  const { data, loading, reload } = useEntities(sort, 200)

  const entities = data?.entities ?? []
  const filtered = search
    ? entities.filter((e) => e.entity_id.toLowerCase().includes(search.toLowerCase()))
    : entities

  return (
    <div className="flex h-full w-72 shrink-0 flex-col border-r border-edge md:w-80">
      <div className="sticky top-0 z-10 border-b border-edge bg-surface-1 p-3 space-y-2">
        <div className="flex items-center gap-2">
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search entity…"
            className="h-9 min-w-0 flex-1 rounded border border-edge-strong bg-surface-2 px-2 text-sm text-ink placeholder:text-ink-faint focus:border-accent/60 focus:outline-none"
          />
          <button
            type="button"
            onClick={reload}
            title="Refresh"
            className="flex h-9 w-9 items-center justify-center rounded border border-edge-strong bg-surface-2 text-ink-faint hover:text-ink transition"
          >
            <RefreshCw size={14} />
          </button>
        </div>
        <select
          value={sort}
          onChange={(e) => setSort(e.target.value)}
          className="h-8 w-full rounded border border-edge-strong bg-surface-2 px-2 text-xs text-ink-dim focus:outline-none"
        >
          <option value="risk_desc">Highest risk first</option>
          <option value="risk_asc">Lowest risk first</option>
          <option value="name_asc">Name A–Z</option>
          <option value="event_desc">Most events first</option>
        </select>
        <div className="text-xs text-ink-faint">
          {loading ? 'Loading…' : `${filtered.length} entities`}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto divide-y divide-edge/60">
        {filtered.length === 0 && !loading && (
          <div className="flex flex-col items-center gap-2 py-10 text-center text-xs text-ink-faint px-4">
            <SearchX size={22} strokeWidth={1.4} />
            <span>No entities yet. Score some events first.</span>
          </div>
        )}
        {filtered.map((entity) => (
          <button
            key={entity.entity_id}
            type="button"
            onClick={() => onSelect(entity.entity_id)}
            className={clsx(
              'flex w-full min-h-[44px] items-center gap-2 px-3 py-2.5 text-left text-xs transition hover:bg-surface-2',
              selectedId === entity.entity_id && 'bg-accent/5 border-l-2 border-l-accent',
            )}
          >
            <EntityIcon type={entity.entity_type} size={12} />
            <span className="min-w-0 flex-1 truncate font-mono text-ink-dim">{entity.entity_id}</span>
            <div className="shrink-0 flex flex-col items-end gap-0.5">
              {entity.mean_risk > 0 && (
                <span
                  className="tnum font-mono font-semibold text-[11px]"
                  style={{ color: riskColor(entity.mean_risk) }}
                >
                  {entity.mean_risk.toFixed(0)}
                </span>
              )}
              {entity.alert_count > 0 && (
                <span className="text-[10px] text-ink-faint">{entity.alert_count} alerts</span>
              )}
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}

/* --------------------------------------------------------- main component */

export function EntityView({ entityId, onSelectEntity, onOpenAlert }: EntityViewProps) {
  const { data, loading, error, reload } = useEntity(entityId)

  if (!entityId) {
    return (
      <div className="flex h-full">
        <EntitySidebar selectedId={null} onSelect={onSelectEntity} />
        <div className="flex flex-1 items-center justify-center">
          <EmptyState title="Select an entity" hint="Choose from the list or search by ID" />
        </div>
      </div>
    )
  }

  if (error) {
    const is404 = (error as { status?: number }).status === 404
    return (
      <div className="flex h-full">
        <EntitySidebar selectedId={entityId} onSelect={onSelectEntity} />
        <div className="flex flex-1 items-center justify-center">
          {is404 ? (
            <EmptyState
              icon={<SearchX size={26} strokeWidth={1.4} />}
              title="Entity not in live store yet"
              hint="This entity will appear once its events flow through the stream."
            />
          ) : (
            <ErrorState error={error} onRetry={reload} />
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full">
      <EntitySidebar selectedId={entityId} onSelect={onSelectEntity} />

      <div className="flex flex-1 min-h-0 flex-col overflow-y-auto">
        {/* header */}
        <div className="shrink-0 border-b border-edge bg-surface-1 px-4 py-3">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => onSelectEntity(null)}
              className="text-2xs text-ink-faint hover:text-ink-dim transition"
            >
              ← Back
            </button>
            {data && (
              <div className="flex flex-wrap items-center gap-2">
                <EntityIcon type={data.entity_type} size={16} />
                <span className="font-mono text-sm font-semibold text-ink">{data.entity_id}</span>
                <span className="chip border-edge-strong bg-surface-2 text-ink-dim">{data.entity_type}</span>
                <span className="chip border-edge-strong bg-surface-2 text-ink-dim">{data.cohort}</span>
                {data.cold_start && (
                  <span className="chip border-sky-400/30 bg-sky-400/10 text-sky-300">❄ cold-start</span>
                )}
                {data.drift_state.drifting && (
                  <span className="chip border-risk-medium/30 bg-risk-medium/10 text-risk-medium">
                    ⚡ drifting
                  </span>
                )}
              </div>
            )}
          </div>
        </div>

        {loading && !data ? (
          <EntitySkeleton />
        ) : data ? (
          <div className="flex flex-col gap-4 p-4 md:p-6">
            {data.drift_state.drifting && (
              <DriftBanner
                detectedAt={data.drift_state.detected_at}
                adapted={data.drift_state.adapted}
              />
            )}

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <Panel title="Profile vs cohort">
                <div className="divide-y divide-edge/60">
                  {data.profile_summary.map((row) => (
                    <div key={row.label} className="flex items-center justify-between gap-3 py-1.5 text-2xs">
                      <span className="text-ink-faint">{row.label}</span>
                      <div className="flex items-center gap-3">
                        <span className="font-mono font-semibold text-ink">{row.value}</span>
                        <span className="text-ink-faint/70">cohort: {row.cohort_value}</span>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-2 grid grid-cols-2 gap-2 text-2xs text-ink-faint">
                  <div>First seen: <span className="font-mono text-ink-dim">{absolute(data.first_seen)}</span></div>
                  <div>Last seen: <span className="font-mono text-ink-dim">{absolute(data.last_seen)}</span></div>
                  <div>Events: <span className="font-mono text-ink-dim">{data.event_count.toLocaleString()}</span></div>
                </div>
              </Panel>

              <Panel title="Peer comparison">
                <PeerRadar axes={data.peer_comparison} />
              </Panel>
            </div>

            <Panel title="Risk over time" subtitle={`${data.risk_timeline.length} events`}>
              <RiskTimeline points={data.risk_timeline} />
            </Panel>

            <Panel title="Activity by hour (24h profile)">
              <ActivityHistogram buckets={data.activity_by_hour} />
            </Panel>

            <Panel title="Top resources" subtitle="new = not in enrolled baseline">
              <TopResources resources={data.top_resources} />
            </Panel>

            <Panel title="Recent alerts for this entity">
              <LiveRecentAlerts entityId={entityId} onOpenAlert={onOpenAlert} />
            </Panel>
          </div>
        ) : null}
      </div>
    </div>
  )
}

function DriftBanner({ detectedAt, adapted }: { detectedAt: string | null; adapted: boolean }) {  return (
    <div className="rounded border border-risk-medium/40 bg-risk-medium/10 px-4 py-3 text-sm text-risk-medium">
      <div className="font-semibold">⚡ Sustained behavioural drift detected</div>
      <div className="mt-1 text-xs text-risk-medium/80">
        {detectedAt
          ? `Page-Hinkley change-point detected ${absolute(detectedAt)}.`
          : 'Drift flagged by statistical change detector.'}{' '}
        {adapted
          ? 'Baseline has been re-baselining — risk should stabilise.'
          : 'Awaiting analyst confirmation before adaptation.'}
      </div>
    </div>
  )
}

function PeerRadar({
  axes,
}: {
  axes: { axis: string; entity: number; cohort_median: number }[]
}) {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <RadarChart cx="50%" cy="50%" outerRadius="68%" data={axes}>
        <PolarGrid stroke="#232936" />
        <PolarAngleAxis dataKey="axis" tick={{ fontSize: 9, fill: '#98a3b5' }} />
        <Radar name="Entity" dataKey="entity" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.18} strokeWidth={1.5} />
        <Radar
          name="Cohort median"
          dataKey="cohort_median"
          stroke="#5d6879"
          fill="#5d6879"
          fillOpacity={0.08}
          strokeWidth={1}
          strokeDasharray="3 2"
        />
        <Tooltip content={<ChartTooltip />} />
      </RadarChart>
    </ResponsiveContainer>
  )
}

function RiskTimeline({
  points,
}: {
  points: { timestamp: string; risk_score: number; is_alert: boolean }[]
}) {
  const data = points.map((p) => ({
    t: dayHour(p.timestamp),
    score: p.risk_score,
    alert: p.is_alert ? p.risk_score : undefined,
  }))

  return (
    <ResponsiveContainer width="100%" height={160}>
      <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.18} />
            <stop offset="95%" stopColor="#22d3ee" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="t" {...AXIS} interval="preserveStartEnd" />
        <YAxis domain={[0, 100]} {...AXIS} width={28} />
        <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" strokeOpacity={0.5} />
        <Tooltip content={<ChartTooltip valueFormatter={(v) => `${Number(v).toFixed(1)}`} />} />
        <Area
          type="monotone"
          dataKey="score"
          stroke="#22d3ee"
          strokeWidth={1.5}
          fill="url(#riskGrad)"
          dot={false}
          activeDot={{ r: 3, strokeWidth: 0 }}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}

function ActivityHistogram({ buckets }: { buckets: number[] }) {
  const data = buckets.map((count, h) => ({ hour: `${String(h).padStart(2, '0')}`, count }))
  const max = Math.max(...buckets, 1)

  return (
    <ResponsiveContainer width="100%" height={120}>
      <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="hour" {...AXIS} />
        <YAxis {...AXIS} width={28} />
        <Tooltip content={<ChartTooltip />} />
        <Bar dataKey="count" maxBarSize={14} radius={[2, 2, 0, 0]}>
          {data.map((entry, i) => (
            <Cell
              key={`bar-${i}`}
              fill={entry.count / max > 0.7 ? '#22d3ee' : entry.count / max > 0.4 ? '#0e7490' : '#1a1f29'}
              fillOpacity={0.9}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function TopResources({
  resources,
}: {
  resources: { resource: string; count: number; is_new: boolean }[]
}) {
  const max = Math.max(...resources.map((r) => r.count), 1)
  return (
    <div className="space-y-1.5">
      {resources.map((r) => (
        <div key={r.resource} className="flex items-center gap-2 text-2xs">
          <span
            className={clsx(
              'min-w-0 flex-1 truncate font-mono',
              r.is_new ? 'text-risk-medium' : 'text-ink-dim',
            )}
          >
            {r.resource}
          </span>
          {r.is_new && (
            <span className="shrink-0 text-[0.625rem] font-semibold uppercase tracking-wider text-risk-medium">
              new
            </span>
          )}
          <div className="w-24 shrink-0 overflow-hidden rounded-sm bg-surface-3 h-1.5">
            <div
              className={clsx('h-full rounded-sm', r.is_new ? 'bg-risk-medium' : 'bg-accent/60')}
              style={{ width: `${(r.count / max) * 100}%` }}
            />
          </div>
          <span className="tnum w-8 text-right font-mono text-ink-faint">{r.count}</span>
        </div>
      ))}
    </div>
  )
}

function LiveRecentAlerts({
  entityId,
  onOpenAlert,
}: {
  entityId: string
  onOpenAlert: (id: string) => void
}) {
  const { data, loading } = useAlerts({ entity_id: entityId, limit: 8, sort: 'risk_desc' })
  const alerts = data?.alerts.filter((e) => e.is_alert) ?? []

  if (loading) return <SkeletonChart className="h-24" />
  if (alerts.length === 0) {
    return <p className="text-2xs text-ink-faint">No alerts for this entity in the current window</p>
  }

  return (
    <div className="space-y-1">
      {alerts.map((a) => (
        <button
          key={a.event_id}
          type="button"
          onClick={() => onOpenAlert(a.event_id)}
          className="flex w-full items-center gap-3 rounded px-2 py-2 text-left text-xs transition hover:bg-surface-2 min-h-[44px]"
        >
          <RiskScore score={a.risk_score} className="w-7 text-right" />
          <AttackChip type={a.predicted_attack_type} />
          <span className="min-w-0 flex-1 truncate font-mono text-ink-faint">{a.event_id}</span>
          <span className="shrink-0 text-ink-faint">{absolute(a.timestamp)}</span>
        </button>
      ))}
    </div>
  )
}

function EntitySkeleton() {
  return (
    <div className="flex flex-col gap-4 p-4">
      <div className="grid grid-cols-2 gap-4">
        <div className="panel p-3 space-y-2">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-3 bg-surface-2 rounded animate-pulse" />
          ))}
        </div>
        <div className="panel p-3">
          <SkeletonChart className="h-40" />
        </div>
      </div>
      <div className="panel p-3">
        <SkeletonChart className="h-32" />
      </div>
    </div>
  )
}
