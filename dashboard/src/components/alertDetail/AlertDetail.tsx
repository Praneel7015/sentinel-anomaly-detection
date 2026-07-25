import clsx from 'clsx'
import { useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../../api/client'
import type { AlertDetail as AlertDetailType, Contribution, ScoredEvent, Verdict } from '../../api/types'
import { DETECTOR_LABEL, RISK_COLOR, bandForScore, riskColor } from '../../lib/domain'
import { absolute, relative } from '../../lib/time'
import { AttackChip, Chip } from '../ui/Chip'
import { GRID, ChartTooltip, getChartTheme } from '../ui/ChartKit'
import { Button } from '../ui/Controls'
import { EntityIcon } from '../ui/EntityIcon'
import { Panel } from '../ui/Panel'
import { RiskGauge } from '../ui/Risk'
import { Skeleton } from '../ui/Skeleton'
import { ErrorState } from '../ui/States'

interface AlertDetailProps {
  eventId: string
  detail: AlertDetailType | null
  loading: boolean
  error: Error | null
  onRetry: () => void
  onClose: () => void
  onOpenEntity: (entityId: string) => void
  onVerdict: (eventId: string, verdict: Verdict) => void
  verdict: Verdict | null
}

export function AlertDetail({
  eventId,
  detail,
  loading,
  error,
  onRetry,
  onClose,
  onOpenEntity,
  onVerdict,
  verdict,
}: AlertDetailProps) {
  if (error) return <ErrorState error={error} onRetry={onRetry} />

  return (
    <div className="flex h-full min-h-0 flex-col overflow-y-auto bg-surface-0">
      <DetailHeader
        detail={detail}
        loading={loading}
        onClose={onClose}
        onOpenEntity={onOpenEntity}
        onVerdict={onVerdict}
        verdict={verdict}
        eventId={eventId}
      />
      {loading && !detail ? (
        <DetailSkeleton />
      ) : detail ? (
        <DetailBody detail={detail} onOpenEntity={onOpenEntity} />
      ) : null}
    </div>
  )
}

/* ----------------------------------------------------------------- header */

function DetailHeader({
  detail,
  loading,
  onClose,
  onOpenEntity,
  onVerdict,
  verdict,
  eventId,
}: {
  detail: AlertDetailType | null
  loading: boolean
  onClose: () => void
  onOpenEntity: (id: string) => void
  onVerdict: (id: string, v: Verdict) => void
  verdict: Verdict | null
  eventId: string
}) {
  return (
    <div className="shrink-0 border-b border-edge bg-surface-1 px-4 py-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          {detail ? (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => onOpenEntity(detail.entity_id)}
                  className="flex items-center gap-1.5 font-mono text-sm font-semibold text-ink hover:text-accent transition"
                >
                  <EntityIcon type={detail.entity_type} size={14} />
                  {detail.entity_id}
                </button>
                <AttackChip type={detail.predicted_attack_type} />
                <RiskBandBadge score={detail.risk_score} />
                {detail.cold_start && (
                  <Chip className="border-sky-500/30 bg-sky-500/10 text-sky-600 dark:border-sky-400/30 dark:bg-sky-400/10 dark:text-sky-300">cold-start</Chip>
                )}
                {detail.is_novel && (
                  <Chip className="border-purple-500/30 bg-purple-500/10 text-purple-600 dark:border-purple-400/30 dark:bg-purple-400/10 dark:text-purple-300">novel</Chip>
                )}
              </div>
              <div className="mt-0.5 flex items-center gap-3 text-2xs text-ink-faint">
                <span className="font-mono">{detail.event_id}</span>
                <span>{absolute(detail.timestamp)}</span>
                <span>({relative(detail.timestamp)} ago)</span>
              </div>
            </>
          ) : (
            <div className="space-y-1">
              <Skeleton className="h-5 w-48" />
              <Skeleton className="h-3 w-64" />
            </div>
          )}
        </div>
        <button
          type="button"
          onClick={onClose}
          className="shrink-0 rounded p-1 text-ink-faint transition hover:bg-surface-3 hover:text-ink"
          aria-label="Close detail"
        >
          ✕
        </button>
      </div>

      {detail && (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          {!loading && <TriageButtons eventId={eventId} onVerdict={onVerdict} current={verdict} />}
        </div>
      )}
    </div>
  )
}

function RiskBandBadge({ score }: { score: number }) {
  const band = bandForScore(score)
  const colors: Record<string, string> = {
    low: 'border-risk-low/30 bg-risk-low/10 text-risk-low',
    medium: 'border-risk-medium/30 bg-risk-medium/10 text-risk-medium',
    high: 'border-risk-high/30 bg-risk-high/10 text-risk-high',
    critical: 'border-risk-critical/30 bg-risk-critical/10 text-risk-critical',
  }
  return (
    <Chip className={colors[band]}>
      {score.toFixed(0)} · {band}
    </Chip>
  )
}

function TriageButtons({
  eventId,
  onVerdict,
  current,
}: {
  eventId: string
  onVerdict: (id: string, v: Verdict) => void
  current: Verdict | null
}) {
  const [note, setNote] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const submit = (v: Verdict) => {
    setSubmitting(true)
    onVerdict(eventId, v)
    api
      .sendFeedback({ event_id: eventId, verdict: v, note: note || undefined })
      .catch(() => null)
      .finally(() => setSubmitting(false))
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button
        variant="danger"
        onClick={() => submit('true_positive')}
        disabled={submitting}
        className={clsx(current === 'true_positive' && 'opacity-100 ring-1 ring-risk-critical/50')}
      >
        <kbd className="kbd">1</kbd> True positive
      </Button>
      <Button
        variant="default"
        onClick={() => submit('false_positive')}
        disabled={submitting}
        className={clsx(current === 'false_positive' && 'opacity-100 ring-1 ring-ink-faint/50')}
      >
        <kbd className="kbd">2</kbd> False positive
      </Button>
      <Button
        variant="accent"
        onClick={() => submit('escalate')}
        disabled={submitting}
        className={clsx(current === 'escalate' && 'opacity-100 ring-1 ring-accent/50')}
      >
        <kbd className="kbd">3</kbd> Escalate
      </Button>
      <input
        value={note}
        onChange={(e) => setNote(e.target.value)}
        placeholder="Analyst note (optional)"
        className="h-7 flex-1 min-w-[160px] rounded border border-edge-strong bg-surface-2 px-2 text-xs text-ink placeholder:text-ink-faint focus:border-accent/60 focus:outline-none"
      />
    </div>
  )
}

/* ------------------------------------------------------------------ body */

function DetailBody({
  detail,
  onOpenEntity,
}: {
  detail: AlertDetailType
  onOpenEntity: (id: string) => void
}) {
  const [jsonOpen, setJsonOpen] = useState(false)

  return (
    <div className="flex flex-col gap-4 p-4">
      {/* Risk gauge + warnings row */}
      <div className="flex flex-wrap items-start gap-4">
        <div className="flex flex-col items-center">
          <RiskGauge score={detail.risk_score} size={148} />
          <div className="mt-2 text-center">
            <div className="text-2xs text-ink-faint">Confidence</div>
            <div className="tnum font-mono text-sm font-semibold text-ink">
              {(detail.attack_type_confidence * 100).toFixed(0)}%
            </div>
          </div>
        </div>

        <div className="flex flex-1 min-w-0 flex-col gap-2">
          {(!detail.classifier_agreement || detail.is_novel) && (
            <div className="rounded border border-risk-medium/40 bg-risk-medium/10 px-3 py-2 text-xs text-risk-medium">
              {detail.is_novel
                ? '⚠ Unknown / novel pattern — classifier and signature matcher disagree. Do not rely on the predicted attack type.'
                : '⚠ Classifier agreement: low — the ML model and rule-based matcher assigned different attack types. Review the contributions carefully.'}
            </div>
          )}
          {detail.cold_start && (
            <div className="rounded border border-sky-400/30 bg-sky-400/10 px-3 py-2 text-xs text-sky-300">
              ❄ Cold-start entity ({detail.entity_event_count} events of history). Score is shrunk toward cohort prior — may under- or overstate true risk.
            </div>
          )}
          <EntitySummaryCard summary={detail.entity_summary} onOpenEntity={onOpenEntity} />
        </div>
      </div>

      {/* Detector radar */}
      <Panel title="Detector scores">
        <DetectorRadar scores={detail.detector_scores} />
      </Panel>

      {/* Contribution waterfall */}
      <Panel title="Risk contributions" subtitle="what drove this score" className="border-t-2 border-t-[#EE3124]">
        <ContributionWaterfall contributions={detail.contributions} totalRisk={detail.risk_score} />
      </Panel>

      {/* Narrative */}
      <Panel title="Analyst narrative">
        <p className="text-sm leading-relaxed text-ink-dim">{detail.narrative}</p>
      </Panel>

      {/* Counterfactuals */}
      {detail.counterfactuals.length > 0 && (
        <Panel title="Counterfactuals" subtitle="if these factors were normal…">
          <div className="space-y-2">
            {detail.counterfactuals.map((cf) => (
              <div key={cf.feature} className="flex items-center justify-between gap-3 rounded bg-surface-2 px-3 py-2">
                <div className="min-w-0">
                  <div className="text-xs font-medium text-ink">{cf.display_name}</div>
                </div>
                <div className="shrink-0 text-right">
                  <span className="tnum font-mono text-sm text-ink-dim">
                    {cf.neutralised_risk.toFixed(1)}
                  </span>
                  <span className="tnum ml-1.5 font-mono text-sm text-risk-low">
                    ({cf.delta.toFixed(1)})
                  </span>
                </div>
              </div>
            ))}
          </div>
          <p className="mt-2 text-2xs text-ink-faint">
            Neutralised risk · delta from current score
          </p>
        </Panel>
      )}

      {/* Similar alerts */}
      {detail.similar_alerts.length > 0 && (
        <Panel title="Similar alerts" subtitle={`${detail.similar_alerts.length} related`}>
          <div className="space-y-1">
            {detail.similar_alerts.slice(0, 4).map((a) => (
              <SimilarAlertRow key={a.event_id} alert={a} />
            ))}
          </div>
        </Panel>
      )}

      {/* Raw event JSON */}
      <Panel
        title="Raw event"
        actions={
          <button
            type="button"
            onClick={() => setJsonOpen((v) => !v)}
            className="text-2xs text-ink-faint hover:text-ink-dim transition"
          >
            {jsonOpen ? 'collapse ▲' : 'expand ▼'}
          </button>
        }
      >
        {jsonOpen ? (
          <pre className="max-h-[320px] overflow-auto rounded bg-surface-0 p-3 font-mono text-2xs leading-relaxed text-ink-dim">
            {JSON.stringify(detail.event, null, 2)}
          </pre>
        ) : (
          <p className="text-2xs text-ink-faint">Click expand to view the raw access log event</p>
        )}
      </Panel>
    </div>
  )
}

/* ---------------------------------------------------------------- entity summary */

function EntitySummaryCard({
  summary,
  onOpenEntity,
}: {
  summary: AlertDetailType['entity_summary']
  onOpenEntity: (id: string) => void
}) {
  return (
    <div className="rounded border border-edge bg-surface-1 p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-2xs font-semibold uppercase tracking-[0.12em] text-ink-faint">
          Entity summary
        </span>
        <button
          type="button"
          onClick={() => onOpenEntity(summary.entity_id)}
          className="text-2xs text-accent hover:underline"
        >
          View full profile →
        </button>
      </div>
      <div className="grid grid-cols-3 gap-x-4 gap-y-1 text-2xs">
        {[
          ['Events', summary.event_count.toLocaleString()],
          ['Alerts', String(summary.alert_count)],
          ['Cohort', summary.cohort],
          ['Mean risk', summary.mean_risk.toFixed(1)],
          ['Max risk', summary.max_risk.toFixed(1)],
          ['Cold-start', summary.cold_start ? 'Yes' : 'No'],
        ].map(([k, v]) => (
          <div key={k}>
            <span className="text-ink-faint">{k}: </span>
            <span className="font-mono text-ink-dim">{v}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ---------------------------------------------------------------- detector radar */

function DetectorRadar({ scores }: { scores: ScoredEvent['detector_scores'] }) {
  const data = Object.entries(scores)
    .filter(([, v]) => v !== null)
    .map(([key, value]) => ({
      subject: DETECTOR_LABEL[key] ?? key,
      value: Math.round((value as number) * 100),
      fullMark: 100,
    }))

  const ct = getChartTheme()

  return (
    <div className="flex flex-col items-center">
      <ResponsiveContainer width="100%" height={200}>
        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
          <PolarGrid stroke={ct.grid} />
          <PolarAngleAxis dataKey="subject" tick={{ fontSize: 10, fill: ct.text }} />
          <Radar
            name="Score"
            dataKey="value"
            stroke="rgb(var(--accent))"
            fill="rgb(var(--accent))"
            fillOpacity={0.18}
            strokeWidth={1.5}
          />
          <Tooltip
            content={
              <ChartTooltip
                valueFormatter={(v) => `${v}`}
              />
            }
          />
        </RadarChart>
      </ResponsiveContainer>
      {scores.gru === null && (
        <p className="text-2xs text-ink-faint">GRU-AE not available (torch absent)</p>
      )}
    </div>
  )
}

/* --------------------------------------------------------------- contribution waterfall */

function ContributionWaterfall({
  contributions,
  totalRisk,
}: {
  contributions: Contribution[]
  totalRisk: number
}) {
  const sorted = [...contributions].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, 10)

  const maxAbs = Math.max(...sorted.map((c) => Math.abs(c.contribution)), 1)

  return (
    <div className="space-y-1">
      {sorted.map((c) => {
        const isPositive = c.contribution >= 0
        const pct = (Math.abs(c.contribution) / maxAbs) * 100
        const color = isPositive ? RISK_COLOR[bandForScore(totalRisk)] : '#22c55e'
        return (
          <div
            key={c.feature}
            className="group relative rounded px-2 py-1.5 hover:bg-surface-2 transition"
            title={c.description}
          >
            <div className="flex items-center justify-between gap-3 text-2xs">
              <span className="min-w-0 truncate text-ink-dim">{c.display_name}</span>
              <div className="flex shrink-0 items-center gap-2">
                <span className="font-mono text-ink-faint">{c.display_value}</span>
                <span
                  className={clsx(
                    'tnum w-12 text-right font-mono font-semibold',
                    isPositive ? 'text-risk-high' : 'text-risk-low',
                  )}
                >
                  {isPositive ? '+' : ''}
                  {c.contribution.toFixed(1)}
                </span>
              </div>
            </div>
            <div className="mt-1 flex h-1.5 overflow-hidden rounded-sm bg-surface-3">
              <div
                className="h-full rounded-sm transition-[width]"
                style={{
                  width: `${pct}%`,
                  backgroundColor: color,
                  opacity: 0.85,
                  marginLeft: isPositive ? 0 : 'auto',
                }}
              />
            </div>
            <div className="absolute inset-x-2 top-full z-10 hidden rounded border border-edge-strong bg-surface-0/95 px-2.5 py-2 text-2xs text-ink-faint shadow-xl group-hover:block">
              {c.description}
            </div>
          </div>
        )
      })}
      <p className="mt-2 text-2xs text-ink-faint">
        Total risk:{' '}
        <span className="tnum font-mono font-semibold" style={{ color: riskColor(totalRisk) }}>
          {totalRisk.toFixed(1)}
        </span>
        &nbsp;·&nbsp; hover rows for detailed description
      </p>
    </div>
  )
}

/* -------------------------------------------------------------- similar alert row */

function SimilarAlertRow({ alert }: { alert: ScoredEvent }) {
  return (
    <div className="flex items-center gap-3 rounded px-2 py-1.5 text-2xs hover:bg-surface-2 transition">
      <span
        className="tnum w-8 shrink-0 text-right font-mono font-semibold"
        style={{ color: riskColor(alert.risk_score) }}
      >
        {alert.risk_score.toFixed(0)}
      </span>
      <AttackChip type={alert.predicted_attack_type} />
      <span className="flex min-w-0 items-center gap-1.5">
        <EntityIcon type={alert.entity_type} size={11} />
        <span className="truncate font-mono text-ink-dim">{alert.entity_id}</span>
      </span>
      <span className="ml-auto shrink-0 text-ink-faint">{relative(alert.timestamp)}</span>
    </div>
  )
}

/* ---------------------------------------------------------------- skeleton */

function DetailSkeleton() {
  return (
    <div className="flex flex-col gap-4 p-4">
      <div className="flex items-start gap-4">
        <Skeleton className="h-20 w-36 rounded" />
        <div className="flex-1 space-y-2">
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-3 w-1/2" />
          <Skeleton className="h-3 w-5/6" />
        </div>
      </div>
      <Skeleton className="h-48 w-full rounded" />
      <Skeleton className="h-64 w-full rounded" />
      <Skeleton className="h-24 w-full rounded" />
    </div>
  )
}

/* ---------------------------------------------------------------- contribution bar chart (recharts) */

export function ContributionBarChart({ contributions }: { contributions: Contribution[] }) {
  const data = [...contributions]
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 8)
    .map((c) => ({
      name: c.display_name,
      value: c.contribution,
      color: c.contribution >= 0 ? '#ea580c' : '#16a34a',
    }))

  const ct = getChartTheme()

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} layout="vertical" margin={{ left: 0, right: 40, top: 4, bottom: 4 }}>
        <CartesianGrid {...GRID} stroke={ct.grid} horizontal={false} />
        <XAxis
          type="number"
          domain={['dataMin', 'dataMax']}
          tick={{ fontSize: 10, fill: ct.text }}
          axisLine={{ stroke: ct.axis }}
          tickLine={false}
        />
        <YAxis
          type="category"
          dataKey="name"
          width={140}
          tick={{ fontSize: 10, fill: ct.text }}
          tickLine={false}
          axisLine={false}
        />
        <Tooltip
          content={<ChartTooltip valueFormatter={(v) => `${Number(v).toFixed(2)}`} />}
        />
        <Bar dataKey="value" radius={[0, 2, 2, 0]} maxBarSize={12}>
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} fillOpacity={0.85} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
