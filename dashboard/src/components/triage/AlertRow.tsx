import clsx from 'clsx'
import { memo } from 'react'
import { ArrowUpRight, Check, Snowflake, Sparkles, TriangleAlert, X } from 'lucide-react'
import type { ScoredEvent, Verdict } from '../../api/types'
import { RISK_TEXT_CLASS, bandForScore } from '../../lib/domain'
import { relative } from '../../lib/time'
import { AttackChip } from '../ui/Chip'
import { EntityIcon } from '../ui/EntityIcon'

const VERDICT_MARK: Record<Verdict, { icon: typeof Check; className: string; label: string }> = {
  true_positive: { icon: Check,        className: 'text-risk-critical', label: 'Confirmed true positive' },
  false_positive: { icon: X,           className: 'text-ink-faint',    label: 'Marked false positive' },
  escalate:       { icon: ArrowUpRight, className: 'text-accent',       label: 'Escalated' },
}

/* Risk band → 4px left border color (via inline style for full-color flexibility) */
const RISK_BORDER_COLOR: Record<string, string> = {
  critical: '#EE3124',
  high:     '#D4540A',
  medium:   '#B87B0A',
  low:      '#2D7D46',
  normal:   'transparent',
}

/* Risk band → thin ring around score badge */
const RISK_RING_COLOR: Record<string, string> = {
  critical: '#EE3124',
  high:     '#D4540A',
  medium:   '#B87B0A',
  low:      '#2D7D46',
  normal:   'rgb(var(--edge-strong))',
}

interface AlertRowProps {
  alert: ScoredEvent
  selected: boolean
  isNew: boolean
  verdict?: Verdict
  onSelect: (eventId: string) => void
}

function AlertRowInner({ alert, selected, isNew, verdict, onSelect }: AlertRowProps) {
  const band = bandForScore(alert.risk_score)
  const mark = verdict ? VERDICT_MARK[verdict] : null
  const MarkIcon = mark?.icon

  const borderColor = selected ? '#EE3124' : RISK_BORDER_COLOR[band]

  return (
    <button
      type="button"
      onClick={() => onSelect(alert.event_id)}
      data-event-id={alert.event_id}
      aria-current={selected}
      className={clsx(
        'group relative w-full border-b border-edge/50 py-2.5 pl-3 pr-3 text-left transition-colors',
        'flex min-h-[52px] items-center gap-3',
        selected
          ? 'bg-[#EE3124]/[0.04]'
          : 'hover:bg-surface-2/70',
        isNew && 'animate-slide-in',
        !alert.is_alert && 'opacity-55',
      )}
      style={{ borderLeftWidth: '4px', borderLeftStyle: 'solid', borderLeftColor: borderColor }}
    >
      {/* ── Risk score with thin colored ring ──────────────────── */}
      <span
        className={clsx(
          'tnum shrink-0 flex h-9 w-9 items-center justify-center rounded-sm font-mono text-[17px] font-bold leading-none',
          RISK_TEXT_CLASS[band],
        )}
        style={{ boxShadow: `0 0 0 1.5px ${RISK_RING_COLOR[band]}` }}
      >
        {alert.risk_score.toFixed(0)}
      </span>

      {/* ── Entity + attack type ────────────────────────────────── */}
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <div className="flex min-w-0 items-center gap-2">
          <EntityIcon type={alert.entity_type} />
          <span
            className={clsx(
              'truncate font-mono text-[13px] leading-snug',
              alert.is_alert ? 'text-ink' : 'text-ink-dim',
            )}
          >
            {alert.entity_id}
          </span>
          {!alert.is_alert && (
            <span className="shrink-0 font-mono text-[9px] uppercase tracking-wide text-ink-faint/60">
              below budget
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          <AttackChip type={alert.predicted_attack_type} />
          {/* Flags */}
          <span className="flex items-center gap-1">
            {alert.cold_start && (
              <Snowflake size={10} className="text-sky-400/80" aria-label="Cold-start entity" />
            )}
            {alert.is_novel && (
              <Sparkles size={10} className="text-purple-400" aria-label="Novel pattern" />
            )}
            {!alert.classifier_agreement && !alert.is_novel && (
              <TriangleAlert size={10} className="text-risk-medium" aria-label="Classifier disagreement" />
            )}
          </span>
        </div>
      </div>

      {/* ── Timestamp + verdict ─────────────────────────────────── */}
      <div className="flex shrink-0 flex-col items-end gap-1">
        <span className="tnum font-mono text-[11px] text-ink-faint">
          {relative(alert.timestamp)}
        </span>
        <span className="flex h-4 w-4 items-center justify-center">
          {MarkIcon ? (
            <MarkIcon size={11} className={mark.className} aria-label={mark.label} />
          ) : (
            <span className="h-1 w-1 rounded-full bg-edge-strong" aria-label="Untriaged" />
          )}
        </span>
      </div>
    </button>
  )
}

export const AlertRow = memo(AlertRowInner)
