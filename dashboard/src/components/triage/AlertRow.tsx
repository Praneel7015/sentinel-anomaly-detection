import clsx from 'clsx'
import { memo } from 'react'
import { ArrowUpRight, Check, Snowflake, Sparkles, TriangleAlert, X } from 'lucide-react'
import type { ScoredEvent, Verdict } from '../../api/types'
import { RISK_TEXT_CLASS, bandForScore } from '../../lib/domain'
import { relative } from '../../lib/time'
import { AttackChip } from '../ui/Chip'
import { EntityIcon } from '../ui/EntityIcon'
import { RiskBar } from '../ui/Risk'

const VERDICT_MARK: Record<Verdict, { icon: typeof Check; className: string; label: string }> = {
  true_positive: { icon: Check, className: 'text-risk-critical', label: 'Confirmed true positive' },
  false_positive: { icon: X, className: 'text-ink-faint', label: 'Marked false positive' },
  escalate: { icon: ArrowUpRight, className: 'text-accent', label: 'Escalated' },
}

interface AlertRowProps {
  alert: ScoredEvent
  selected: boolean
  isNew: boolean
  verdict?: Verdict
  onSelect: (eventId: string) => void
}

const RISK_LEFT_BORDER: Record<string, string> = {
  critical: 'border-l-[#EE3124]',
  high: 'border-l-[#ea580c]',
  medium: 'border-l-[#b45309]',
  low: 'border-l-[#16a34a]',
  normal: 'border-l-transparent',
}

function AlertRowInner({ alert, selected, isNew, verdict, onSelect }: AlertRowProps) {
  const band = bandForScore(alert.risk_score)
  const mark = verdict ? VERDICT_MARK[verdict] : null
  const MarkIcon = mark?.icon

  return (
    <button
      type="button"
      onClick={() => onSelect(alert.event_id)}
      data-event-id={alert.event_id}
      aria-current={selected}
      className={clsx(
        'group relative grid w-full items-center gap-2.5 border-b border-edge/60 border-l-4 py-3 pl-3 pr-2 text-left transition-colors min-h-[48px]',
        'grid-cols-[2rem_5rem_4.5rem_minmax(0,1fr)_auto_3rem_1rem]',
        selected ? 'bg-[#EE3124]/5 border-l-[#EE3124]' : RISK_LEFT_BORDER[band],
        !selected && 'hover:bg-surface-2/80',
        isNew && 'animate-slide-in',
        !alert.is_alert && 'opacity-50',
      )}
    >
      <span className={clsx('tnum text-right font-mono text-xl font-bold', RISK_TEXT_CLASS[band])}>
        {alert.risk_score.toFixed(0)}
      </span>

      <RiskBar score={alert.risk_score} />

      <AttackChip type={alert.predicted_attack_type} className="justify-center" />

      <span className="flex min-w-0 items-center gap-1.5">
        <EntityIcon type={alert.entity_type} />
        <span
          className={clsx(
            'truncate font-mono text-sm',
            alert.is_alert ? 'text-ink' : 'text-ink-dim',
          )}
        >
          {alert.entity_id}
        </span>
        {!alert.is_alert && (
          <span className="shrink-0 text-[0.625rem] uppercase tracking-wide text-ink-faint/70">below budget</span>
        )}
      </span>

      <span className="flex shrink-0 items-center gap-1">
        {alert.cold_start && (
          <Snowflake size={11} className="text-sky-400/80" aria-label="Cold-start entity: provisional score" />
        )}
        {alert.is_novel && (
          <Sparkles size={11} className="text-purple-400" aria-label="Novel or unknown pattern" />
        )}
        {!alert.classifier_agreement && !alert.is_novel && (
          <TriangleAlert size={11} className="text-risk-medium" aria-label="Classifier disagreement" />
        )}
      </span>

      <span className="tnum text-right font-mono text-xs text-ink-faint">{relative(alert.timestamp)}</span>

      <span className="flex justify-center">
        {MarkIcon ? (
          <MarkIcon size={11} className={mark.className} aria-label={mark.label} />
        ) : (
          <span className="h-1 w-1 rounded-full bg-surface-4" aria-label="Untriaged" />
        )}
      </span>
    </button>
  )
}

export const AlertRow = memo(AlertRowInner)
