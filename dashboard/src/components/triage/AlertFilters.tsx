import { Gauge, Search } from 'lucide-react'
import type { AlertSort } from '../../api/types'
import { ATTACK_TYPES, ENTITY_TYPES, ENTITY_TYPE_LABEL, attackMeta } from '../../lib/domain'
import { Select, Slider, TextInput } from '../ui/Controls'

export interface TriageFilters {
  search: string
  attackType: string
  entityType: string
  minRisk: number
  sort: AlertSort
}

const SORTS: { value: AlertSort; label: string }[] = [
  { value: 'risk_desc', label: 'Risk ↓' },
  { value: 'risk_asc', label: 'Risk ↑' },
  { value: 'time_desc', label: 'Newest' },
  { value: 'time_asc', label: 'Oldest' },
]

interface AlertFiltersProps {
  filters: TriageFilters
  onChange: (filters: TriageFilters) => void
  budgetPct: number
  onBudgetChange: (pct: number) => void
  alertCount: number
  totalCount: number
  threshold: number
}

export function AlertFilters({
  filters,
  onChange,
  budgetPct,
  onBudgetChange,
  alertCount,
  totalCount,
  threshold,
}: AlertFiltersProps) {
  const set = <K extends keyof TriageFilters>(key: K, value: TriageFilters[K]) =>
    onChange({ ...filters, [key]: value })

  return (
    <div className="shrink-0 border-b border-edge bg-surface-1">
      <div className="flex flex-wrap items-center gap-1.5 px-3 py-2">
        <TextInput
          value={filters.search}
          onChange={(v) => set('search', v)}
          placeholder="Search entity, resource, IP, event id…"
          icon={<Search size={12} />}
          className="min-w-[150px] flex-1"
        />
        <Select
          value={filters.attackType}
          onChange={(v) => set('attackType', v)}
          options={[
            { value: 'all', label: 'All types' },
            { value: 'normal', label: 'Normal' },
            { value: 'unknown_novel', label: 'Unknown / novel' },
            ...ATTACK_TYPES.map((t) => ({ value: t as string, label: attackMeta(t).label })),
          ]}
        />
        <Select
          value={filters.entityType}
          onChange={(v) => set('entityType', v)}
          options={[
            { value: 'all', label: 'All entities' },
            ...ENTITY_TYPES.map((t) => ({ value: t as string, label: ENTITY_TYPE_LABEL[t] })),
          ]}
        />
        <Select value={filters.sort} onChange={(v) => set('sort', v)} options={SORTS} />
      </div>

      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-edge/60 px-3 py-2">
        <div className="flex min-w-[190px] flex-1 items-center gap-2">
          <span className="shrink-0 text-[0.65rem] font-semibold uppercase tracking-[0.18em] text-ink-faint">
            Min risk
          </span>
          <Slider value={filters.minRisk} min={0} max={95} step={1} onChange={(v) => set('minRisk', v)} className="flex-1" />
          <span className="tnum w-6 shrink-0 text-right font-mono text-xs text-ink-dim">{filters.minRisk}</span>
        </div>

        <div className="flex min-w-[260px] flex-[1.4] items-center gap-2">
          <span className="flex shrink-0 items-center gap-1 text-[0.65rem] font-semibold uppercase tracking-[0.18em] text-[#EE3124]">
            <Gauge size={11} />
            Alert budget
          </span>
          <Slider value={budgetPct} min={0.1} max={5} step={0.05} onChange={onBudgetChange} className="flex-1" />
          <span className="tnum w-11 shrink-0 text-right font-mono text-xs font-semibold text-[#EE3124]">
            {budgetPct.toFixed(2)}%
          </span>
        </div>

        <div className="flex shrink-0 items-center gap-2 text-2xs text-ink-faint">
          <span className="inline-flex items-center gap-1 rounded bg-[#EE3124] px-2 py-0.5 text-[0.65rem] font-semibold text-white">
            <span className="tnum font-mono">{alertCount}</span> alerts · threshold <span className="tnum font-mono">{threshold.toFixed(1)}</span>
          </span>
          <span className="text-ink-faint/50">/</span>
          <span>
            <span className="tnum font-mono text-ink-dim">{totalCount}</span> ranked
          </span>
        </div>
      </div>
    </div>
  )
}
