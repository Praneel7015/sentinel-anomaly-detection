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
  { value: 'risk_asc',  label: 'Risk ↑' },
  { value: 'time_desc', label: 'Newest' },
  { value: 'time_asc',  label: 'Oldest' },
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
    <div className="shrink-0 border-b border-edge bg-surface-0">
      {/* ── Row 1: search + dropdowns ──────────────────────────── */}
      <div className="flex flex-wrap items-center gap-2 px-4 py-2.5">
        <TextInput
          value={filters.search}
          onChange={(v) => set('search', v)}
          placeholder="Search entity, resource, IP, event id…"
          icon={<Search size={12} />}
          className="min-w-[160px] flex-1"
        />
        <Select
          value={filters.attackType}
          onChange={(v) => set('attackType', v)}
          options={[
            { value: 'all',           label: 'All types' },
            { value: 'normal',        label: 'Normal' },
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
        <Select
          value={filters.sort}
          onChange={(v) => set('sort', v)}
          options={SORTS}
        />
      </div>

      {/* ── Row 2: sliders + counter ────────────────────────────── */}
      <div className="flex flex-col gap-2 border-t border-edge/60 px-4 py-2.5 sm:flex-row sm:flex-wrap sm:items-center sm:gap-x-6 sm:gap-y-1.5">
        {/* Min risk slider */}
        <div className="flex min-w-[180px] flex-1 items-center gap-2.5">
          <span className="shrink-0 font-mono text-[10px] font-semibold uppercase tracking-[0.16em] text-ink-faint">
            Min risk
          </span>
          <Slider
            value={filters.minRisk}
            min={0}
            max={95}
            step={1}
            onChange={(v) => set('minRisk', v)}
            className="flex-1"
          />
          <span className="tnum w-6 shrink-0 text-right font-mono text-[12px] font-semibold text-ink-dim">
            {filters.minRisk}
          </span>
        </div>

        {/* Alert budget slider */}
        <div className="flex min-w-[240px] flex-[1.4] items-center gap-2.5">
          <span className="flex shrink-0 items-center gap-1.5 font-mono text-[10px] font-semibold uppercase tracking-[0.16em] text-[#EE3124]">
            <Gauge size={12} />
            Budget
          </span>
          <Slider
            value={budgetPct}
            min={0.1}
            max={5}
            step={0.05}
            onChange={onBudgetChange}
            className="flex-1"
          />
          <span className="tnum w-12 shrink-0 text-right font-mono text-[12px] font-bold text-[#EE3124]">
            {budgetPct.toFixed(2)}%
          </span>
        </div>

        {/* Alert count badge */}
        <div className="flex shrink-0 items-center gap-2">
          <div className="flex items-center gap-1.5 border border-[#EE3124]/30 bg-[#EE3124]/[0.06] px-2.5 py-1">
            <span className="tnum font-mono text-[13px] font-bold text-[#EE3124]">{alertCount}</span>
            <span className="font-mono text-[10px] text-[#EE3124]/70">
              alerts · ≥{threshold.toFixed(1)}
            </span>
          </div>
          <span className="font-mono text-[10px] text-ink-faint">
            <span className="tnum text-ink-dim">{totalCount}</span> ranked
          </span>
        </div>
      </div>
    </div>
  )
}
