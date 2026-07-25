import clsx from 'clsx'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { useMetrics } from '../../hooks/useMetrics'
import { formatPct } from '../../lib/domain'
import { AXIS, ChartTooltip, GRID } from '../ui/ChartKit'
import { Slider } from '../ui/Controls'
import { Panel } from '../ui/Panel'
import { SkeletonChart } from '../ui/Skeleton'
import { ErrorState } from '../ui/States'
import { StatTile } from '../ui/StatTile'

interface OpsViewProps {
  budgetPct: number
  onBudgetChange: (pct: number) => void
}

export function OpsView({ budgetPct, onBudgetChange }: OpsViewProps) {
  const { data, loading, error, reload } = useMetrics()

  if (error) return <ErrorState error={error} onRetry={reload} />

  return (
    <div className="h-full overflow-y-auto">
      <div className="flex flex-col gap-4 p-4">
        {/* Top KPI bar */}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatTile
            label="PR-AUC"
            value={data ? data.pr_auc.toFixed(3) : '—'}
            tone={data && data.pr_auc > 0.75 ? 'good' : 'warn'}
            hint="Precision-recall area under curve (event level)"
          />
          <StatTile
            label="ROC-AUC"
            value={data ? data.roc_auc.toFixed(4) : '—'}
            tone="good"
            hint="Receiver operating characteristic AUC (event level)"
          />
          <StatTile
            label="FP rate (confounders)"
            value={data ? formatPct(data.fp_rate_confounders, 1) : '—'}
            tone={data && data.fp_rate_confounders < 0.2 ? 'warn' : 'bad'}
            hint="False-positive rate on benign confounders (travel, device enrolment…)"
          />
          <StatTile
            label="FP rate (insider drift)"
            value={data ? formatPct(data.fp_rate_insider_drift, 1) : '—'}
            tone="good"
            hint="False-positive rate on insider-drift edge case (post-adaptation)"
          />
        </div>

        {/* Latency KPIs */}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {data
            ? (['p50', 'p95', 'p99', 'mean'] as const).map((k) => (
                <StatTile
                  key={k}
                  label={`Latency ${k}`}
                  value={data.latency_ms[k].toFixed(2)}
                  unit="ms"
                  tone="accent"
                  hint={`Scoring latency ${k}`}
                />
              ))
            : Array.from({ length: 4 }, (_, i) => (
                <StatTile key={i} label="—" value="—" />
              ))}
        </div>

        {/* Budget slider + budget curve */}
        <Panel
          title="Alert budget"
          subtitle="precision & recall trade-off"
          actions={
            <div className="flex items-center gap-2">
              <span className="text-2xs text-ink-faint">Budget</span>
              <Slider value={budgetPct} min={0.1} max={5} step={0.05} onChange={onBudgetChange} className="w-28" />
              <span className="tnum w-10 text-right font-mono text-xs font-semibold text-accent">
                {budgetPct.toFixed(2)}%
              </span>
            </div>
          }
        >
          {loading && !data ? (
            <SkeletonChart className="h-52" />
          ) : data ? (
            <BudgetCurve data={data.budget_curve} currentBudget={budgetPct} />
          ) : null}
        </Panel>

        {/* PR curve */}
        <Panel title="Precision-Recall curve" subtitle="event-level">
          {loading && !data ? (
            <SkeletonChart className="h-52" />
          ) : data ? (
            <PrCurve points={data.pr_curve} prAuc={data.pr_auc} />
          ) : null}
        </Panel>

        {/* Per-attack recall */}
        <Panel title="Per-attack recall" subtitle="at 1% alert budget">
          {loading && !data ? (
            <SkeletonChart className="h-40" />
          ) : data ? (
            <PerAttackRecallChart data={data.per_attack_recall} />
          ) : null}
        </Panel>

        {/* Subgroup KPIs */}
        <div className="grid grid-cols-2 gap-4">
          <Panel title="Cold-start subgroup">
            {data ? (
              <SubgroupCard
                label="Cold-start"
                precision={data.cold_start.precision}
                recall={data.cold_start.recall}
                hint="Entities with fewer than 10 events; profile shrunk toward cohort prior."
              />
            ) : (
              <SkeletonChart className="h-16" />
            )}
          </Panel>
          <Panel title="Post-drift subgroup">
            {data ? (
              <SubgroupCard
                label="Post-drift"
                precision={data.post_drift.precision}
                recall={data.post_drift.recall}
                hint="Events scored after Page-Hinkley triggered fast re-baselining."
              />
            ) : (
              <SkeletonChart className="h-16" />
            )}
          </Panel>
        </div>

        {/* MTTD table */}
        <Panel title="Mean time to detect" subtitle="per episode">
          {data ? (
            <MttdTable rows={data.mttd} />
          ) : (
            <SkeletonChart className="h-32" />
          )}
        </Panel>

        {/* Ablation table */}
        <Panel title="Detector ablation" subtitle="PR-AUC by detector stack">
          {data ? (
            <AblationTable rows={data.ablation} />
          ) : (
            <SkeletonChart className="h-40" />
          )}
        </Panel>

        {/* Confusion matrix */}
        <Panel title="Confusion matrix" subtitle="attack-type classifier">
          {data ? (
            <ConfusionMatrix matrix={data.confusion_matrix} />
          ) : (
            <SkeletonChart className="h-64" />
          )}
        </Panel>
      </div>
    </div>
  )
}

/* ----------------------------------------------------------------- budget curve */

function BudgetCurve({
  data,
  currentBudget,
}: {
  data: { budget_pct: number; precision: number; recall: number; alerts: number; analyst_hours: number }[]
  currentBudget: number
}) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
        <CartesianGrid {...GRID} />
        <XAxis
          dataKey="budget_pct"
          {...AXIS}
          tickFormatter={(v: number) => `${v}%`}
          label={{ value: 'Alert budget %', position: 'insideBottom', offset: -4, fill: '#5d6879', fontSize: 10 }}
        />
        <YAxis domain={[0, 1]} {...AXIS} width={32} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
        <ReferenceLine x={currentBudget} stroke="#22d3ee" strokeDasharray="3 3" strokeOpacity={0.7} label={{ value: 'current', fill: '#22d3ee', fontSize: 9 }} />
        <Tooltip
          content={
            <ChartTooltip
              valueFormatter={(v) => `${(Number(v) * 100).toFixed(1)}%`}
            />
          }
        />
        <Line dataKey="precision" stroke="#22d3ee" strokeWidth={2} dot={false} name="Precision" />
        <Line dataKey="recall" stroke="#f97316" strokeWidth={2} dot={false} name="Recall" strokeDasharray="4 2" />
      </LineChart>
    </ResponsiveContainer>
  )
}

/* ----------------------------------------------------------------- PR curve */

function PrCurve({
  points,
  prAuc,
}: {
  points: { recall: number; precision: number }[]
  prAuc: number
}) {
  return (
    <div>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={points} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid {...GRID} />
          <XAxis
            dataKey="recall"
            {...AXIS}
            tickFormatter={(v: number) => v.toFixed(1)}
            label={{ value: 'Recall', position: 'insideBottom', offset: -4, fill: '#5d6879', fontSize: 10 }}
          />
          <YAxis domain={[0, 1]} {...AXIS} width={32} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
          <Tooltip
            content={
              <ChartTooltip
                valueFormatter={(v) => `${(Number(v) * 100).toFixed(1)}%`}
              />
            }
          />
          <Line
            dataKey="precision"
            stroke="#22d3ee"
            strokeWidth={2}
            dot={false}
            name="Precision"
          />
        </LineChart>
      </ResponsiveContainer>
      <p className="mt-1 text-2xs text-ink-faint">
        PR-AUC: <span className="tnum font-mono font-semibold text-accent">{prAuc.toFixed(3)}</span>
      </p>
    </div>
  )
}

/* ----------------------------------------------------------------- per-attack recall */

function PerAttackRecallChart({
  data,
}: {
  data: { attack_type: string; recall: number; support: number; detected: number }[]
}) {
  const sorted = [...data].sort((a, b) => b.recall - a.recall)

  return (
    <div>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart
          data={sorted}
          layout="vertical"
          margin={{ top: 4, right: 8, bottom: 0, left: 0 }}
        >
          <CartesianGrid {...GRID} horizontal={false} />
          <XAxis
            type="number"
            domain={[0, 1]}
            {...AXIS}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
          />
          <YAxis
            type="category"
            dataKey="attack_type"
            width={120}
            tick={{ fontSize: 9, fill: '#98a3b5' }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            content={
              <ChartTooltip
                valueFormatter={(v) => `${(Number(v) * 100).toFixed(1)}%`}
              />
            }
          />
          <Bar dataKey="recall" maxBarSize={14} radius={[0, 2, 2, 0]} name="Recall">
            {sorted.map((entry, i) => (
              <Cell
                key={`cell-${i}`}
                fill={entry.recall > 0.85 ? '#22c55e' : entry.recall > 0.7 ? '#f97316' : '#ef4444'}
                fillOpacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="mt-2 flex flex-wrap gap-3 text-2xs text-ink-faint">
        {sorted.map((r) => (
          <span key={r.attack_type}>
            <span className="text-ink-dim">{r.attack_type}</span>: {r.detected}/{r.support}
          </span>
        ))}
      </div>
    </div>
  )
}

/* ----------------------------------------------------------------- subgroup card */

function SubgroupCard({
  precision,
  recall,
  hint,
}: {
  label?: string
  precision: number
  recall: number
  hint: string
}) {
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="text-2xs text-ink-faint">Precision</div>
          <div className={clsx('tnum font-mono text-xl font-semibold', precision > 0.7 ? 'text-risk-low' : 'text-risk-medium')}>
            {formatPct(precision)}
          </div>
        </div>
        <div>
          <div className="text-2xs text-ink-faint">Recall</div>
          <div className={clsx('tnum font-mono text-xl font-semibold', recall > 0.7 ? 'text-risk-low' : 'text-risk-medium')}>
            {formatPct(recall)}
          </div>
        </div>
      </div>
      <p className="text-2xs leading-relaxed text-ink-faint">{hint}</p>
    </div>
  )
}

/* ----------------------------------------------------------------- MTTD table */

function MttdTable({
  rows,
}: {
  rows: { attack_type: string; mean_events: number; mean_minutes: number }[]
}) {
  return (
    <table className="w-full text-2xs">
      <thead>
        <tr className="border-b border-edge text-ink-faint">
          <th className="py-1.5 text-left font-semibold uppercase tracking-[0.1em]">Attack type</th>
          <th className="py-1.5 text-right font-semibold uppercase tracking-[0.1em]">Mean events</th>
          <th className="py-1.5 text-right font-semibold uppercase tracking-[0.1em]">Mean time</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-edge/60">
        {rows.map((r) => (
          <tr key={r.attack_type} className="hover:bg-surface-2">
            <td className="py-1.5 font-mono text-ink-dim">{r.attack_type}</td>
            <td className="tnum py-1.5 text-right font-mono text-ink">{r.mean_events.toFixed(1)}</td>
            <td
              className={clsx(
                'tnum py-1.5 text-right font-mono font-semibold',
                r.mean_minutes < 5
                  ? 'text-risk-low'
                  : r.mean_minutes < 30
                    ? 'text-risk-medium'
                    : 'text-risk-high',
              )}
            >
              {r.mean_minutes < 60
                ? `${r.mean_minutes.toFixed(1)} min`
                : `${(r.mean_minutes / 60).toFixed(1)} h`}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/* ----------------------------------------------------------------- ablation table */

function AblationTable({
  rows,
}: {
  rows: { variant: string; pr_auc: number; precision_at_1pct: number }[]
}) {
  const baseline = rows[0]?.pr_auc ?? 0

  return (
    <table className="w-full text-2xs">
      <thead>
        <tr className="border-b border-edge text-ink-faint">
          <th className="py-1.5 text-left font-semibold uppercase tracking-[0.1em]">Variant</th>
          <th className="py-1.5 text-right font-semibold uppercase tracking-[0.1em]">PR-AUC</th>
          <th className="py-1.5 text-right font-semibold uppercase tracking-[0.1em]">Prec @1%</th>
          <th className="py-1.5 text-right font-semibold uppercase tracking-[0.1em]">Δ PR-AUC</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-edge/60">
        {rows.map((r, i) => {
          const delta = r.pr_auc - baseline
          const isBaseline = i === 0
          return (
            <tr key={r.variant} className={clsx('hover:bg-surface-2', isBaseline && 'bg-surface-2/40')}>
              <td className={clsx('py-1.5', isBaseline ? 'font-semibold text-accent' : 'font-mono text-ink-dim')}>
                {r.variant}
              </td>
              <td
                className={clsx(
                  'tnum py-1.5 text-right font-mono font-semibold',
                  r.pr_auc === baseline ? 'text-accent' : r.pr_auc > baseline * 0.9 ? 'text-ink' : 'text-risk-high',
                )}
              >
                {r.pr_auc.toFixed(3)}
              </td>
              <td className="tnum py-1.5 text-right font-mono text-ink">{r.precision_at_1pct.toFixed(3)}</td>
              <td
                className={clsx(
                  'tnum py-1.5 text-right font-mono',
                  isBaseline ? 'text-ink-faint' : delta < 0 ? 'text-risk-high' : 'text-risk-low',
                )}
              >
                {isBaseline ? '—' : delta > 0 ? `+${delta.toFixed(3)}` : delta.toFixed(3)}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

/* ----------------------------------------------------------------- confusion matrix */

function ConfusionMatrix({
  matrix,
}: {
  matrix: { labels: string[]; matrix: number[][] }
}) {
  const { labels, matrix: cells } = matrix

  const rowMaxes = cells.map((row) => Math.max(...row, 1))

  const shortLabel = (l: string) => l.replace('_', '\u200b_').slice(0, 10)

  return (
    <div className="overflow-x-auto">
      <table className="text-[0.6rem]">
        <thead>
          <tr>
            <th className="w-20 text-right pr-2 text-ink-faint font-normal">Actual ↓ / Pred →</th>
            {labels.map((l) => (
              <th key={l} className="w-12 text-center font-mono font-semibold text-ink-dim py-1 px-0.5 max-w-[3rem] truncate" title={l}>
                {shortLabel(l)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cells.map((row, ri) => (
            <tr key={labels[ri]}>
              <td className="text-right pr-2 font-mono text-ink-faint py-0.5" title={labels[ri]}>
                {shortLabel(labels[ri])}
              </td>
              {row.map((val, ci) => {
                const isDiag = ri === ci
                const intensity = val / rowMaxes[ri]
                const bg = isDiag
                  ? `rgba(34,211,238,${0.12 + intensity * 0.55})`
                  : intensity > 0.2
                    ? `rgba(239,68,68,${intensity * 0.45})`
                    : `rgba(26,31,41,${0.4 + intensity * 0.3})`
                return (
                  <td
                    key={ci}
                    className="w-12 py-0.5 px-0.5 text-center font-mono"
                    style={{
                      background: bg,
                      color: isDiag ? '#22d3ee' : val > 0 ? '#e6ebf2' : '#5d6879',
                    }}
                  >
                    {val}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-2xs text-ink-faint">Diagonal = correct; off-diagonal = misclassification. Shading is row-normalised.</p>
    </div>
  )
}
