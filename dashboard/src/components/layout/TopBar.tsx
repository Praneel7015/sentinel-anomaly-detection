import clsx from 'clsx'
import { Activity, Cpu, Radar, ShieldAlert, Zap } from 'lucide-react'
import { USE_MOCK } from '../../api/client'
import { useStream } from '../../hooks/useStream'
import { formatCompact, formatDuration } from '../../lib/domain'
import type { ViewId } from '../../views'
import { VIEWS } from '../../views'

function Counter({ icon, label, value, tone }: { icon: React.ReactNode; label: string; value: string; tone?: string }) {
  return (
    <div className="flex items-center gap-2 border-l border-edge px-3 first:border-l-0">
      <span className="text-ink-faint">{icon}</span>
      <div className="leading-tight">
        <div className={clsx('tnum font-mono text-sm font-semibold', tone ?? 'text-ink')}>{value}</div>
        <div className="text-[0.625rem] uppercase tracking-[0.1em] text-ink-faint">{label}</div>
      </div>
    </div>
  )
}

export function TopBar({ view, onNavigate }: { view: ViewId; onNavigate: (view: ViewId) => void }) {
  const { health, stats, connection } = useStream()

  const healthy = health?.status === 'ok'
  const dotClass = healthy ? 'bg-risk-low' : connection === 'error' ? 'bg-risk-critical' : 'bg-risk-medium'

  return (
    <header className="flex h-14 shrink-0 items-center gap-4 border-b border-edge bg-surface-1 pl-4 pr-3">
      <div className="flex items-center gap-2.5">
        <div className="relative flex h-8 w-8 items-center justify-center rounded bg-accent/10 ring-1 ring-inset ring-accent/30">
          <Radar size={17} className="text-accent" />
        </div>
        <div className="leading-none">
          <div className="text-[0.95rem] font-bold tracking-[0.22em] text-ink">SENTINEL</div>
          <div className="mt-0.5 text-[0.625rem] uppercase tracking-[0.14em] text-ink-faint">
            Behavioural anomaly detection
          </div>
        </div>
      </div>

      <nav className="ml-2 flex items-center gap-0.5 rounded border border-edge bg-surface-0 p-0.5">
        {VIEWS.map((v) => (
          <button
            key={v.id}
            type="button"
            onClick={() => onNavigate(v.id)}
            className={clsx(
              'inline-flex items-center gap-1.5 rounded px-2.5 py-1.5 text-xs font-medium transition',
              view === v.id ? 'bg-surface-3 text-ink shadow-sm' : 'text-ink-dim hover:text-ink',
            )}
          >
            <v.icon size={13} />
            {v.label}
          </button>
        ))}
      </nav>

      <div className="ml-auto flex items-center">
        <Counter
          icon={<Activity size={14} />}
          label="events"
          value={stats ? formatCompact(stats.events_processed) : '—'}
        />
        <Counter
          icon={<ShieldAlert size={14} />}
          label="alerts"
          value={stats ? formatCompact(stats.alerts_raised) : '—'}
          tone="text-risk-high"
        />
        <Counter
          icon={<Zap size={14} />}
          label="events/s"
          value={stats ? stats.events_per_sec.toFixed(0) : '—'}
          tone="text-accent"
        />
        <Counter
          icon={<Cpu size={14} />}
          label="uptime"
          value={stats ? formatDuration(stats.uptime_s) : '—'}
        />
      </div>

      <div
        className="flex items-center gap-2 rounded border border-edge bg-surface-0 px-2.5 py-1.5"
        title={
          health
            ? `${health.status} · v${health.version} · model ${health.model_loaded ? 'loaded' : 'missing'} · torch ${health.torch_available ? 'available' : 'absent'}`
            : 'Backend unreachable'
        }
      >
        <span className={clsx('h-2 w-2 rounded-full', dotClass, healthy && 'animate-pulse-dot')} />
        <div className="leading-tight">
          <div className="text-2xs font-semibold tracking-wide text-ink">{USE_MOCK ? 'MOCK' : 'LIVE'}</div>
          <div className="text-[0.625rem] text-ink-faint">{health ? `v${health.version}` : 'offline'}</div>
        </div>
      </div>
    </header>
  )
}
