import clsx from 'clsx'
import { Activity, Cpu, Moon, Radar, ShieldAlert, Sun, Zap } from 'lucide-react'
import { USE_MOCK } from '../../api/client'
import { useStream } from '../../hooks/useStream'
import { formatCompact, formatDuration } from '../../lib/domain'
import type { Theme } from '../../App'
import type { ViewId } from '../../views'
import { VIEWS } from '../../views'

interface StatCounterProps {
  icon: React.ReactNode
  label: string
  value: string
  tone?: string
}

function StatCounter({ icon, label, value, tone }: StatCounterProps) {
  return (
    <div className="flex items-center gap-2 border-l border-edge px-3 first:border-l-0">
      <span className="text-ink-faint">{icon}</span>
      <div className="leading-tight">
        <div className={clsx('tnum font-mono text-[15px] font-bold', tone ?? 'text-ink')}>{value}</div>
        <div className="text-[10px] font-medium uppercase tracking-[0.12em] text-ink-faint">{label}</div>
      </div>
    </div>
  )
}

interface TopBarProps {
  view: ViewId
  onNavigate: (view: ViewId) => void
  theme: Theme
  onToggleTheme: () => void
}

export function TopBar({ view, onNavigate, theme, onToggleTheme }: TopBarProps) {
  const { health, stats, connection } = useStream()

  const healthy = health?.status === 'ok'
  const connectionColor = healthy
    ? 'bg-risk-low'
    : connection === 'error'
      ? 'bg-risk-critical'
      : 'bg-risk-medium'

  return (
    <header className="flex h-14 shrink-0 items-center border-b border-edge bg-surface-1 pl-5 pr-3">
      {/* ── Logo ────────────────────────────────────────────────── */}
      <div className="flex shrink-0 items-center gap-3 pr-6">
        <div className="flex h-8 w-8 items-center justify-center bg-[#EE3124]">
          <Radar size={16} className="text-white" strokeWidth={2.5} />
        </div>
        <div className="leading-none">
          <div className="text-[15px] font-bold tracking-[0.22em] text-ink">SENTINEL</div>
          <div className="mt-0.5 text-[9px] font-medium uppercase tracking-[0.18em] text-ink-faint">
            Behavioural Anomaly Detection
          </div>
        </div>
      </div>

      {/* ── Divider ─────────────────────────────────────────────── */}
      <div className="mr-1 h-6 w-px bg-edge-strong/60" />

      {/* ── Navigation — bottom-border indicator style ───────────── */}
      <nav className="flex h-14 items-stretch">
        {VIEWS.map((v) => {
          const active = view === v.id
          return (
            <button
              key={v.id}
              type="button"
              onClick={() => onNavigate(v.id)}
              className={clsx(
                'nav-tab',
                active ? 'nav-tab--active' : 'nav-tab--inactive',
              )}
            >
              <v.icon size={13} strokeWidth={active ? 2.5 : 2} />
              <span className="hidden text-sm sm:inline">{v.label}</span>
            </button>
          )
        })}
      </nav>

      {/* ── Stats ───────────────────────────────────────────────── */}
      <div className="ml-auto flex items-center">
        <StatCounter
          icon={<Activity size={13} />}
          label="events"
          value={stats ? formatCompact(stats.events_processed) : '—'}
        />
        <StatCounter
          icon={<ShieldAlert size={13} />}
          label="alerts"
          value={stats ? formatCompact(stats.alerts_raised) : '—'}
          tone="text-risk-high"
        />
        <div className="hidden lg:flex">
          <StatCounter
            icon={<Zap size={13} />}
            label="events/s"
            value={stats ? stats.events_per_sec.toFixed(0) : '—'}
            tone="text-accent"
          />
          <StatCounter
            icon={<Cpu size={13} />}
            label="uptime"
            value={stats ? formatDuration(stats.uptime_s) : '—'}
          />
        </div>
      </div>

      {/* ── Connection status ────────────────────────────────────── */}
      <div
        className="ml-3 hidden sm:flex items-center gap-2 border border-edge bg-surface-0 px-2.5 py-1.5"
        title={
          health
            ? `${health.status} · v${health.version} · model ${health.model_loaded ? 'loaded' : 'missing'}`
            : 'Backend unreachable'
        }
      >
        <span
          className={clsx(
            'h-[7px] w-[7px] rounded-full',
            connectionColor,
            healthy && 'animate-pulse-dot',
          )}
        />
        <div className="leading-tight">
          <div className="text-[11px] font-semibold tracking-wide text-ink">
            {USE_MOCK ? 'MOCK' : 'LIVE'}
          </div>
          <div className="text-[9px] text-ink-faint">
            {health ? `v${health.version}` : 'offline'}
          </div>
        </div>
      </div>

      {/* ── Theme toggle ─────────────────────────────────────────── */}
      <button
        type="button"
        onClick={onToggleTheme}
        className="ml-2 flex h-8 w-8 items-center justify-center border border-edge bg-surface-0 text-ink-faint transition-colors hover:bg-surface-2 hover:text-ink"
        title={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
        aria-label="Toggle theme"
      >
        {theme === 'light' ? <Moon size={14} /> : <Sun size={14} />}
      </button>
    </header>
  )
}
