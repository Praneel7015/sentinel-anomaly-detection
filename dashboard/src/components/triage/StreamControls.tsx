import clsx from 'clsx'
import { ArrowDownToLine, Pause, Play, RotateCcw } from 'lucide-react'
import { useStream } from '../../hooks/useStream'
import { IconButton } from '../ui/Controls'

const SPEEDS = [0.5, 1, 2, 4]

interface StreamControlsProps {
  heldCount: number
  follow: boolean
  onToggleFollow: () => void
  onFlush: () => void
  received: number
}

export function StreamControls({ heldCount, follow, onToggleFollow, onFlush, received }: StreamControlsProps) {
  const { live, setLive, speed, setSpeed, connection, reset } = useStream()

  const dot =
    connection === 'open'
      ? 'bg-risk-low animate-pulse-dot'
      : connection === 'connecting'
        ? 'bg-risk-medium animate-pulse-dot'
        : connection === 'error'
          ? 'bg-risk-critical'
          : 'bg-ink-faint'

  return (
    <div className="flex shrink-0 items-center gap-1.5">
      <span className="flex items-center gap-1.5 rounded border border-edge bg-surface-0 px-2 py-1">
        <span className={clsx('h-1.5 w-1.5 rounded-full', dot)} />
        <span className="text-2xs uppercase tracking-wider text-ink-dim">{connection}</span>
      </span>

      <IconButton onClick={() => setLive(!live)} title={live ? 'Pause stream' : 'Resume stream'} active={live}>
        {live ? <Pause size={13} /> : <Play size={13} />}
      </IconButton>

      <div className="flex overflow-hidden rounded border border-edge-strong">
        {SPEEDS.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => setSpeed(s)}
            className={clsx(
              'h-7 px-1.5 text-2xs font-mono transition',
              speed === s ? 'bg-accent/15 text-accent' : 'bg-surface-2 text-ink-faint hover:text-ink-dim',
            )}
          >
            {s}×
          </button>
        ))}
      </div>

      <IconButton onClick={reset} title="Reset stream counters">
        <RotateCcw size={13} />
      </IconButton>

      <button
        type="button"
        onClick={follow && heldCount === 0 ? onToggleFollow : onFlush}
        title={follow ? 'Auto-following the live stream — click to hold' : 'Show held alerts'}
        className={clsx(
          'inline-flex h-7 items-center gap-1.5 rounded border px-2 text-2xs font-medium transition',
          heldCount > 0
            ? 'border-accent bg-accent/15 text-accent'
            : follow
              ? 'border-edge-strong bg-surface-2 text-ink-dim hover:text-ink'
              : 'border-edge-strong bg-surface-2 text-ink-faint hover:text-ink',
        )}
      >
        <ArrowDownToLine size={12} className={heldCount > 0 ? 'animate-pulse-dot' : undefined} />
        {heldCount > 0 ? `${heldCount} new` : follow ? `following · ${received}` : 'held'}
      </button>
    </div>
  )
}
