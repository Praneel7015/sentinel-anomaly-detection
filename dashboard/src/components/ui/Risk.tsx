import clsx from 'clsx'
import { RISK_BAR_CLASS, RISK_TEXT_CLASS, bandForScore, riskColor } from '../../lib/domain'

/** Thin horizontal risk bar used in dense list rows. */
export function RiskBar({ score, className }: { score: number; className?: string }) {
  const band = bandForScore(score)
  return (
    <div className={clsx('h-1.5 w-full overflow-hidden rounded-sm bg-surface-3', className)}>
      <div
        className={clsx('h-full rounded-sm transition-[width] duration-500', RISK_BAR_CLASS[band])}
        style={{ width: `${Math.max(2, score)}%` }}
      />
    </div>
  )
}

/** Inline risk score — tabular mono with band color. */
export function RiskScore({ score, className }: { score: number; className?: string }) {
  const band = bandForScore(score)
  return (
    <span className={clsx('tnum font-mono font-bold', RISK_TEXT_CLASS[band], className)}>
      {score.toFixed(0)}
    </span>
  )
}

/**
 * Semicircular gauge for alert detail header.
 * SVG arc with thin colored ring around the score number.
 */
export function RiskGauge({ score, size = 180 }: { score: number; size?: number }) {
  const band = bandForScore(score)
  const color = riskColor(score)
  const stroke = 10
  const radius = (size - stroke) / 2
  const cx = size / 2
  const cy = size / 2
  const circumference = Math.PI * radius
  const filled = (Math.min(100, Math.max(0, score)) / 100) * circumference

  const arc = `M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`

  /* Threshold tick positions */
  const TICKS = [0, 40, 65, 85]

  return (
    <div className="relative flex flex-col items-center" style={{ width: size, height: size / 2 + 30 }}>
      <svg
        width={size}
        height={size / 2 + 8}
        viewBox={`0 0 ${size} ${size / 2 + 8}`}
        role="img"
        aria-label={`Risk score ${score}`}
      >
        {/* Track */}
        <path
          d={arc}
          fill="none"
          stroke="rgb(var(--surface-3))"
          strokeWidth={stroke}
          strokeLinecap="butt"
        />
        {/* Filled arc */}
        <path
          d={arc}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="butt"
          strokeDasharray={`${filled} ${circumference}`}
          style={{ transition: 'stroke-dasharray 600ms ease-out, stroke 300ms' }}
        />
        {/* Threshold ticks */}
        {TICKS.map((mark) => {
          const angle = Math.PI - (mark / 100) * Math.PI
          const inner = radius - stroke / 2 - 2
          const outer = radius + stroke / 2 + 2
          return (
            <line
              key={mark}
              x1={cx + Math.cos(angle) * inner}
              y1={cy - Math.sin(angle) * inner}
              x2={cx + Math.cos(angle) * outer}
              y2={cy - Math.sin(angle) * outer}
              stroke="rgb(var(--surface-0))"
              strokeWidth={1.5}
            />
          )
        })}
      </svg>

      {/* Score + band label — thin ring around the number */}
      <div className="pointer-events-none absolute inset-x-0 bottom-0 flex flex-col items-center">
        <span
          className={clsx(
            'tnum flex h-12 w-12 items-center justify-center rounded-sm font-mono text-3xl font-bold leading-none',
            RISK_TEXT_CLASS[band],
          )}
          style={{ boxShadow: `0 0 0 1.5px ${color}` }}
        >
          {score.toFixed(0)}
        </span>
        <span className="mt-1.5 font-mono text-[9px] font-semibold uppercase tracking-[0.2em] text-ink-faint">
          {band} risk
        </span>
      </div>
    </div>
  )
}
