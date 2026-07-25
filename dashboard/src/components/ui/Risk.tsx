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

export function RiskScore({ score, className }: { score: number; className?: string }) {
  const band = bandForScore(score)
  return (
    <span className={clsx('tnum font-mono font-semibold', RISK_TEXT_CLASS[band], className)}>
      {score.toFixed(0)}
    </span>
  )
}

/**
 * Semicircular gauge for the alert detail header. Drawn as an SVG arc so the
 * risk ramp reads at a glance without a chart library.
 */
export function RiskGauge({ score, size = 168 }: { score: number; size?: number }) {
  const band = bandForScore(score)
  const color = riskColor(score)
  const stroke = 12
  const radius = (size - stroke) / 2
  const cx = size / 2
  const cy = size / 2
  const circumference = Math.PI * radius
  const filled = (Math.min(100, Math.max(0, score)) / 100) * circumference

  const arc = `M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`

  return (
    <div className="relative flex flex-col items-center" style={{ width: size, height: size / 2 + 26 }}>
      <svg width={size} height={size / 2 + 6} viewBox={`0 0 ${size} ${size / 2 + 6}`} role="img" aria-label={`Risk score ${score}`}>
        <path d={arc} fill="none" stroke="#1a1f29" strokeWidth={stroke} strokeLinecap="round" />
        <path
          d={arc}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${filled} ${circumference}`}
          style={{ transition: 'stroke-dasharray 600ms ease-out, stroke 300ms' }}
        />
        {[0, 40, 65, 85].map((mark) => {
          const angle = Math.PI - (mark / 100) * Math.PI
          const inner = radius - stroke / 2 - 3
          const outer = radius + stroke / 2 + 3
          return (
            <line
              key={mark}
              x1={cx + Math.cos(angle) * inner}
              y1={cy - Math.sin(angle) * inner}
              x2={cx + Math.cos(angle) * outer}
              y2={cy - Math.sin(angle) * outer}
              stroke="#0a0c10"
              strokeWidth={2}
            />
          )
        })}
      </svg>
      <div className="pointer-events-none absolute inset-x-0 bottom-0 flex flex-col items-center">
        <span className={clsx('tnum font-mono text-4xl font-bold leading-none', RISK_TEXT_CLASS[band])}>
          {score.toFixed(0)}
        </span>
        <span className="mt-1 text-2xs font-semibold uppercase tracking-[0.18em] text-ink-faint">
          {band} risk
        </span>
      </div>
    </div>
  )
}
