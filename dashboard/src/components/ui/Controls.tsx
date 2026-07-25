import clsx from 'clsx'
import type { ReactNode } from 'react'

export function Field({ label, children, className }: { label: string; children: ReactNode; className?: string }) {
  return (
    <label className={clsx('flex min-w-0 flex-col gap-1', className)}>
      <span className="text-2xs font-semibold uppercase tracking-[0.12em] text-ink-faint">{label}</span>
      {children}
    </label>
  )
}

const inputBase =
  'h-7 rounded border border-edge-strong bg-surface-2 px-2 text-xs text-ink placeholder:text-ink-faint transition focus:border-accent/60 focus:outline-none'

export function Select<T extends string>({
  value,
  onChange,
  options,
  className,
}: {
  value: T
  onChange: (value: T) => void
  options: { value: T; label: string }[]
  className?: string
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value as T)}
      className={clsx(inputBase, 'cursor-pointer appearance-none pr-6', className)}
      style={{
        backgroundImage:
          "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 10 10'><path d='M2 4l3 3 3-3' fill='none' stroke='%2398a3b5' stroke-width='1.4'/></svg>\")",
        backgroundRepeat: 'no-repeat',
        backgroundPosition: 'right 6px center',
      }}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value} className="bg-surface-2">
          {o.label}
        </option>
      ))}
    </select>
  )
}

export function TextInput({
  value,
  onChange,
  placeholder,
  className,
  icon,
}: {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  className?: string
  icon?: ReactNode
}) {
  return (
    <div className={clsx('relative flex min-w-0 items-center', className)}>
      {icon && <span className="pointer-events-none absolute left-2 text-ink-faint">{icon}</span>}
      <input
        type="text"
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        className={clsx(inputBase, 'w-full', icon && 'pl-7')}
      />
    </div>
  )
}

export function Slider({
  value,
  min,
  max,
  step,
  onChange,
  className,
}: {
  value: number
  min: number
  max: number
  step: number
  onChange: (value: number) => void
  className?: string
}) {
  const pct = ((value - min) / (max - min)) * 100
  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className={clsx(
        'h-1.5 cursor-pointer appearance-none rounded-full bg-surface-3 outline-none',
        '[&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:appearance-none',
        '[&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border [&::-webkit-slider-thumb]:border-surface-0',
        '[&::-webkit-slider-thumb]:bg-accent [&::-webkit-slider-thumb]:shadow-[0_0_0_3px_rgba(34,211,238,0.18)]',
        '[&::-moz-range-thumb]:h-3.5 [&::-moz-range-thumb]:w-3.5 [&::-moz-range-thumb]:rounded-full',
        '[&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:bg-accent',
        className,
      )}
      style={{ background: `linear-gradient(to right, #22d3ee 0%, #22d3ee ${pct}%, #232936 ${pct}%, #232936 100%)` }}
    />
  )
}

export function IconButton({
  onClick,
  title,
  active,
  children,
  className,
}: {
  onClick: () => void
  title: string
  active?: boolean
  children: ReactNode
  className?: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      aria-label={title}
      aria-pressed={active}
      className={clsx(
        'inline-flex h-7 w-7 items-center justify-center rounded border transition',
        active
          ? 'border-accent/50 bg-accent/10 text-accent'
          : 'border-edge-strong bg-surface-2 text-ink-dim hover:border-edge-strong hover:text-ink',
        className,
      )}
    >
      {children}
    </button>
  )
}

export function Button({
  onClick,
  children,
  variant = 'default',
  size = 'sm',
  disabled,
  className,
  title,
}: {
  onClick?: () => void
  children: ReactNode
  variant?: 'default' | 'accent' | 'success' | 'danger' | 'warn' | 'ghost'
  size?: 'sm' | 'md'
  disabled?: boolean
  className?: string
  title?: string
}) {
  const variants: Record<string, string> = {
    default: 'border-edge-strong bg-surface-2 text-ink-dim hover:text-ink hover:border-ink-faint',
    accent: 'border-accent/50 bg-accent/10 text-accent hover:bg-accent/20',
    success: 'border-risk-low/50 bg-risk-low/10 text-risk-low hover:bg-risk-low/20',
    danger: 'border-risk-critical/50 bg-risk-critical/10 text-risk-critical hover:bg-risk-critical/20',
    warn: 'border-risk-high/50 bg-risk-high/10 text-risk-high hover:bg-risk-high/20',
    ghost: 'border-transparent bg-transparent text-ink-faint hover:text-ink',
  }
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={clsx(
        'inline-flex items-center justify-center gap-1.5 rounded border font-medium transition disabled:cursor-not-allowed disabled:opacity-40',
        size === 'sm' ? 'h-7 px-2.5 text-xs' : 'h-8 px-3 text-sm',
        variants[variant],
        className,
      )}
    >
      {children}
    </button>
  )
}
