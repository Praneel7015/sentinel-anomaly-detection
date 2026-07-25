import clsx from 'clsx'
import type { ReactNode } from 'react'

interface PanelProps {
  title?: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
  bodyClassName?: string
  /** Optional accent color for the top border — defaults to edge color */
  accent?: boolean
}

export function Panel({ title, subtitle, actions, children, className, bodyClassName, accent }: PanelProps) {
  return (
    <section
      className={clsx(
        'panel flex min-w-0 flex-col',
        accent && 'border-t-2 border-t-accent',
        className,
      )}
    >
      {(title || actions) && (
        <header className="panel-header shrink-0">
          <div className="flex min-w-0 items-baseline gap-2">
            <h2 className="panel-title truncate">{title}</h2>
            {subtitle && (
              <span className="truncate text-[10px] font-normal text-ink-faint">{subtitle}</span>
            )}
          </div>
          {actions && (
            <div className="flex shrink-0 items-center gap-1.5">{actions}</div>
          )}
        </header>
      )}
      <div className={clsx('min-w-0 flex-1', bodyClassName ?? 'p-4')}>{children}</div>
    </section>
  )
}
