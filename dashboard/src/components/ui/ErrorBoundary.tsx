import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[SENTINEL] Uncaught render error:', error, info)
  }

  reset = () => this.setState({ hasError: false, error: null })

  render() {
    if (!this.state.hasError) return this.props.children

    if (this.props.fallback) return this.props.fallback

    return (
      <div className="flex h-full items-center justify-center bg-surface-0">
        <div className="max-w-md rounded-xl border border-edge bg-surface-1 p-8 text-center shadow-lg">
          <div className="mb-4 text-4xl">⚠</div>
          <h2 className="mb-2 text-lg font-semibold text-ink">Something went wrong</h2>
          <p className="mb-4 text-sm text-ink-dim">
            {this.state.error?.message ?? 'A render error occurred.'}
          </p>
          <button
            type="button"
            onClick={this.reset}
            className="rounded bg-accent px-4 py-2 text-sm font-medium text-white hover:opacity-90"
          >
            Try again
          </button>
        </div>
      </div>
    )
  }
}
