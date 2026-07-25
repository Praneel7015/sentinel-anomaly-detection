import { useCallback, useEffect, useRef, useState } from 'react'

export interface AsyncState<T> {
  data: T | null
  loading: boolean
  error: Error | null
  reload: () => void
}

/**
 * Runs an aborting async loader whenever `deps` change. Keeps the previous data
 * visible while refetching so the layout does not collapse into skeletons on
 * every filter tweak.
 */
export function useAsync<T>(loader: (signal: AbortSignal) => Promise<T>, deps: unknown[]): AsyncState<T> {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)
  const [nonce, setNonce] = useState(0)
  const loaderRef = useRef(loader)
  loaderRef.current = loader

  useEffect(() => {
    const controller = new AbortController()
    let cancelled = false
    setLoading(true)

    loaderRef
      .current(controller.signal)
      .then((value) => {
        if (cancelled) return
        setData(value)
        setError(null)
      })
      .catch((err: unknown) => {
        if (cancelled || (err instanceof DOMException && err.name === 'AbortError')) return
        setError(err instanceof Error ? err : new Error(String(err)))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
      controller.abort()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce])

  const reload = useCallback(() => setNonce((n) => n + 1), [])

  return { data, loading, error, reload }
}
