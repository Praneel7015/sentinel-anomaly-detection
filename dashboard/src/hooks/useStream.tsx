import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { api } from '../api/client'
import type { HealthResponse, ScoredEvent, StatsResponse } from '../api/types'

export type ConnectionState = 'connecting' | 'open' | 'paused' | 'error'

interface StreamContextValue {
  live: boolean
  setLive: (live: boolean) => void
  speed: number
  setSpeed: (speed: number) => void
  connection: ConnectionState
  stats: StatsResponse | null
  health: HealthResponse | null
  /** Alerts received since the queue last consumed them. */
  pending: ScoredEvent[]
  /** Rolling buffer of everything seen this session, newest first. */
  recent: ScoredEvent[]
  consumePending: () => ScoredEvent[]
  reset: () => void
  lastHeartbeat: number | null
}

const StreamContext = createContext<StreamContextValue | null>(null)

const MAX_RECENT = 400

export function StreamProvider({ children }: { children: ReactNode }) {
  const [live, setLiveState] = useState(true)
  const [speed, setSpeedState] = useState(1)
  const [connection, setConnection] = useState<ConnectionState>('connecting')
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [recent, setRecent] = useState<ScoredEvent[]>([])
  const [lastHeartbeat, setLastHeartbeat] = useState<number | null>(null)
  const pendingRef = useRef<ScoredEvent[]>([])
  const [pending, setPending] = useState<ScoredEvent[]>([])

  useEffect(() => {
    let cancelled = false
    api
      .getHealth()
      .then((h) => !cancelled && setHealth(h))
      .catch(() => !cancelled && setHealth(null))
    api
      .getStats()
      .then((s) => !cancelled && setStats(s))
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!live) {
      setConnection('paused')
      return
    }
    setConnection('connecting')
    const unsubscribe = api.subscribeStream({
      onOpen: () => setConnection('open'),
      onAlert: (alert) => {
        pendingRef.current = [alert, ...pendingRef.current].slice(0, MAX_RECENT)
        setPending(pendingRef.current)
        setRecent((prev) => [alert, ...prev].slice(0, MAX_RECENT))
      },
      onStats: setStats,
      onHeartbeat: () => setLastHeartbeat(Date.now()),
      onError: () => setConnection('error'),
    })
    return unsubscribe
  }, [live])

  const setLive = useCallback((next: boolean) => {
    setLiveState(next)
    void api.controlStream({ action: next ? 'start' : 'pause' }).catch(() => undefined)
  }, [])

  const setSpeed = useCallback((next: number) => {
    setSpeedState(next)
    void api.controlStream({ action: 'start', speed: next }).catch(() => undefined)
  }, [])

  const consumePending = useCallback(() => {
    const drained = pendingRef.current
    pendingRef.current = []
    setPending([])
    return drained
  }, [])

  const reset = useCallback(() => {
    pendingRef.current = []
    setPending([])
    setRecent([])
    void api.controlStream({ action: 'reset' }).catch(() => undefined)
  }, [])

  const value = useMemo<StreamContextValue>(
    () => ({
      live,
      setLive,
      speed,
      setSpeed,
      connection,
      stats,
      health,
      pending,
      recent,
      consumePending,
      reset,
      lastHeartbeat,
    }),
    [live, setLive, speed, setSpeed, connection, stats, health, pending, recent, consumePending, reset, lastHeartbeat],
  )

  return <StreamContext.Provider value={value}>{children}</StreamContext.Provider>
}

export function useStream(): StreamContextValue {
  const ctx = useContext(StreamContext)
  if (!ctx) throw new Error('useStream must be used inside <StreamProvider>')
  return ctx
}
