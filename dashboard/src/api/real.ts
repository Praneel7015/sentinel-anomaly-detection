/**
 * HTTP/SSE implementation of `SentinelApi` against the FastAPI service.
 * Requests go to `/api/...`; the Vite dev server proxies that to 127.0.0.1:8000.
 */

import type {
  AlertDetail,
  AlertQuery,
  AlertsResponse,
  EntityDetail,
  FeedbackRequest,
  FeedbackResponse,
  HealthResponse,
  MetricsResponse,
  ScoredEvent,
  SentinelApi,
  StatsResponse,
  StreamControlRequest,
  StreamControlResponse,
  StreamHandlers,
  StreamUnsubscribe,
} from './types'

const BASE = '/api'

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { Accept: 'application/json', ...(init?.body ? { 'Content-Type': 'application/json' } : {}) },
    ...init,
  })
  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new ApiError(response.status, text || `${response.status} ${response.statusText}`)
  }
  return (await response.json()) as T
}

function queryString(query: AlertQuery): string {
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined && value !== null && value !== '') params.set(key, String(value))
  }
  const s = params.toString()
  return s ? `?${s}` : ''
}

export const realApi: SentinelApi = {
  mode: 'live',

  getHealth: (signal) => request<HealthResponse>('/health', { signal }),

  getStats: (signal) => request<StatsResponse>('/stats', { signal }),

  score: (event, signal) =>
    request<ScoredEvent>('/score', { method: 'POST', body: JSON.stringify(event), signal }),

  getAlerts: (query, signal) => request<AlertsResponse>(`/alerts${queryString(query)}`, { signal }),

  getAlert: (eventId, signal) => request<AlertDetail>(`/alerts/${encodeURIComponent(eventId)}`, { signal }),

  getEntity: (entityId, signal) => request<EntityDetail>(`/entities/${encodeURIComponent(entityId)}`, { signal }),

  sendFeedback: (body: FeedbackRequest, signal) =>
    request<FeedbackResponse>('/feedback', { method: 'POST', body: JSON.stringify(body), signal }),

  getMetrics: (signal) => request<MetricsResponse>('/metrics', { signal }),

  subscribeStream(handlers: StreamHandlers): StreamUnsubscribe {
    const source = new EventSource(`${BASE}/stream`)

    const parse = <T,>(raw: string, onValue: (value: T) => void) => {
      try {
        onValue(JSON.parse(raw) as T)
      } catch (error) {
        handlers.onError?.(error instanceof Error ? error : new Error('Malformed SSE payload'))
      }
    }

    source.addEventListener('alert', (e) => parse<ScoredEvent>((e as MessageEvent<string>).data, (v) => handlers.onAlert?.(v)))
    source.addEventListener('stats', (e) => parse<StatsResponse>((e as MessageEvent<string>).data, (v) => handlers.onStats?.(v)))
    source.addEventListener('heartbeat', () => handlers.onHeartbeat?.())
    source.onopen = () => handlers.onOpen?.()
    source.onerror = () => handlers.onError?.(new Error('SSE connection lost'))

    return () => source.close()
  },

  controlStream: (body: StreamControlRequest, signal) =>
    request<StreamControlResponse>('/stream/control', { method: 'POST', body: JSON.stringify(body), signal }),
}
