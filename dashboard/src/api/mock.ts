/**
 * Mock implementation of `SentinelApi`.
 *
 * Serves the deterministic dataset from `mockDataset.ts` behind the exact same
 * interface as the real backend, including a timer-driven stand-in for the SSE
 * stream so the live view is demoable with no Python process running.
 */

import type {
  AlertDetail,
  AlertQuery,
  AlertsResponse,
  EntityDetail,
  EntityListResponse,
  EntitySummary,
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
import {
  MOCK_BASE_EVENTS_PROCESSED,
  MOCK_DATASET,
  MOCK_METRICS,
  buildEntityDetail,
  generateStreamEvent,
  thresholdForBudget,
} from './mockDataset'

const DEFAULT_BUDGET_PCT = 1
const BASE_TICK_MS = 1500

function delay<T>(value: T, signal?: AbortSignal, ms = 90 + Math.random() * 130): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => resolve(value), ms)
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(timer)
        reject(new DOMException('Aborted', 'AbortError'))
      },
      { once: true },
    )
  })
}

/* ------------------------------------------------------------ stream engine */

interface Subscriber {
  handlers: StreamHandlers
}

const startedAt = Date.now()

const engine = {
  subscribers: new Set<Subscriber>(),
  timer: null as ReturnType<typeof setInterval> | null,
  heartbeat: null as ReturnType<typeof setInterval> | null,
  speed: 1,
  state: 'running' as 'running' | 'paused',
  emitted: 0,
  alertsRaised: 0,
  alertsByType: {} as Record<string, number>,
}

function currentStats(): StatsResponse {
  const uptime = (Date.now() - startedAt) / 1000
  const processed = MOCK_BASE_EVENTS_PROCESSED + MOCK_DATASET.events.length + engine.emitted
  const baselineAlerts = MOCK_DATASET.events.filter((e) => e.risk_score >= 70).length
  const alertsByType: Record<string, number> = { ...engine.alertsByType }
  for (const e of MOCK_DATASET.events) {
    if (e.risk_score >= 70) {
      alertsByType[e.predicted_attack_type] = (alertsByType[e.predicted_attack_type] ?? 0) + 1
    }
  }
  return {
    events_processed: processed,
    alerts_raised: baselineAlerts + engine.alertsRaised,
    alerts_by_type: alertsByType,
    events_per_sec: engine.state === 'running' ? Number(((1000 / (BASE_TICK_MS / engine.speed)) * 847).toFixed(1)) : 0,
    uptime_s: Math.round(uptime),
    stream_position: MOCK_DATASET.events.length + engine.emitted,
    stream_total: 24_000,
  }
}

function broadcastStats(): void {
  const stats = currentStats()
  for (const sub of engine.subscribers) sub.handlers.onStats?.(stats)
}

function tick(): void {
  if (engine.state !== 'running' || engine.subscribers.size === 0) return
  const event = generateStreamEvent()
  engine.emitted += 1
  if (event.risk_score >= 70) {
    engine.alertsRaised += 1
    engine.alertsByType[event.predicted_attack_type] = (engine.alertsByType[event.predicted_attack_type] ?? 0) + 1
  }
  for (const sub of engine.subscribers) sub.handlers.onAlert?.(event)
  if (engine.emitted % 2 === 0) broadcastStats()
}

function ensureTimers(): void {
  if (engine.timer === null) {
    engine.timer = setInterval(tick, Math.max(120, BASE_TICK_MS / engine.speed))
  }
  if (engine.heartbeat === null) {
    engine.heartbeat = setInterval(() => {
      for (const sub of engine.subscribers) sub.handlers.onHeartbeat?.()
    }, 8000)
  }
}

function clearTimers(): void {
  if (engine.timer !== null) {
    clearInterval(engine.timer)
    engine.timer = null
  }
  if (engine.heartbeat !== null) {
    clearInterval(engine.heartbeat)
    engine.heartbeat = null
  }
}

function restartTimer(): void {
  if (engine.timer !== null) {
    clearInterval(engine.timer)
    engine.timer = null
  }
  if (engine.subscribers.size > 0) ensureTimers()
}

/* -------------------------------------------------------------- feedback */

const feedbackLog = new Map<string, FeedbackRequest>()

export function mockFeedbackFor(eventId: string): FeedbackRequest | undefined {
  return feedbackLog.get(eventId)
}

/* ------------------------------------------------------------- selectors */

function withBudget(event: ScoredEvent, threshold: number): ScoredEvent {
  const isAlert = event.risk_score >= threshold
  return isAlert === event.is_alert ? event : { ...event, is_alert: isAlert }
}

function entitySummaryFor(entityId: string): EntitySummary {
  const events = MOCK_DATASET.byEntity.get(entityId) ?? []
  const entity = MOCK_DATASET.entityById.get(entityId)
  const risks = events.map((e) => e.risk_score)
  const timestamps = events.map((e) => Date.parse(e.timestamp))
  return {
    entity_id: entityId,
    entity_type: entity?.entity_type ?? 'user',
    cohort: entity?.cohort ?? 'unknown',
    event_count: entity?.cold_start ? events.length : events.length + 240,
    alert_count: events.filter((e) => e.risk_score >= 70).length,
    first_seen: new Date(entity?.first_seen ?? Math.min(...timestamps)).toISOString(),
    last_seen: new Date(timestamps.length ? Math.max(...timestamps) : Date.now()).toISOString(),
    cold_start: entity?.cold_start ?? false,
    mean_risk: risks.length ? Number((risks.reduce((a, b) => a + b, 0) / risks.length).toFixed(1)) : 0,
    max_risk: risks.length ? Math.max(...risks) : 0,
  }
}

/* ------------------------------------------------------------------- api */

export const mockApi: SentinelApi = {
  mode: 'mock',

  getHealth(signal) {
    return delay<HealthResponse>(
      {
        status: 'ok',
        version: '0.1.0-mock',
        model_loaded: true,
        torch_available: MOCK_DATASET.torchOn,
      },
      signal,
      60,
    )
  },

  getStats(signal) {
    return delay(currentStats(), signal, 50)
  },

  score(event, signal) {
    // Re-score by echoing the closest existing event; the real backend runs the stack.
    const entityId = String(event.entity_id ?? '')
    const candidate = MOCK_DATASET.byEntity.get(entityId)?.[0] ?? MOCK_DATASET.events[0]
    return delay({ ...candidate, event: { ...candidate.event, ...event } }, signal)
  },

  getAlerts(query: AlertQuery, signal) {
    const {
      limit = 200,
      offset = 0,
      min_risk = 0,
      attack_type,
      entity_id,
      budget_pct = DEFAULT_BUDGET_PCT,
      sort = 'risk_desc',
    } = query

    const threshold = thresholdForBudget(budget_pct)

    let rows = MOCK_DATASET.events.filter((e) => {
      if (e.risk_score < min_risk) return false
      if (attack_type && e.predicted_attack_type !== attack_type) return false
      if (entity_id && e.entity_id !== entity_id) return false
      return true
    })

    rows = [...rows].sort((a, b) => {
      switch (sort) {
        case 'risk_asc':
          return a.risk_score - b.risk_score
        case 'time_asc':
          return Date.parse(a.timestamp) - Date.parse(b.timestamp)
        case 'time_desc':
          return Date.parse(b.timestamp) - Date.parse(a.timestamp)
        default:
          return b.risk_score - a.risk_score
      }
    })

    const total = rows.length
    const page = rows.slice(offset, offset + limit).map((e) => withBudget(e, threshold))
    return delay<AlertsResponse>({ alerts: page, total }, signal)
  },

  getAlert(eventId, signal) {
    const event = MOCK_DATASET.byId.get(eventId)
    if (!event) return Promise.reject(new Error(`Alert ${eventId} not found`))

    const similar_alerts = MOCK_DATASET.events
      .filter(
        (e) =>
          e.event_id !== eventId &&
          (e.predicted_attack_type === event.predicted_attack_type || e.entity_id === event.entity_id) &&
          e.risk_score >= 45,
      )
      .sort((a, b) => Math.abs(a.risk_score - event.risk_score) - Math.abs(b.risk_score - event.risk_score))
      .slice(0, 5)

    return delay<AlertDetail>(
      { ...event, entity_summary: entitySummaryFor(event.entity_id), similar_alerts },
      signal,
    )
  },

  getEntity(entityId, signal) {
    const detail = buildEntityDetail(entityId)
    if (!detail) return Promise.reject(new Error(`Entity ${entityId} not found`))
    return delay<EntityDetail>(detail, signal, 140)
  },

  listEntities(params, signal) {
    const allIds = [...MOCK_DATASET.entityById.keys()]
    const summaries: EntitySummary[] = allIds.map(entitySummaryFor)
    const { limit = 100, offset = 0, sort = 'risk_desc' } = params
    const sorted = [...summaries].sort((a, b) => {
      if (sort === 'risk_desc') return b.mean_risk - a.mean_risk
      if (sort === 'risk_asc') return a.mean_risk - b.mean_risk
      if (sort === 'name_asc') return a.entity_id.localeCompare(b.entity_id)
      if (sort === 'event_desc') return b.event_count - a.event_count
      return 0
    })
    const page = sorted.slice(offset, offset + limit)
    return delay<EntityListResponse>({ entities: page, total: allIds.length }, signal, 80)
  },

  sendFeedback(body, signal) {
    feedbackLog.set(body.event_id, body)
    const response: FeedbackResponse = {
      ok: true,
      // The backend nudges the threshold when analysts consistently reject alerts.
      updated_threshold:
        body.verdict === 'false_positive' ? Number((70 + feedbackLog.size * 0.15).toFixed(2)) : undefined,
    }
    return delay(response, signal, 120)
  },

  getMetrics(signal) {
    return delay<MetricsResponse>(MOCK_METRICS, signal, 220)
  },

  subscribeStream(handlers: StreamHandlers): StreamUnsubscribe {
    const sub: Subscriber = { handlers }
    engine.subscribers.add(sub)
    ensureTimers()
    const openTimer = setTimeout(() => {
      handlers.onOpen?.()
      handlers.onStats?.(currentStats())
    }, 120)

    return () => {
      clearTimeout(openTimer)
      engine.subscribers.delete(sub)
      if (engine.subscribers.size === 0) clearTimers()
    }
  },

  controlStream(body: StreamControlRequest, signal) {
    if (body.speed !== undefined && body.speed > 0) {
      engine.speed = body.speed
      restartTimer()
    }
    switch (body.action) {
      case 'start':
        engine.state = 'running'
        break
      case 'pause':
        engine.state = 'paused'
        break
      case 'reset':
        engine.state = 'running'
        engine.emitted = 0
        engine.alertsRaised = 0
        engine.alertsByType = {}
        break
    }
    return delay<StreamControlResponse>({ ok: true, state: engine.state }, signal, 40)
  },
}
