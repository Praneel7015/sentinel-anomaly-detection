import { useMemo } from 'react'
import { api } from '../api/client'
import type { AlertQuery, AlertsResponse } from '../api/types'
import { useAsync } from './useAsync'

export function useAlerts(query: AlertQuery) {
  const { limit, offset, min_risk, attack_type, entity_id, budget_pct, sort } = query

  const stable = useMemo<AlertQuery>(
    () => ({ limit, offset, min_risk, attack_type, entity_id, budget_pct, sort }),
    [limit, offset, min_risk, attack_type, entity_id, budget_pct, sort],
  )

  return useAsync<AlertsResponse>((signal) => api.getAlerts(stable, signal), [stable])
}

export function useAlertDetail(eventId: string | null) {
  return useAsync((signal) => (eventId ? api.getAlert(eventId, signal) : Promise.resolve(null)), [eventId])
}
