import { api } from '../api/client'
import type { MetricsResponse } from '../api/types'
import { useAsync } from './useAsync'

export function useMetrics() {
  return useAsync<MetricsResponse>((signal) => api.getMetrics(signal), [])
}
