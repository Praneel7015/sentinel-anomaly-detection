import { api } from '../api/client'
import type { EntityListResponse } from '../api/types'
import { useAsync } from './useAsync'

export function useEntities(sort = 'risk_desc', limit = 200) {
  return useAsync<EntityListResponse>(
    (signal) => api.listEntities({ limit, sort }, signal),
    [sort, limit],
  )
}
