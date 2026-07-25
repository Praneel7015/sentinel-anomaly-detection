import { api } from '../api/client'
import type { EntityDetail } from '../api/types'
import { useAsync } from './useAsync'

export function useEntity(entityId: string | null) {
  return useAsync<EntityDetail | null>(
    (signal) => (entityId ? api.getEntity(entityId, signal) : Promise.resolve(null)),
    [entityId],
  )
}
