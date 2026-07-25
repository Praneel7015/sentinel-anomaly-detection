/**
 * The single switch between the mock dataset and the real FastAPI backend.
 *
 * Set `VITE_USE_MOCK=false` in `dashboard/.env` and restart the dev server to
 * talk to the Python service. Nothing else in the app needs to change.
 */

import { mockApi } from './mock'
import { realApi } from './real'
import type { SentinelApi } from './types'

const rawFlag = import.meta.env.VITE_USE_MOCK
export const USE_MOCK: boolean = String(rawFlag ?? 'true').toLowerCase() !== 'false'

export const api: SentinelApi = USE_MOCK ? mockApi : realApi

export type { SentinelApi }
