import { BarChart3, ListFilter, User } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

export type ViewId = 'triage' | 'entity' | 'ops'

export interface ViewDescriptor {
  id: ViewId
  label: string
  icon: LucideIcon
}

export const VIEWS: ViewDescriptor[] = [
  { id: 'triage', label: 'Triage', icon: ListFilter },
  { id: 'entity', label: 'Entity', icon: User },
  { id: 'ops', label: 'Operations', icon: BarChart3 },
]
