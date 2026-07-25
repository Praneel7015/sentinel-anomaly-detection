import { Cpu, Server, User } from 'lucide-react'
import { ENTITY_TYPE_LABEL } from '../../lib/domain'

export function EntityIcon({ type, size = 12, className }: { type: string; size?: number; className?: string }) {
  const label = ENTITY_TYPE_LABEL[type] ?? type
  const Icon = type === 'service_account' ? Server : type === 'edge_device' ? Cpu : User
  return <Icon size={size} className={className ?? 'text-ink-faint'} aria-label={label} />
}
