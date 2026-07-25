import type { AttackType, EntityType, RiskBand } from '../api/types'

export const ATTACK_TYPES: AttackType[] = [
  'brute_force',
  'impossible_travel',
  'credential_stuffing',
  'lateral_movement',
  'device_spoofing',
  'low_and_slow_exfil',
  'insider_drift',
]

export const ENTITY_TYPES: EntityType[] = ['user', 'service_account', 'edge_device']

export const COHORTS = [
  'finance',
  'engineering',
  'sales',
  'hr',
  'it_ops',
  'exec',
  'ci_runner',
  'backup_svc',
  'iot_sensor',
  'pos_terminal',
] as const

interface AttackMeta {
  label: string
  short: string
  /** Hex, used for chart series where Tailwind classes cannot reach. */
  color: string
  /** Tailwind classes for the chip: text + border + translucent background. */
  chip: string
  description: string
}

const UNKNOWN_META: AttackMeta = {
  label: 'Unknown / novel',
  short: 'NOVEL',
  color: '#c084fc',
  chip: 'text-purple-700 border-purple-400/50 bg-purple-50 dark:text-purple-300 dark:border-purple-400/40 dark:bg-purple-400/10',
  description: 'Detectors and signature matcher disagree — pattern not in the known taxonomy.',
}

export const ATTACK_META: Record<string, AttackMeta> = {
  brute_force: {
    label: 'Brute force',
    short: 'BRUTE',
    color: '#ef4444',
    chip: 'text-red-700 border-red-400/50 bg-red-50 dark:text-red-300 dark:border-red-400/40 dark:bg-red-400/10',
    description: 'High-rate authentication failures against a single account.',
  },
  impossible_travel: {
    label: 'Impossible travel',
    short: 'TRAVEL',
    color: '#f97316',
    chip: 'text-orange-700 border-orange-400/50 bg-orange-50 dark:text-orange-300 dark:border-orange-400/40 dark:bg-orange-400/10',
    description: 'Geographically implausible session velocity between consecutive logins.',
  },
  credential_stuffing: {
    label: 'Credential stuffing',
    short: 'STUFF',
    color: '#eab308',
    chip: 'text-yellow-700 border-yellow-400/50 bg-yellow-50 dark:text-yellow-300 dark:border-yellow-400/40 dark:bg-yellow-400/10',
    description: 'Replayed credential pairs sprayed across many accounts from shared infrastructure.',
  },
  lateral_movement: {
    label: 'Lateral movement',
    short: 'LATERAL',
    color: '#22d3ee',
    chip: 'text-cyan-700 border-cyan-400/50 bg-cyan-50 dark:text-cyan-300 dark:border-cyan-400/40 dark:bg-cyan-400/10',
    description: 'Entity reaching resources well outside its cohort access graph.',
  },
  device_spoofing: {
    label: 'Device spoofing',
    short: 'SPOOF',
    color: '#a78bfa',
    chip: 'text-violet-700 border-violet-400/50 bg-violet-50 dark:text-violet-300 dark:border-violet-400/40 dark:bg-violet-400/10',
    description: 'Device fingerprint inconsistent with the entity enrolment history.',
  },
  low_and_slow_exfil: {
    label: 'Low & slow exfil',
    short: 'EXFIL',
    color: '#f472b6',
    chip: 'text-pink-700 border-pink-400/50 bg-pink-50 dark:text-pink-300 dark:border-pink-400/40 dark:bg-pink-400/10',
    description: 'Sustained off-hours data egress kept under per-event volume thresholds.',
  },
  insider_drift: {
    label: 'Insider drift',
    short: 'DRIFT',
    color: '#94a3b8',
    chip: 'text-slate-600 border-slate-400/50 bg-slate-100 dark:text-slate-300 dark:border-slate-400/40 dark:bg-slate-400/10',
    description: 'Gradual, persistent behavioural shift — often a benign role change.',
  },
  normal: {
    label: 'Normal',
    short: 'NORMAL',
    color: '#22c55e',
    chip: 'text-green-700 border-green-400/50 bg-green-50 dark:text-green-300 dark:border-green-400/40 dark:bg-green-400/10',
    description: 'Consistent with the entity baseline and its cohort.',
  },
  unknown_novel: UNKNOWN_META,
}

export function attackMeta(type: string): AttackMeta {
  return ATTACK_META[type] ?? UNKNOWN_META
}

export const ENTITY_TYPE_LABEL: Record<string, string> = {
  user: 'User',
  service_account: 'Service account',
  edge_device: 'Edge device',
}

/* ------------------------------------------------------------------- risk */

export const RISK_COLOR: Record<RiskBand, string> = {
  low: '#16a34a',
  medium: '#b45309',
  high: '#ea580c',
  critical: '#EE3124',
}

export const RISK_BAND_CLASS: Record<RiskBand, string> = {
  low: 'text-risk-low border-risk-low/40 bg-risk-low/10',
  medium: 'text-risk-medium border-risk-medium/40 bg-risk-medium/10',
  high: 'text-risk-high border-risk-high/40 bg-risk-high/10',
  critical: 'text-risk-critical border-risk-critical/40 bg-risk-critical/10',
}

export const RISK_BAR_CLASS: Record<RiskBand, string> = {
  low: 'bg-risk-low',
  medium: 'bg-risk-medium',
  high: 'bg-risk-high',
  critical: 'bg-risk-critical',
}

export const RISK_TEXT_CLASS: Record<RiskBand, string> = {
  low: 'text-risk-low',
  medium: 'text-risk-medium',
  high: 'text-risk-high',
  critical: 'text-risk-critical',
}

/** Thresholds mirror the backend banding in `fusion.py`. */
export function bandForScore(score: number): RiskBand {
  if (score >= 85) return 'critical'
  if (score >= 65) return 'high'
  if (score >= 40) return 'medium'
  return 'low'
}

export function riskColor(score: number): string {
  return RISK_COLOR[bandForScore(score)]
}

/* ------------------------------------------------------------- formatting */

export function formatPct(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`
}

export function formatCompact(value: number): string {
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
  if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(1)}k`
  return String(Math.round(value))
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`
}

export function formatDuration(seconds: number): string {
  const s = Math.floor(seconds)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const rem = s % 60
  if (h > 0) return `${h}h ${m}m`
  if (m > 0) return `${m}m ${rem}s`
  return `${rem}s`
}

export const DETECTOR_LABEL: Record<string, string> = {
  profile: 'Profile',
  isolation: 'Isolation',
  sequence: 'Sequence',
  graph: 'Graph',
  gru: 'GRU-AE',
}
