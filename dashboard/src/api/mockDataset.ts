/**
 * Deterministic mock dataset for the SENTINEL dashboard.
 *
 * ~2000 scored events across ~120 entities covering every attack type plus the
 * benign confounders, generated from a fixed seed so the UI looks identical on
 * every reload. Only the absolute time anchor is taken from the wall clock, so
 * that relative timestamps ("4m ago") stay believable in a live demo.
 */

import type {
  Contribution,
  Counterfactual,
  DetectorScores,
  EntityDetail,
  MetricsResponse,
  RiskBand,
  ScoredEvent,
} from './types'
import {
  ACTIONS,
  ATTACK_SIGNATURES,
  CONFOUNDERS,
  COHORT_CATALOG,
  COHORT_NAMES,
  DEVICE_OS,
  FEATURES,
  FIRST_NAMES,
  HOME_CITIES,
  HOSTILE_CITIES,
  LAST_NAMES,
  SENSITIVE_RESOURCES,
  USER_AGENTS,
  type City,
} from './mockCatalog'
import { chance, clamp, gauss, makeRng, pick, pickMany, randFloat, randInt, round, type Rng } from './mockRandom'

const SEED = 0x5e17e1
const ENTITY_COUNT = 120
const EVENT_COUNT = 2000
const WINDOW_MS = 7 * 24 * 3600 * 1000
const NOW = Date.now()
const WINDOW_START = NOW - WINDOW_MS

const ATTACK_KINDS = [
  'brute_force',
  'impossible_travel',
  'credential_stuffing',
  'lateral_movement',
  'device_spoofing',
  'low_and_slow_exfil',
] as const

export interface MockEntity {
  entity_id: string
  entity_type: string
  cohort: string
  home: City
  resources: string[]
  devices: { device_id: string; os: string; enrolled_days: number }[]
  ipPrefix: string
  /** Peak hour of the circadian profile, local-ish. */
  peakHour: number
  spreadHours: number
  cold_start: boolean
  drifting: boolean
  first_seen: number
}

/* --------------------------------------------------------------- entities */

function buildEntities(rng: Rng): MockEntity[] {
  const userCohorts = COHORT_NAMES.filter((c) => COHORT_CATALOG[c].entityType === 'user')
  const svcCohorts = COHORT_NAMES.filter((c) => COHORT_CATALOG[c].entityType === 'service_account')
  const devCohorts = COHORT_NAMES.filter((c) => COHORT_CATALOG[c].entityType === 'edge_device')

  const entities: MockEntity[] = []
  const used = new Set<string>()

  const makeId = (cohort: string, index: number): string => {
    const kind = COHORT_CATALOG[cohort].entityType
    if (kind === 'user') {
      for (let attempt = 0; attempt < 40; attempt++) {
        const id = `u.${pick(rng, FIRST_NAMES)}.${pick(rng, LAST_NAMES)}`
        if (!used.has(id)) return id
      }
      return `u.user${index}`
    }
    if (kind === 'service_account') return `svc-${cohort.replace('_', '-')}-${String(index).padStart(2, '0')}`
    return `dev-${cohort === 'iot_sensor' ? 'iot' : 'pos'}-${1000 + index}`
  }

  for (let i = 0; i < ENTITY_COUNT; i++) {
    const cohort =
      i < 80 ? userCohorts[i % userCohorts.length] : i < 100 ? svcCohorts[i % svcCohorts.length] : devCohorts[i % devCohorts.length]
    const spec = COHORT_CATALOG[cohort]
    const entity_id = makeId(cohort, i)
    used.add(entity_id)

    const isAutomated = spec.entityType !== 'user'
    const coldStart = i >= ENTITY_COUNT - 14
    const deviceCount = isAutomated ? 1 : randInt(rng, 1, 3)

    entities.push({
      entity_id,
      entity_type: spec.entityType,
      cohort,
      home: pick(rng, HOME_CITIES),
      resources: pickMany(rng, spec.resources, Math.min(spec.resources.length, randInt(rng, 3, 5))),
      devices: Array.from({ length: deviceCount }, (_, d) => ({
        device_id: `${isAutomated ? 'node' : 'wks'}-${randInt(rng, 100000, 999999)}`,
        os: isAutomated ? 'Ubuntu 22.04' : DEVICE_OS[(i + d) % DEVICE_OS.length],
        enrolled_days: randInt(rng, 20, 900),
      })),
      ipPrefix: isAutomated ? `10.${randInt(rng, 20, 60)}` : `10.${randInt(rng, 61, 120)}`,
      peakHour: isAutomated ? randInt(rng, 0, 23) : randInt(rng, 8, 17),
      spreadHours: isAutomated ? 8 : randFloat(rng, 1.4, 3.2),
      cold_start: coldStart,
      drifting: false,
      first_seen: coldStart ? NOW - randInt(rng, 4, 40) * 3600 * 1000 : WINDOW_START - randInt(rng, 40, 500) * 24 * 3600 * 1000,
    })
  }
  return entities
}

/* ------------------------------------------------------- event construction */

type EventKind =
  | { sort: 'normal' }
  | { sort: 'attack'; attack: string }
  | { sort: 'confounder'; kind: string; mimics: string; label: string }

interface EventContext {
  city: City
  source_ip: string
  device_id: string
  device_os: string
  user_agent: string
  resource: string
  action: string
  auth_result: 'success' | 'failure'
  bytes_transferred: number
  protocol: string
  status_code: number
  session_id: string
  features: Record<string, number>
}

function ipFor(rng: Rng, entity: MockEntity, hostile: boolean): string {
  if (hostile) return `${pick(rng, ['185', '45', '91', '103'])}.${randInt(rng, 2, 250)}.${randInt(rng, 2, 250)}.${randInt(rng, 2, 250)}`
  return `${entity.ipPrefix}.${randInt(rng, 1, 250)}.${randInt(rng, 2, 250)}`
}

/** Interpolate a feature between its typical and extreme value. */
function featureValue(rng: Rng, key: string, intensity: number): number {
  const [lo, hi] = FEATURES[key].range
  const jitter = clamp(intensity + gauss(rng, 0, 0.08), 0, 1)
  // Ease-in so extreme values stay rare even at high intensity.
  return lo + (hi - lo) * jitter ** 1.7
}

function buildContext(rng: Rng, entity: MockEntity, kind: EventKind, hour: number, intensity: number): EventContext {
  const attackish = kind.sort === 'attack' ? kind.attack : kind.sort === 'confounder' ? kind.mimics : 'normal'
  const signature = ATTACK_SIGNATURES[attackish] ?? ATTACK_SIGNATURES.normal

  const features: Record<string, number> = {}
  for (const key of signature.primary) features[key] = featureValue(rng, key, clamp(intensity + 0.15, 0, 1))
  for (const key of pickMany(rng, signature.secondary, randInt(rng, 1, 3))) {
    features[key] = featureValue(rng, key, clamp(intensity * 0.65, 0, 1))
  }
  if (!('hour_surprisal' in features)) features.hour_surprisal = featureValue(rng, 'hour_surprisal', intensity * 0.4)

  const hostile = attackish === 'impossible_travel' || attackish === 'credential_stuffing' || attackish === 'brute_force'
  const useHostileGeo = kind.sort === 'attack' && hostile && chance(rng, 0.75)
  const city = useHostileGeo ? pick(rng, HOSTILE_CITIES) : kind.sort === 'confounder' && kind.kind === 'benign_travel' ? pick(rng, HOME_CITIES) : entity.home

  const device = pick(rng, entity.devices)
  const spoofed = attackish === 'device_spoofing' && chance(rng, 0.8)
  const newEnrol = kind.sort === 'confounder' && kind.kind === 'new_device_enrolment'

  const lateral = attackish === 'lateral_movement'
  const resource = lateral && chance(rng, 0.7) ? pick(rng, SENSITIVE_RESOURCES) : pick(rng, entity.resources)

  const failing = attackish === 'brute_force' || attackish === 'credential_stuffing'
  const auth_result: 'success' | 'failure' = failing && chance(rng, 0.82) ? 'failure' : 'success'

  const exfil = attackish === 'low_and_slow_exfil'
  const bytes = exfil
    ? randInt(rng, 4_000_000, 48_000_000)
    : entity.entity_type === 'edge_device'
      ? randInt(rng, 800, 240_000)
      : randInt(rng, 2_000, 3_500_000)

  const action = failing ? 'login' : exfil ? 'file_download' : lateral ? pick(rng, ['ssh_connect', 'db_query', 'share_mount', 'config_change']) : pick(rng, ACTIONS)

  return {
    city,
    source_ip: ipFor(rng, entity, useHostileGeo || (kind.sort === 'attack' && chance(rng, 0.3))),
    device_id: spoofed ? `wks-${randInt(rng, 100000, 999999)}` : newEnrol ? `wks-${randInt(rng, 100000, 999999)}` : device.device_id,
    device_os: spoofed ? pick(rng, DEVICE_OS) : device.os,
    user_agent: spoofed || failing ? pick(rng, USER_AGENTS.slice(3)) : pick(rng, USER_AGENTS.slice(0, 3)),
    resource,
    action,
    auth_result,
    bytes_transferred: bytes,
    protocol: pick(rng, ['https', 'ssh', 'rdp', 'smb', 'mqtt']),
    status_code: auth_result === 'failure' ? 401 : pick(rng, [200, 200, 200, 206, 302]),
    session_id: `s-${randInt(rng, 100000, 999999)}`,
    features: { ...features, ...(hour < 6 || hour > 21 ? { hour_surprisal: Math.max(features.hour_surprisal ?? 0, featureValue(rng, 'hour_surprisal', 0.72)) } : {}) },
  }
}

const BASE_RISK = 6

function buildContributions(rng: Rng, ctx: EventContext, riskScore: number): Contribution[] {
  const raw: { key: string; weight: number }[] = Object.entries(ctx.features).map(([key, value]) => {
    const [lo, hi] = FEATURES[key].range
    const norm = clamp((value - lo) / (hi - lo), 0, 1)
    return { key, weight: 0.25 + norm ** 0.8 * 2.4 }
  })

  // One or two mitigating factors so the waterfall genuinely diverges.
  const mitigators: { key: string; weight: number }[] = []
  if (!('device_age' in ctx.features) && chance(rng, 0.7)) {
    mitigators.push({ key: 'device_age', weight: -randFloat(rng, 0.2, 0.9) })
  }
  if (!('session_duration' in ctx.features) && chance(rng, 0.45)) {
    mitigators.push({ key: 'session_duration', weight: -randFloat(rng, 0.15, 0.6) })
  }

  const all = [...raw, ...mitigators]
  const sum = all.reduce((acc, r) => acc + r.weight, 0)
  const target = Math.max(riskScore - BASE_RISK, 1.5)
  const scale = sum > 0.01 ? target / sum : 1

  return all
    .map(({ key, weight }) => {
      const spec = FEATURES[key]
      const value = ctx.features[key] ?? featureValue(rng, key, randFloat(rng, 0.05, 0.35))
      const display_value = spec.format(value)
      const contribution = round(weight * scale, 2)
      return {
        feature: spec.feature,
        display_name: spec.display_name,
        value: round(value, 3),
        display_value,
        contribution,
        direction: (contribution >= 0 ? 'increases' : 'decreases') as Contribution['direction'],
        description: spec.describe(display_value),
      }
    })
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
}

function buildCounterfactuals(contributions: Contribution[], riskScore: number): Counterfactual[] {
  return contributions
    .filter((c) => c.contribution > 0)
    .slice(0, 3)
    .map((c) => {
      const neutralised = round(clamp(riskScore - c.contribution, 0, 100), 1)
      return {
        feature: c.feature,
        display_name: c.display_name,
        neutralised_risk: neutralised,
        delta: round(neutralised - riskScore, 1),
      }
    })
}

function buildDetectorScores(rng: Rng, attackish: string, intensity: number, torchOn: boolean): DetectorScores {
  const sig = ATTACK_SIGNATURES[attackish] ?? ATTACK_SIGNATURES.normal
  const jitter = (base: number) => round(clamp(base * (0.55 + intensity * 0.6) + gauss(rng, 0, 0.06), 0.01, 0.99), 3)
  return {
    profile: jitter(sig.detectors.profile),
    isolation: jitter(sig.detectors.isolation),
    sequence: jitter(sig.detectors.sequence),
    graph: jitter(sig.detectors.graph),
    gru: torchOn ? jitter(sig.detectors.gru) : null,
  }
}

function bandFor(score: number): RiskBand {
  if (score >= 85) return 'critical'
  if (score >= 65) return 'high'
  if (score >= 40) return 'medium'
  return 'low'
}

/* ------------------------------------------------------------- narratives */

function narrativeFor(
  entity: MockEntity,
  kind: EventKind,
  ctx: EventContext,
  contributions: Contribution[],
  riskScore: number,
  when: Date,
): string {
  const f = ctx.features
  const time = when.toISOString().slice(11, 16)
  const geo = `${ctx.city.city}, ${ctx.city.country}`
  const top = contributions[0]
  const attackish = kind.sort === 'attack' ? kind.attack : kind.sort === 'confounder' ? kind.mimics : 'normal'

  switch (attackish) {
    case 'brute_force':
      return `${entity.entity_id} accumulated ${Math.round(f.auth_fail_entity ?? 12)} failed authentications in the trailing 15 minutes from ${ctx.source_ip} (${geo}), against a 30-day baseline of under 2 failures per day. The same source address has failed ${Math.round(f.auth_fail_ip ?? 40)} times across the estate in that window, so this is a targeted credential attack rather than a forgotten password. Activity at ${time} also falls outside the entity's usual ${entity.peakHour}:00 working pattern.`
    case 'impossible_travel':
      return `A session for ${entity.entity_id} opened from ${geo} implying ${Math.round(f.geo_velocity ?? 1200)} km/h of travel since the previous login from ${entity.home.city}, ${entity.home.country}. No commercial routing explains the gap. The presenting device fingerprint also differs from the enrolled baseline (distance ${(f.device_mismatch ?? 0.4).toFixed(2)}), which rules out the common false positive of a VPN egress change on a known laptop.`
    case 'credential_stuffing':
      return `Source ${ctx.source_ip} (${geo}) has driven ${Math.round(f.auth_fail_ip ?? 120)} authentication failures across multiple accounts, touching ${entity.entity_id} among them, and the entity has been seen from ${Math.round(f.distinct_ips ?? 6)} distinct addresses in 24 hours. The per-account failure rate is deliberately low, which is why entity-level thresholds alone would miss this — the graph detector flagged the shared-infrastructure pattern.`
    case 'lateral_movement':
      return `${entity.entity_id} (${entity.cohort}) accessed ${ctx.resource}, a resource with cohort-relative novelty ${(f.cohort_novelty ?? 0.8).toFixed(2)} — effectively unseen for this role. Resource breadth reached ${Math.round(f.resource_breadth ?? 12)} distinct targets in the trailing hour versus a cohort norm of 2-4, and the action sequence carries ${(f.sequence_surprisal ?? 6).toFixed(1)} bits of surprisal. The pattern is consistent with an operator enumerating reachable systems from a foothold.`
    case 'device_spoofing':
      return `The device fingerprint presented by ${entity.entity_id} diverges from every enrolled device (distance ${(f.device_mismatch ?? 0.7).toFixed(2)}) while reusing the entity's session cookie. The claimed device was first observed ${FEATURES.device_age.format(f.device_age ?? 0.1)} ago and the user agent (${ctx.user_agent.slice(0, 38)}...) does not match the declared OS ${ctx.device_os}.`
    case 'low_and_slow_exfil':
      return `${entity.entity_id} has moved ${FEATURES.offhours_bytes.format(f.offhours_bytes ?? 2400)} outside its active window over the last 24 hours, split across many small transfers that individually sit under per-event volume thresholds. Each request is unremarkable; the cumulative off-hours counter is what surfaces it. Sequence surprisal of ${(f.sequence_surprisal ?? 7).toFixed(1)} bits indicates a scripted read loop rather than interactive work.`
    case 'insider_drift':
      return `${entity.entity_id} sits ${(f.peer_deviation ?? 2.4).toFixed(1)}σ from the ${entity.cohort} cohort median, with steadily rising novel-resource access (${(f.resource_novelty ?? 0.6).toFixed(2)}). The shift is gradual and monotonic rather than bursty, which is the signature of a role change rather than an intrusion. Page-Hinkley flagged a sustained baseline shift and the profile has been re-based.`
    default:
      return riskScore >= 40
        ? `${entity.entity_id} deviates modestly from baseline: ${top.display_name.toLowerCase()} at ${top.display_value} contributes ${top.contribution.toFixed(1)} points. Nothing in the detector stack agrees strongly enough to name an attack type, and the entity remains within ${(f.peer_deviation ?? 1.2).toFixed(1)}σ of its cohort. Most likely benign variation — surfaced for completeness at the current alert budget.`
        : `${entity.entity_id} performed ${ctx.action} on ${ctx.resource} from ${geo} at ${time}, consistent with its established baseline and with the ${entity.cohort} cohort. All detector scores sit in the lower quartile.`
  }
}

function confounderNarrativeSuffix(kind: string): string {
  switch (kind) {
    case 'benign_travel':
      return ' Note: a travel-approval record exists for this window, which the model does not consume — a known confounder.'
    case 'new_device_enrolment':
      return ' Note: a matching device-enrolment event was recorded shortly before, a known benign trigger for this signature.'
    case 'password_rotation':
      return ' Note: the failures cluster immediately after a scheduled credential rotation, a known benign trigger.'
    case 'vacation_return':
      return ' Note: the entity had a 14-day activity gap immediately prior, so baseline staleness inflates the surprisal terms.'
    case 'maintenance_burst':
      return ' Note: the burst falls inside a declared maintenance window for this service account.'
    default:
      return ''
  }
}

/* ------------------------------------------------------------ event factory */

interface MakeEventArgs {
  rng: Rng
  entity: MockEntity
  kind: EventKind
  timestamp: number
  episodeId: string | null
  sequence: number
  entityEventCount: number
  torchOn: boolean
}

function makeScoredEvent(args: MakeEventArgs): ScoredEvent {
  const { rng, entity, kind, timestamp, episodeId, sequence, entityEventCount, torchOn } = args
  const when = new Date(timestamp)
  const hour = when.getUTCHours()

  const intensity =
    kind.sort === 'attack'
      ? clamp(gauss(rng, 0.72, 0.17), 0.12, 1)
      : kind.sort === 'confounder'
        ? clamp(gauss(rng, 0.48, 0.13), 0.1, 0.9)
        : clamp(gauss(rng, 0.16, 0.12), 0.01, 0.75)

  const ctx = buildContext(rng, entity, kind, hour, intensity)

  let riskScore: number
  if (kind.sort === 'attack') riskScore = clamp(gauss(rng, 22 + intensity * 74, 9), 12, 99)
  else if (kind.sort === 'confounder') riskScore = clamp(gauss(rng, 24 + intensity * 72, 11), 14, 93)
  else riskScore = clamp(gauss(rng, 8 + intensity * 62, 7), 1, 78)
  if (entity.cold_start) riskScore = clamp(riskScore * 0.86 + 6, 1, 99)
  riskScore = round(riskScore, 1)

  const attackish = kind.sort === 'attack' ? kind.attack : kind.sort === 'confounder' ? kind.mimics : 'normal'
  const detector_scores = buildDetectorScores(rng, attackish, intensity, torchOn)
  const contributions = buildContributions(rng, ctx, riskScore)
  const counterfactuals = buildCounterfactuals(contributions, riskScore)

  // Classifier behaviour: mostly right on real attacks, sometimes disagrees with
  // the signature matcher, which is exactly when we surface "unknown / novel".
  const agreementProb = kind.sort === 'attack' ? 0.86 : kind.sort === 'confounder' ? 0.72 : 0.94
  const classifier_agreement = chance(rng, agreementProb)
  const is_novel = !classifier_agreement && riskScore >= 52 && chance(rng, 0.65)

  let predicted: string
  if (is_novel) predicted = 'unknown_novel'
  else if (riskScore < 38) predicted = 'normal'
  else if (kind.sort === 'attack') predicted = classifier_agreement ? kind.attack : pick(rng, ATTACK_KINDS)
  else if (kind.sort === 'confounder') predicted = kind.mimics
  else predicted = chance(rng, 0.55) ? 'normal' : pick(rng, ATTACK_KINDS)

  const attack_type_confidence = round(
    predicted === 'normal'
      ? clamp(gauss(rng, 0.82, 0.1), 0.4, 0.99)
      : classifier_agreement
        ? clamp(gauss(rng, 0.79, 0.12), 0.35, 0.98)
        : clamp(gauss(rng, 0.44, 0.11), 0.15, 0.68),
    3,
  )

  let narrative = narrativeFor(entity, kind, ctx, contributions, riskScore, when)
  if (kind.sort === 'confounder') narrative += confounderNarrativeSuffix(kind.kind)
  if (entity.cold_start) {
    narrative += ` ${entity.entity_id} has only ${entityEventCount} events of history, so scoring is provisional and shrunk heavily toward the ${entity.cohort} cohort prior.`
  }
  if (is_novel) {
    narrative += ' The attack classifier and the transparent signature matcher disagree, so this is surfaced as an unknown or novel pattern rather than being forced into a known class.'
  }

  const ground_truth =
    kind.sort === 'attack'
      ? { label: kind.attack, is_anomaly: true }
      : kind.sort === 'confounder'
        ? { label: kind.kind, is_anomaly: false }
        : { label: 'normal', is_anomaly: false }

  const event_id = `evt_${timestamp.toString(36)}_${sequence.toString(36).padStart(4, '0')}`

  return {
    event_id,
    entity_id: entity.entity_id,
    entity_type: entity.entity_type,
    cohort: entity.cohort,
    timestamp: when.toISOString(),
    risk_score: riskScore,
    risk_band: bandFor(riskScore),
    is_alert: riskScore >= 70,
    predicted_attack_type: predicted,
    attack_type_confidence,
    classifier_agreement,
    is_novel,
    detector_scores,
    contributions,
    narrative,
    counterfactuals,
    cold_start: entity.cold_start,
    entity_event_count: entityEventCount,
    event: {
      event_id,
      episode_id: episodeId,
      timestamp: when.toISOString(),
      entity_id: entity.entity_id,
      entity_type: entity.entity_type,
      cohort: entity.cohort,
      action: ctx.action,
      resource: ctx.resource,
      source_ip: ctx.source_ip,
      geo_country: ctx.city.country,
      geo_city: ctx.city.city,
      latitude: round(ctx.city.lat, 3),
      longitude: round(ctx.city.lon, 3),
      device_id: ctx.device_id,
      device_os: ctx.device_os,
      user_agent: ctx.user_agent,
      auth_result: ctx.auth_result,
      bytes_transferred: ctx.bytes_transferred,
      session_id: ctx.session_id,
      protocol: ctx.protocol,
      status_code: ctx.status_code,
    },
    ground_truth,
  }
}

/* ---------------------------------------------------------------- dataset */

export interface MockDataset {
  entities: MockEntity[]
  entityById: Map<string, MockEntity>
  events: ScoredEvent[]
  byId: Map<string, ScoredEvent>
  byEntity: Map<string, ScoredEvent[]>
  torchOn: boolean
}

function buildDataset(): MockDataset {
  const rng = makeRng(SEED)
  const entities = buildEntities(rng)
  const torchOn = true
  const events: ScoredEvent[] = []
  const perEntityCount = new Map<string, number>()

  const nextCount = (id: string): number => {
    const n = (perEntityCount.get(id) ?? 0) + 1
    perEntityCount.set(id, n)
    return n
  }

  // Seed every entity with baseline history so entity_event_count is plausible.
  for (const e of entities) perEntityCount.set(e.entity_id, e.cold_start ? randInt(rng, 1, 9) : randInt(rng, 180, 4200))

  let sequence = 0

  // 1. Attack episodes — clustered in time, biased toward the recent window.
  const episodeCount = 46
  for (let ep = 0; ep < episodeCount; ep++) {
    const attack = ATTACK_KINDS[ep % ATTACK_KINDS.length]
    const entity = pick(rng, entities.filter((e) => (attack === 'low_and_slow_exfil' ? true : !e.cold_start || chance(rng, 0.5))))
    const episodeId = `ep_${attack}_${ep.toString().padStart(3, '0')}`
    const length = attack === 'low_and_slow_exfil' ? randInt(rng, 8, 22) : randInt(rng, 3, 12)
    const spanMs = attack === 'low_and_slow_exfil' ? randInt(rng, 6, 30) * 3600 * 1000 : randInt(rng, 4, 90) * 60 * 1000
    const start = WINDOW_START + Math.floor(rng() ** 0.55 * (WINDOW_MS - spanMs))
    for (let i = 0; i < length; i++) {
      const ts = start + Math.floor((spanMs * i) / length) + randInt(rng, 0, 30_000)
      events.push(
        makeScoredEvent({
          rng,
          entity,
          kind: { sort: 'attack', attack },
          timestamp: ts,
          episodeId,
          sequence: sequence++,
          entityEventCount: nextCount(entity.entity_id),
          torchOn,
        }),
      )
    }
  }

  // 2. Insider drift — the benign edge case that must stop firing after adaptation.
  for (let d = 0; d < 4; d++) {
    const entity = entities[10 + d * 7]
    entity.drifting = true
    const episodeId = `ep_insider_drift_${d}`
    for (let i = 0; i < 14; i++) {
      const ts = WINDOW_START + Math.floor((WINDOW_MS * (0.35 + i * 0.04)) % WINDOW_MS)
      events.push(
        makeScoredEvent({
          rng,
          entity,
          kind: { sort: 'confounder', kind: 'insider_drift', mimics: 'insider_drift', label: 'Insider drift' },
          timestamp: ts,
          episodeId,
          sequence: sequence++,
          entityEventCount: nextCount(entity.entity_id),
          torchOn,
        }),
      )
    }
  }

  // 3. Benign confounders — the reason precision is hard.
  for (let c = 0; c < 34; c++) {
    const conf = CONFOUNDERS[c % CONFOUNDERS.length]
    const entity = pick(rng, entities)
    const episodeId = `ep_conf_${conf.kind}_${c}`
    const length = randInt(rng, 2, 7)
    const start = WINDOW_START + Math.floor(rng() * WINDOW_MS)
    for (let i = 0; i < length; i++) {
      events.push(
        makeScoredEvent({
          rng,
          entity,
          kind: { sort: 'confounder', kind: conf.kind, mimics: conf.mimics, label: conf.label },
          timestamp: start + i * randInt(rng, 60_000, 900_000),
          episodeId,
          sequence: sequence++,
          entityEventCount: nextCount(entity.entity_id),
          torchOn,
        }),
      )
    }
  }

  // 4. Normal background traffic, shaped by each entity's circadian profile.
  while (events.length < EVENT_COUNT) {
    const entity = pick(rng, entities)
    const dayOffset = Math.floor(rng() * 7)
    const hour = clamp(Math.round(gauss(rng, entity.peakHour, entity.spreadHours)), 0, 23)
    const ts = WINDOW_START + dayOffset * 24 * 3600 * 1000 + hour * 3600 * 1000 + randInt(rng, 0, 3_599_000)
    events.push(
      makeScoredEvent({
        rng,
        entity,
        kind: { sort: 'normal' },
        timestamp: Math.min(ts, NOW - 1000),
        episodeId: null,
        sequence: sequence++,
        entityEventCount: nextCount(entity.entity_id),
        torchOn,
      }),
    )
  }

  events.sort((a, b) => Date.parse(b.timestamp) - Date.parse(a.timestamp))

  const byId = new Map(events.map((e) => [e.event_id, e]))
  const byEntity = new Map<string, ScoredEvent[]>()
  for (const e of events) {
    const list = byEntity.get(e.entity_id)
    if (list) list.push(e)
    else byEntity.set(e.entity_id, [e])
  }

  return {
    entities,
    entityById: new Map(entities.map((e) => [e.entity_id, e])),
    events,
    byId,
    byEntity,
    torchOn,
  }
}

export const MOCK_DATASET: MockDataset = buildDataset()

/** Baseline event count so streaming counters keep climbing from a believable number. */
export const MOCK_BASE_EVENTS_PROCESSED = 418_233

/* ------------------------------------------------------------ live stream */

const streamRng = makeRng(SEED ^ 0x9e3779b9)
let streamSequence = 900_000

/**
 * Mint a brand-new scored event at the current wall-clock time and register it in
 * the dataset, so alerts arriving over the simulated SSE feed are fully clickable.
 */
export function generateStreamEvent(): ScoredEvent {
  const rng = streamRng
  const entity = pick(rng, MOCK_DATASET.entities)
  const roll = rng()
  const kind: EventKind =
    roll < 0.3
      ? { sort: 'attack', attack: pick(rng, ATTACK_KINDS) }
      : roll < 0.45
        ? (() => {
            const conf = pick(rng, CONFOUNDERS)
            return { sort: 'confounder', kind: conf.kind, mimics: conf.mimics, label: conf.label }
          })()
        : { sort: 'normal' }

  const event = makeScoredEvent({
    rng,
    entity,
    kind,
    timestamp: Date.now(),
    episodeId: kind.sort === 'normal' ? null : `ep_live_${streamSequence}`,
    sequence: streamSequence++,
    entityEventCount: (MOCK_DATASET.byEntity.get(entity.entity_id)?.length ?? 0) + 200,
    torchOn: MOCK_DATASET.torchOn,
  })

  MOCK_DATASET.events.unshift(event)
  MOCK_DATASET.byId.set(event.event_id, event)
  const list = MOCK_DATASET.byEntity.get(event.entity_id)
  if (list) list.unshift(event)
  else MOCK_DATASET.byEntity.set(event.entity_id, [event])

  return event
}

/* ---------------------------------------------------------- entity detail */

const PEER_AXES = [
  'Off-hours activity',
  'Resource breadth',
  'Auth failures',
  'Data egress',
  'Geo spread',
  'Sequence entropy',
]

export function buildEntityDetail(entityId: string): EntityDetail | null {
  const entity = MOCK_DATASET.entityById.get(entityId)
  if (!entity) return null

  const rng = makeRng(hashString(entityId))
  const events = [...(MOCK_DATASET.byEntity.get(entityId) ?? [])].sort(
    (a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp),
  )

  const activity_by_hour = Array.from({ length: 24 }, (_, h) => {
    const dist = Math.min(Math.abs(h - entity.peakHour), 24 - Math.abs(h - entity.peakHour))
    const base = Math.exp(-(dist ** 2) / (2 * entity.spreadHours ** 2))
    return Math.round(base * randInt(rng, 40, 140) + randInt(rng, 0, 6))
  })
  for (const e of events) {
    activity_by_hour[new Date(e.timestamp).getHours()] += 1
  }

  const resourceCounts = new Map<string, number>()
  for (const e of events) {
    const r = String(e.event.resource ?? 'unknown')
    resourceCounts.set(r, (resourceCounts.get(r) ?? 0) + 1)
  }
  for (const r of entity.resources) {
    resourceCounts.set(r, (resourceCounts.get(r) ?? 0) + randInt(rng, 20, 240))
  }

  const top_resources = [...resourceCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([resource, count]) => ({
      resource,
      count,
      is_new: !entity.resources.includes(resource),
    }))

  const risk_timeline = events.map((e) => ({
    timestamp: e.timestamp,
    risk_score: e.risk_score,
    is_alert: e.risk_score >= 70,
  }))

  const maxRisk = events.reduce((m, e) => Math.max(m, e.risk_score), 0)
  const peer_comparison = PEER_AXES.map((axis, i) => ({
    axis,
    entity: round(clamp(gauss(rng, 34 + maxRisk * 0.42, 16) + (i === 1 ? 12 : 0), 4, 98), 1),
    cohort_median: round(clamp(gauss(rng, 32, 8), 8, 62), 1),
  }))

  const lastEvent = events[events.length - 1]
  const driftDetectedAt = entity.drifting
    ? new Date(NOW - randInt(rng, 20, 60) * 3600 * 1000).toISOString()
    : null

  const profile_summary = [
    {
      label: 'Typical active window',
      value: `${String(entity.peakHour).padStart(2, '0')}:00-${String((entity.peakHour + 8) % 24).padStart(2, '0')}:00`,
      cohort_value: entity.entity_type === 'user' ? '09:00-18:00' : '00:00-24:00',
    },
    {
      label: 'Home geography',
      value: `${entity.home.city}, ${entity.home.country}`,
      cohort_value: `${randInt(rng, 2, 6)} sites`,
    },
    {
      label: 'Distinct resources / day',
      value: String(randInt(rng, 3, 11)),
      cohort_value: String(randInt(rng, 3, 6)),
    },
    {
      label: 'Enrolled devices',
      value: String(entity.devices.length),
      cohort_value: entity.entity_type === 'user' ? '2' : '1',
    },
    {
      label: 'Mean egress / session',
      value: `${randInt(rng, 4, 90)} MB`,
      cohort_value: `${randInt(rng, 6, 40)} MB`,
    },
    {
      label: 'Off-hours activity',
      value: `${randInt(rng, 2, 44)}%`,
      cohort_value: entity.entity_type === 'user' ? '7%' : '51%',
    },
    {
      label: 'Auth failure rate',
      value: `${(randFloat(rng, 0.1, 6)).toFixed(1)}%`,
      cohort_value: '0.8%',
    },
  ]

  return {
    entity_id: entity.entity_id,
    entity_type: entity.entity_type,
    cohort: entity.cohort,
    first_seen: new Date(entity.first_seen).toISOString(),
    last_seen: lastEvent ? lastEvent.timestamp : new Date(NOW).toISOString(),
    event_count: entity.cold_start ? events.length + randInt(rng, 0, 6) : events.length + randInt(rng, 200, 3800),
    cold_start: entity.cold_start,
    profile_summary,
    risk_timeline,
    activity_by_hour,
    top_resources,
    peer_comparison,
    drift_state: {
      drifting: entity.drifting,
      detected_at: driftDetectedAt,
      adapted: entity.drifting,
    },
  }
}

function hashString(value: string): number {
  let h = 2166136261
  for (let i = 0; i < value.length; i++) {
    h ^= value.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

/* ---------------------------------------------------------------- metrics */

function buildMetrics(): MetricsResponse {
  const rng = makeRng(SEED ^ 0x2545f491)

  const budget_curve = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5].map((budget_pct) => {
    const alerts = Math.round((budget_pct / 100) * EVENT_COUNT * 12)
    const precision = round(clamp(0.94 - 0.115 * Math.log1p(budget_pct * 5.2), 0.18, 0.98), 4)
    const recall = round(clamp(0.16 + 0.33 * Math.log1p(budget_pct * 3.1), 0.05, 0.97), 4)
    return {
      budget_pct,
      precision,
      recall,
      alerts,
      analyst_hours: round((alerts * 6.5) / 60, 1),
    }
  })

  const pr_curve = Array.from({ length: 41 }, (_, i) => {
    const recall = round(i / 40, 3)
    const precision = round(clamp(0.985 - 0.82 * recall ** 2.4 - gauss(rng, 0, 0.012), 0.06, 1), 4)
    return { recall, precision }
  })

  const per_attack_recall = [
    { attack_type: 'brute_force', support: 68, recall: 0.941 },
    { attack_type: 'impossible_travel', support: 54, recall: 0.907 },
    { attack_type: 'credential_stuffing', support: 61, recall: 0.836 },
    { attack_type: 'lateral_movement', support: 77, recall: 0.792 },
    { attack_type: 'device_spoofing', support: 49, recall: 0.755 },
    { attack_type: 'low_and_slow_exfil', support: 143, recall: 0.671 },
  ].map((r) => ({ ...r, detected: Math.round(r.support * r.recall) }))

  const labels = ['normal', 'brute_force', 'impossible_travel', 'credential_stuffing', 'lateral_movement', 'device_spoofing', 'low_and_slow_exfil', 'unknown_novel']
  const matrix = labels.map((_, i) =>
    labels.map((__, j) => {
      if (i === 0 && j === 0) return 1487
      if (i === j) return randInt(rng, 34, 92)
      if (j === 0) return randInt(rng, 2, 19)
      if (i === 0) return randInt(rng, 1, 14)
      return randInt(rng, 0, 7)
    }),
  )

  return {
    pr_auc: 0.783,
    roc_auc: 0.9714,
    budget_curve,
    per_attack_recall,
    confusion_matrix: { labels, matrix },
    pr_curve,
    fp_rate_confounders: 0.187,
    fp_rate_insider_drift: 0.062,
    mttd: [
      { attack_type: 'brute_force', mean_events: 3.2, mean_minutes: 4.6 },
      { attack_type: 'impossible_travel', mean_events: 1.4, mean_minutes: 1.1 },
      { attack_type: 'credential_stuffing', mean_events: 5.8, mean_minutes: 12.3 },
      { attack_type: 'lateral_movement', mean_events: 4.1, mean_minutes: 9.7 },
      { attack_type: 'device_spoofing', mean_events: 2.3, mean_minutes: 3.4 },
      { attack_type: 'low_and_slow_exfil', mean_events: 11.6, mean_minutes: 214.8 },
    ],
    cold_start: { precision: 0.612, recall: 0.704 },
    post_drift: { precision: 0.741, recall: 0.688 },
    latency_ms: { p50: 1.82, p95: 4.61, p99: 9.34, mean: 2.37 },
    ablation: [
      { variant: 'Full stack', pr_auc: 0.783, precision_at_1pct: 0.842 },
      { variant: '− profile detector', pr_auc: 0.641, precision_at_1pct: 0.688 },
      { variant: '− isolation forest', pr_auc: 0.724, precision_at_1pct: 0.791 },
      { variant: '− sequence (Markov)', pr_auc: 0.702, precision_at_1pct: 0.764 },
      { variant: '− graph detector', pr_auc: 0.688, precision_at_1pct: 0.735 },
      { variant: '− GRU autoencoder (torch off)', pr_auc: 0.761, precision_at_1pct: 0.822 },
      { variant: 'Profile only (baseline)', pr_auc: 0.488, precision_at_1pct: 0.512 },
      { variant: 'Naive rule baseline', pr_auc: 0.297, precision_at_1pct: 0.331 },
    ],
  }
}

export const MOCK_METRICS: MetricsResponse = buildMetrics()

/* ----------------------------------------------------------- budget logic */

/**
 * Resolve the risk threshold implied by an alert budget, then restamp `is_alert`.
 * This is what makes the budget slider visibly move the alert line through the queue.
 */
export function thresholdForBudget(budgetPct: number): number {
  const sorted = MOCK_DATASET.events.map((e) => e.risk_score).sort((a, b) => b - a)
  const index = clamp(Math.floor((budgetPct / 100) * sorted.length), 1, sorted.length - 1)
  return sorted[index]
}
