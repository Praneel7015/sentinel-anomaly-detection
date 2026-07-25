/**
 * Wire types for the SENTINEL scoring service.
 *
 * Field names here are load-bearing: the FastAPI backend in `src/sentinel/serving`
 * is generated from the same spec, so renaming anything in this file breaks the
 * contract on both sides.
 */

export type RiskBand = 'low' | 'medium' | 'high' | 'critical'

export type AttackType =
  | 'brute_force'
  | 'impossible_travel'
  | 'credential_stuffing'
  | 'lateral_movement'
  | 'device_spoofing'
  | 'low_and_slow_exfil'
  | 'insider_drift'

/** What the classifier can emit: a known attack, benign, or an unrecognised pattern. */
export type PredictedAttackType = AttackType | 'normal' | 'unknown_novel'

export type EntityType = 'user' | 'service_account' | 'edge_device'

export type Verdict = 'true_positive' | 'false_positive' | 'escalate'

export type ContributionDirection = 'increases' | 'decreases'

export interface DetectorScores {
  profile: number
  isolation: number
  sequence: number
  graph: number
  /** null when the optional PyTorch GRU autoencoder is not loaded. */
  gru: number | null
}

export interface Contribution {
  feature: string
  display_name: string
  value: number
  display_value: string
  contribution: number
  direction: ContributionDirection
  description: string
}

export interface Counterfactual {
  feature: string
  display_name: string
  neutralised_risk: number
  delta: number
}

export interface GroundTruth {
  label: string
  is_anomaly: boolean
}

export interface ScoredEvent {
  event_id: string
  entity_id: string
  entity_type: string
  cohort: string
  timestamp: string
  /** 0-100. */
  risk_score: number
  risk_band: RiskBand
  is_alert: boolean
  predicted_attack_type: string
  /** 0-1. */
  attack_type_confidence: number
  classifier_agreement: boolean
  is_novel: boolean
  detector_scores: DetectorScores
  contributions: Contribution[]
  narrative: string
  counterfactuals: Counterfactual[]
  cold_start: boolean
  entity_event_count: number
  /** Raw access-log event, shape defined by the generator schema. */
  event: Record<string, unknown>
  ground_truth: GroundTruth | null
}

/* ------------------------------------------------------------------ health */

export interface HealthResponse {
  status: string
  version: string
  model_loaded: boolean
  torch_available: boolean
}

/* ------------------------------------------------------------------- stats */

export interface StatsResponse {
  events_processed: number
  alerts_raised: number
  alerts_by_type: Record<string, number>
  events_per_sec: number
  uptime_s: number
  stream_position: number
  stream_total: number
}

/* ------------------------------------------------------------------ alerts */

export type AlertSort = 'risk_desc' | 'risk_asc' | 'time_desc' | 'time_asc'

export interface AlertQuery {
  limit?: number
  offset?: number
  min_risk?: number
  attack_type?: string
  entity_id?: string
  /** Alert budget as a percentage of the stream, 0.1-5. Drives the threshold. */
  budget_pct?: number
  sort?: AlertSort
}

export interface AlertsResponse {
  alerts: ScoredEvent[]
  total: number
}

export interface EntitySummary {
  entity_id: string
  entity_type: string
  cohort: string
  event_count: number
  alert_count: number
  first_seen: string
  last_seen: string
  cold_start: boolean
  mean_risk: number
  max_risk: number
}

export interface EntityListResponse {
  entities: EntitySummary[]
  total: number
}

export type AlertDetail = ScoredEvent & {
  entity_summary: EntitySummary
  similar_alerts: ScoredEvent[]
}

/* ---------------------------------------------------------------- entities */

export interface ProfileSummaryRow {
  label: string
  value: string
  cohort_value: string
}

export interface RiskTimelinePoint {
  timestamp: string
  risk_score: number
  is_alert: boolean
}

export interface TopResource {
  resource: string
  count: number
  is_new: boolean
}

export interface PeerComparisonAxis {
  axis: string
  entity: number
  cohort_median: number
}

export interface DriftState {
  drifting: boolean
  detected_at: string | null
  adapted: boolean
}

export interface EntityDetail {
  entity_id: string
  entity_type: string
  cohort: string
  first_seen: string
  last_seen: string
  event_count: number
  cold_start: boolean
  profile_summary: ProfileSummaryRow[]
  risk_timeline: RiskTimelinePoint[]
  /** 24 buckets, index = hour of day. */
  activity_by_hour: number[]
  top_resources: TopResource[]
  peer_comparison: PeerComparisonAxis[]
  drift_state: DriftState
}

/* ---------------------------------------------------------------- feedback */

export interface FeedbackRequest {
  event_id: string
  verdict: Verdict
  note?: string
}

export interface FeedbackResponse {
  ok: boolean
  updated_threshold?: number
}

/* ----------------------------------------------------------------- metrics */

export interface BudgetCurvePoint {
  budget_pct: number
  precision: number
  recall: number
  alerts: number
  analyst_hours: number
}

export interface PerAttackRecall {
  attack_type: string
  recall: number
  support: number
  detected: number
}

export interface ConfusionMatrix {
  labels: string[]
  matrix: number[][]
}

export interface PrCurvePoint {
  recall: number
  precision: number
}

export interface MttdRow {
  attack_type: string
  mean_events: number
  mean_minutes: number
}

export interface SubgroupScore {
  precision: number
  recall: number
}

export interface LatencyMs {
  p50: number
  p95: number
  p99: number
  mean: number
}

export interface AblationRow {
  variant: string
  pr_auc: number
  precision_at_1pct: number
}

export interface MetricsResponse {
  pr_auc: number
  roc_auc: number
  budget_curve: BudgetCurvePoint[]
  per_attack_recall: PerAttackRecall[]
  confusion_matrix: ConfusionMatrix
  pr_curve: PrCurvePoint[]
  fp_rate_confounders: number
  fp_rate_insider_drift: number
  mttd: MttdRow[]
  cold_start: SubgroupScore
  post_drift: SubgroupScore
  latency_ms: LatencyMs
  ablation: AblationRow[]
}

/* ------------------------------------------------------------------ stream */

export type StreamAction = 'start' | 'pause' | 'reset'

export interface StreamControlRequest {
  action: StreamAction
  speed?: number
}

export interface StreamControlResponse {
  ok: boolean
  state: string
}

export interface StreamHandlers {
  onAlert?: (alert: ScoredEvent) => void
  onStats?: (stats: StatsResponse) => void
  onHeartbeat?: () => void
  onError?: (error: Error) => void
  onOpen?: () => void
}

/** Returned by `subscribeStream`; call to tear the connection down. */
export type StreamUnsubscribe = () => void

/* --------------------------------------------------------------------- api */

export interface SentinelApi {
  readonly mode: 'mock' | 'live'
  getHealth(signal?: AbortSignal): Promise<HealthResponse>
  getStats(signal?: AbortSignal): Promise<StatsResponse>
  score(event: Record<string, unknown>, signal?: AbortSignal): Promise<ScoredEvent>
  getAlerts(query: AlertQuery, signal?: AbortSignal): Promise<AlertsResponse>
  getAlert(eventId: string, signal?: AbortSignal): Promise<AlertDetail>
  listEntities(params: { limit?: number; offset?: number; sort?: string }, signal?: AbortSignal): Promise<EntityListResponse>
  getEntity(entityId: string, signal?: AbortSignal): Promise<EntityDetail>
  sendFeedback(body: FeedbackRequest, signal?: AbortSignal): Promise<FeedbackResponse>
  getMetrics(signal?: AbortSignal): Promise<MetricsResponse>
  subscribeStream(handlers: StreamHandlers): StreamUnsubscribe
  controlStream(body: StreamControlRequest, signal?: AbortSignal): Promise<StreamControlResponse>
}
