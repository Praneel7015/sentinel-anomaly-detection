/** Static catalogs backing the mock generator: geography, resources, and the feature dictionary. */

export interface City {
  city: string
  country: string
  lat: number
  lon: number
}

export const HOME_CITIES: City[] = [
  { city: 'London', country: 'GB', lat: 51.51, lon: -0.13 },
  { city: 'Frankfurt', country: 'DE', lat: 50.11, lon: 8.68 },
  { city: 'New York', country: 'US', lat: 40.71, lon: -74.01 },
  { city: 'Chicago', country: 'US', lat: 41.88, lon: -87.63 },
  { city: 'Austin', country: 'US', lat: 30.27, lon: -97.74 },
  { city: 'Bengaluru', country: 'IN', lat: 12.97, lon: 77.59 },
  { city: 'Singapore', country: 'SG', lat: 1.35, lon: 103.82 },
  { city: 'Sydney', country: 'AU', lat: -33.87, lon: 151.21 },
  { city: 'Toronto', country: 'CA', lat: 43.65, lon: -79.38 },
  { city: 'Dublin', country: 'IE', lat: 53.35, lon: -6.26 },
  { city: 'Warsaw', country: 'PL', lat: 52.23, lon: 21.01 },
  { city: 'Tel Aviv', country: 'IL', lat: 32.09, lon: 34.78 },
  { city: 'Tokyo', country: 'JP', lat: 35.68, lon: 139.69 },
  { city: 'São Paulo', country: 'BR', lat: -23.55, lon: -46.63 },
]

/** Geographies that read as suspicious for the impossible-travel and stuffing injectors. */
export const HOSTILE_CITIES: City[] = [
  { city: 'Bucharest', country: 'RO', lat: 44.43, lon: 26.1 },
  { city: 'Kyiv', country: 'UA', lat: 50.45, lon: 30.52 },
  { city: 'Karachi', country: 'PK', lat: 24.86, lon: 67.01 },
  { city: 'Guangzhou', country: 'CN', lat: 23.13, lon: 113.26 },
  { city: 'Lagos', country: 'NG', lat: 6.52, lon: 3.38 },
  { city: 'Chisinau', country: 'MD', lat: 47.01, lon: 28.86 },
]

export const COHORT_CATALOG: Record<string, { entityType: string; resources: string[] }> = {
  finance: {
    entityType: 'user',
    resources: [
      'erp/gl-postings',
      'erp/accounts-payable',
      'reports/quarterly-close',
      'sharepoint/finance-shared',
      'treasury/payment-runs',
      'bi/revenue-cube',
    ],
  },
  engineering: {
    entityType: 'user',
    resources: [
      'git/monorepo',
      'git/infra-terraform',
      'k8s/prod-cluster',
      'artifactory/releases',
      'jira/platform',
      'grafana/prod-dashboards',
    ],
  },
  sales: {
    entityType: 'user',
    resources: [
      'crm/opportunities',
      'crm/accounts',
      'sharepoint/sales-collateral',
      'bi/pipeline-cube',
      'docusign/contracts',
    ],
  },
  hr: {
    entityType: 'user',
    resources: [
      'workday/employee-records',
      'workday/compensation',
      'sharepoint/hr-policies',
      'ats/candidates',
      'payroll/runs',
    ],
  },
  it_ops: {
    entityType: 'user',
    resources: [
      'vcenter/prod',
      'ad/domain-controllers',
      'jump/bastion-01',
      'vault/secrets-prod',
      'sccm/deployments',
      'netbox/inventory',
    ],
  },
  exec: {
    entityType: 'user',
    resources: [
      'boardroom/deck-archive',
      'bi/exec-scorecard',
      'sharepoint/ma-diligence',
      'reports/quarterly-close',
    ],
  },
  ci_runner: {
    entityType: 'service_account',
    resources: [
      'artifactory/releases',
      'git/monorepo',
      'k8s/staging-cluster',
      'registry/container-images',
      's3/build-cache',
    ],
  },
  backup_svc: {
    entityType: 'service_account',
    resources: [
      's3/nightly-snapshots',
      'nas/archive-tier',
      'db/replica-readonly',
      'vault/backup-keys',
    ],
  },
  iot_sensor: {
    entityType: 'edge_device',
    resources: ['mqtt/telemetry-ingest', 'ota/firmware-channel', 'api/device-registry'],
  },
  pos_terminal: {
    entityType: 'edge_device',
    resources: ['payments/authorise', 'inventory/stock-sync', 'ota/firmware-channel'],
  },
}

export const COHORT_NAMES = Object.keys(COHORT_CATALOG)

/** Resources that are off-limits for most cohorts — used by the lateral-movement injector. */
export const SENSITIVE_RESOURCES = [
  'ad/domain-controllers',
  'vault/secrets-prod',
  'workday/compensation',
  'treasury/payment-runs',
  'sharepoint/ma-diligence',
  'db/prod-customer-pii',
  'jump/bastion-01',
  'k8s/prod-cluster',
]

export const ACTIONS = [
  'login',
  'logout',
  'file_read',
  'file_write',
  'file_download',
  'api_call',
  'db_query',
  'ssh_connect',
  'rdp_session',
  'config_change',
  'vpn_connect',
  'share_mount',
]

export const DEVICE_OS = ['Windows 11 23H2', 'macOS 14.5', 'Ubuntu 22.04', 'iOS 17.5', 'Android 14']

export const USER_AGENTS = [
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0.0.0',
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) Safari/17.5',
  'Mozilla/5.0 (X11; Linux x86_64) Firefox/127.0',
  'python-requests/2.32.3',
  'curl/8.7.1',
  'SentinelAgent/2.1 (device-telemetry)',
]

export const FIRST_NAMES = [
  'aisha', 'marcus', 'lena', 'priya', 'tomas', 'nadia', 'kofi', 'yuki', 'oscar', 'freya',
  'dmitri', 'sofia', 'hassan', 'clara', 'noah', 'imani', 'viktor', 'mei', 'jonas', 'ruth',
]

export const LAST_NAMES = [
  'okafor', 'lindqvist', 'moreau', 'nakamura', 'silva', 'kowalski', 'haddad', 'osei', 'novak',
  'bergstrom', 'ferreira', 'ibrahim', 'castillo', 'dubois', 'petrov', 'ashworth', 'mbeki', 'tan',
  'kaur', 'ellis',
]

/* ------------------------------------------------------------ feature dictionary */

export interface FeatureSpec {
  feature: string
  display_name: string
  /** Renders the numeric value the way an analyst expects to read it. */
  format: (value: number) => string
  /** Plain-English explanation shown on hover in the waterfall. */
  describe: (displayValue: string) => string
  /** [typical, extreme] — intensity 0..1 interpolates between them. */
  range: [number, number]
}

const int = (v: number) => String(Math.round(v))
const one = (v: number) => v.toFixed(1)
const two = (v: number) => v.toFixed(2)

export const FEATURES: Record<string, FeatureSpec> = {
  hour_surprisal: {
    feature: 'hour_surprisal',
    display_name: 'Time-of-day surprisal',
    format: (v) => `${one(v)} bits`,
    describe: (d) =>
      `Negative log-likelihood of this timestamp under the entity's circadian model (${d}). Above ~4 bits means the entity has effectively never been active at this hour.`,
    range: [0.6, 9.2],
  },
  geo_velocity: {
    feature: 'geo_velocity',
    display_name: 'Geo velocity',
    format: (v) => `${int(v)} km/h`,
    describe: (d) =>
      `Implied travel speed between this login and the previous one (${d}). Commercial aviation tops out near 900 km/h.`,
    range: [12, 3400],
  },
  resource_novelty: {
    feature: 'resource_novelty',
    display_name: 'Resource novelty',
    format: (v) => two(v),
    describe: (d) =>
      `Probability that this entity has never touched this resource before, shrunk toward its cohort prior (${d}).`,
    range: [0.05, 0.98],
  },
  resource_breadth: {
    feature: 'resource_breadth',
    display_name: 'Resource breadth (1h)',
    format: (v) => `${int(v)} distinct`,
    describe: (d) =>
      `Distinct resources touched in the trailing hour (${d}) versus a baseline of 2-4 for this cohort.`,
    range: [2, 31],
  },
  auth_fail_entity: {
    feature: 'auth_fail_entity',
    display_name: 'Auth failures (entity, 15m)',
    format: int,
    describe: (d) =>
      `Failed authentications for this entity in the trailing 15 minutes (${d}). Rolling counter, past events only.`,
    range: [0, 74],
  },
  auth_fail_ip: {
    feature: 'auth_fail_ip',
    display_name: 'Auth failures (source IP, 15m)',
    format: int,
    describe: (d) =>
      `Failed authentications from this source IP across all accounts (${d}) — the signal that separates stuffing from a single forgotten password.`,
    range: [0, 260],
  },
  device_mismatch: {
    feature: 'device_mismatch',
    display_name: 'Device fingerprint mismatch',
    format: two,
    describe: (d) =>
      `Jaccard distance between the presented device fingerprint and the entity's enrolled devices (${d}).`,
    range: [0.02, 0.96],
  },
  sequence_surprisal: {
    feature: 'sequence_surprisal',
    display_name: 'Action-sequence surprisal',
    format: (v) => `${one(v)} bits`,
    describe: (d) =>
      `Markov surprisal of the last five actions under the entity's transition model (${d}).`,
    range: [1.1, 12.4],
  },
  offhours_bytes: {
    feature: 'offhours_bytes',
    display_name: 'Cumulative off-hours egress',
    format: (v) => (v > 1024 ? `${(v / 1024).toFixed(2)} GB` : `${int(v)} MB`),
    describe: (d) =>
      `Bytes transferred outside the entity's active window, accumulated over 24h (${d}). Catches exfiltration kept under per-event thresholds.`,
    range: [4, 7400],
  },
  peer_deviation: {
    feature: 'peer_deviation',
    display_name: 'Peer deviation (cohort)',
    format: (v) => `${one(v)}σ`,
    describe: (d) =>
      `Standard deviations from the cohort median on the combined behaviour vector (${d}).`,
    range: [0.3, 6.1],
  },
  cohort_novelty: {
    feature: 'cohort_novelty',
    display_name: 'Cohort-relative access novelty',
    format: two,
    describe: (d) =>
      `How unusual this resource is for the entity's whole cohort (${d}), not just the entity. High values indicate movement outside the role's access graph.`,
    range: [0.03, 0.95],
  },
  distinct_ips: {
    feature: 'distinct_ips',
    display_name: 'Distinct source IPs (24h)',
    format: int,
    describe: (d) => `Unique source addresses used by this entity in the last 24 hours (${d}).`,
    range: [1, 19],
  },
  device_age: {
    feature: 'device_age',
    display_name: 'Device enrolment age',
    format: (v) => (v < 1 ? `${int(v * 24)} h` : `${int(v)} d`),
    describe: (d) =>
      `How long the presenting device has been enrolled (${d}). Brand-new devices are weakly suspicious on their own.`,
    range: [0.02, 400],
  },
  session_duration: {
    feature: 'session_duration',
    display_name: 'Session duration',
    format: (v) => `${int(v)} min`,
    describe: (d) => `Length of the containing session (${d}) against the entity's usual span.`,
    range: [3, 420],
  },
  privilege_delta: {
    feature: 'privilege_delta',
    display_name: 'Privilege delta',
    format: two,
    describe: (d) =>
      `Gap between the privilege level required by this action and the entity's habitual level (${d}).`,
    range: [0.0, 0.92],
  },
}

/** Features that pull risk *down* — shown as the negative side of the waterfall. */
export const MITIGATING_FEATURES = [
  'device_age',
  'session_duration',
  'distinct_ips',
] as const

export interface AttackSignature {
  /** Features the injector reliably lights up. */
  primary: string[]
  /** Supporting features, sampled. */
  secondary: string[]
  /** Relative weight per detector, before noise. */
  detectors: { profile: number; isolation: number; sequence: number; graph: number; gru: number }
}

export const ATTACK_SIGNATURES: Record<string, AttackSignature> = {
  brute_force: {
    primary: ['auth_fail_entity', 'auth_fail_ip'],
    secondary: ['hour_surprisal', 'distinct_ips', 'sequence_surprisal'],
    detectors: { profile: 0.86, isolation: 0.79, sequence: 0.71, graph: 0.24, gru: 0.68 },
  },
  impossible_travel: {
    primary: ['geo_velocity', 'device_mismatch'],
    secondary: ['hour_surprisal', 'distinct_ips', 'peer_deviation'],
    detectors: { profile: 0.91, isolation: 0.74, sequence: 0.33, graph: 0.29, gru: 0.52 },
  },
  credential_stuffing: {
    primary: ['auth_fail_ip', 'distinct_ips'],
    secondary: ['auth_fail_entity', 'device_mismatch', 'hour_surprisal'],
    detectors: { profile: 0.62, isolation: 0.88, sequence: 0.58, graph: 0.77, gru: 0.61 },
  },
  lateral_movement: {
    primary: ['cohort_novelty', 'resource_breadth', 'privilege_delta'],
    secondary: ['resource_novelty', 'peer_deviation', 'sequence_surprisal'],
    detectors: { profile: 0.58, isolation: 0.66, sequence: 0.81, graph: 0.94, gru: 0.72 },
  },
  device_spoofing: {
    primary: ['device_mismatch', 'device_age'],
    secondary: ['geo_velocity', 'hour_surprisal', 'distinct_ips'],
    detectors: { profile: 0.77, isolation: 0.83, sequence: 0.41, graph: 0.36, gru: 0.59 },
  },
  low_and_slow_exfil: {
    primary: ['offhours_bytes', 'hour_surprisal'],
    secondary: ['resource_breadth', 'peer_deviation', 'session_duration'],
    detectors: { profile: 0.69, isolation: 0.55, sequence: 0.87, graph: 0.48, gru: 0.9 },
  },
  insider_drift: {
    primary: ['peer_deviation', 'resource_novelty'],
    secondary: ['cohort_novelty', 'hour_surprisal', 'resource_breadth'],
    detectors: { profile: 0.64, isolation: 0.46, sequence: 0.39, graph: 0.52, gru: 0.44 },
  },
  normal: {
    primary: ['hour_surprisal', 'resource_novelty'],
    secondary: ['peer_deviation', 'sequence_surprisal', 'resource_breadth', 'distinct_ips'],
    detectors: { profile: 0.18, isolation: 0.22, sequence: 0.15, graph: 0.12, gru: 0.19 },
  },
}

/** Benign patterns the generator injects to make precision genuinely hard to win. */
export const CONFOUNDERS = [
  { kind: 'benign_travel', mimics: 'impossible_travel', label: 'Approved business travel' },
  { kind: 'new_device_enrolment', mimics: 'device_spoofing', label: 'New device enrolment' },
  { kind: 'password_rotation', mimics: 'brute_force', label: 'Password rotation retries' },
  { kind: 'vacation_return', mimics: 'insider_drift', label: 'Return from extended leave' },
  { kind: 'maintenance_burst', mimics: 'lateral_movement', label: 'Maintenance-window burst' },
] as const

export type ConfounderKind = (typeof CONFOUNDERS)[number]['kind']
