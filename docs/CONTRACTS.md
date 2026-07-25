# SENTINEL - cross-component contracts

This is the binding interface between every part of the system. If a component
needs something different from what is written here, this document changes
first, then the code.

Enforcement is mechanical wherever possible:

| Contract | Enforced by |
| --- | --- |
| Event / label schema | `src/sentinel/schema.py`, `tests/test_schema.py` |
| Labels never in the events file | `sentinel.io.read_events` / `write_events` (raise `LabelLeakageError`) |
| Config keys | `src/sentinel/config.py` (unknown key -> `ConfigError`), `tests/test_config.py` |
| REST/SSE payload shapes | `src/sentinel/serving/models.py`, `tests/test_api_contract.py` |

---

## 1. Data schema

### 1.1 `events.parquet` - the observable event

Column order below is authoritative (`sentinel.schema.EVENT_FIELDS`).

| # | Field | Arrow type | Null? | Notes |
| --- | --- | --- | --- | --- |
| 1 | `event_id` | `string` | no | 16 lowercase hex chars, deterministic from the run seed. **Extension.** |
| 2 | `episode_id` | `string` | yes | Groups events of one injected episode. `null` for baseline. **Extension.** |
| 3 | `entity_id` | `string` | no | `usr_0001` / `svc_0001` / `dev_0001` |
| 4 | `entity_type` | `string` | no | `user` \| `service_account` \| `edge_device` |
| 5 | `cohort` | `string` | no | Role cohort, e.g. `finance_analyst`, `plc_gateway` |
| 6 | `timestamp` | `timestamp[us, tz=UTC]` | no | Always timezone-aware UTC |
| 7 | `source_ip` | `string` | no | |
| 8 | `geo_country` | `string` | no | ISO 3166-1 alpha-2 |
| 9 | `geo_city` | `string` | no | |
| 10 | `geo_lat` | `double` | no | -90..90 |
| 11 | `geo_lon` | `double` | no | -180..180 |
| 12 | `resource_accessed` | `string` | no | `fs:/finance/q3.xlsx` \| `api:/v1/payments` \| `port:445` \| `func:valve_setpoint` |
| 13 | `resource_type` | `string` | no | `file` \| `endpoint` \| `port` \| `device_function` |
| 14 | `auth_method` | `string` | no | `password` \| `token` \| `certificate` \| `biometric` \| `mfa_push` |
| 15 | `auth_result` | `string` | no | `success` \| `failure`. **Extension.** |
| 16 | `session_duration_s` | `double` | no | >= 0 |
| 17 | `command_sequence` | `list<string>` | no | Ordered actions; **empty list** (never null) for non-privileged sessions |
| 18 | `device_os` | `string` | no | |
| 19 | `device_os_version` | `string` | no | |
| 20 | `device_mac` | `string` | no | |
| 21 | `device_protocol` | `string` | no | `https` \| `ssh` \| `rdp` \| `smb` \| `modbus` \| `mqtt` |
| 22 | `device_fingerprint` | `string` | no | Stable hash of `os\|version\|mac\|protocol` |
| 23 | `bytes_transferred` | `int64` | no | >= 0. **Extension.** |
| 24 | `split` | `string` | no | `train` \| `val` \| `test` |

### 1.2 Why the four extensions exist

These are not conveniences. Without them, two of the six required attack
families cannot be represented in the data at all, and the other four cannot be
evaluated honestly.

- **`auth_result`** - a brute-force or credential-stuffing campaign *is* a burst
  of authentication **failures**. A log that records only successful accesses
  cannot express "40 failed logins then one success". Every failure-rate
  feature, and two of the six attack types, depend on this column.
- **`bytes_transferred`** - exfiltration is defined by volume. Without a size,
  `low_and_slow_exfil` is indistinguishable from ordinary file access, and the
  "low and slow" characteristic (small transfers, sustained, off-hours) has
  nothing to measure.
- **`event_id`** - a stable primary key. It is what joins `events.parquet` to
  `labels.parquet`, what the API addresses alerts by, and what makes the corpus
  reproducible from the seed.
- **`episode_id`** - attacks are multi-event campaigns, not lone events.
  Episode grouping is what makes episode-level recall and mean-time-to-detect
  computable. It is deliberately **not** a leak: benign confounders and
  `insider_drift` also carry episode ids, so "has an episode_id" says nothing
  about maliciousness.

### 1.3 `labels.parquet` - ground truth, stored separately

| Field | Arrow type | Null? | Notes |
| --- | --- | --- | --- |
| `event_id` | `string` | no | Join key into `events.parquet` |
| `episode_id` | `string` | yes | Mirrors the event's episode |
| `label` | `string` | no | One of `ALL_LABELS` |
| `is_anomaly` | `bool` | no | **True only for the six `ATTACK_TYPES`** |
| `confounder_type` | `string` | yes | One of `CONFOUNDER_TYPES` when `label == "benign_confounder"` |
| `attack_stage` | `string` | yes | `recon` \| `initial_access` \| `escalation` \| `action_on_objective` |

Labels live in their own file so that "ground truth is hidden at inference
time" is a property of the file layout rather than of developer discipline.
`sentinel.io.read_events` raises `LabelLeakageError` if it finds `label`,
`is_anomaly`, `confounder_type` or `attack_stage` in an events file, and
`write_events` refuses to write them.

### 1.4 Label vocabulary and the anomaly rule

```
ALL_LABELS = ["normal"] + ATTACK_TYPES + EDGE_CASE_TYPES + ["benign_confounder"]

ATTACK_TYPES     = brute_force, impossible_travel, credential_stuffing,
                   lateral_movement, device_spoofing, low_and_slow_exfil
EDGE_CASE_TYPES  = insider_drift
CONFOUNDER_TYPES = legit_travel, new_device_enrollment, password_rotation,
                   vacation_return, maintenance_burst
```

> **Never write `label != "normal"`.** Use `sentinel.schema.is_anomalous_label`.

| Label | `is_anomaly` | Alerting on it counts as |
| --- | --- | --- |
| the six `ATTACK_TYPES` | `True` | true positive |
| `normal` | `False` | false positive |
| `benign_confounder` | `False` | false positive |
| `insider_drift` | `False` | false positive |

`insider_drift` is the ambiguous case on purpose: an employee who genuinely
changes role looks anomalous and *is* legitimate. A brief alert while the
baseline is stale is acceptable; still alerting a week later is a failure of the
drift-adaptation layer. It is scored as a negative so that
`fp_rate_insider_drift` measures exactly that failure.

### 1.5 Risk bands

Half-open intervals, so boundaries are unambiguous:

| Band | Range |
| --- | --- |
| `low` | `[0, 40)` |
| `medium` | `[40, 65)` |
| `high` | `[65, 85)` |
| `critical` | `[85, inf)` |

Scores below 0 clamp to `low`; above 100 clamp to `critical`. Use
`sentinel.schema.risk_band(score)`. The cut points are duplicated in
`configs/model.yaml` under `risk_thresholds` and the config loader asserts they
match the code.

### 1.6 Detector names

`profile`, `isolation`, `sequence`, `graph`, `gru` - in that order. Used as the
keys of `fusion.weights` in `configs/model.yaml`, of the model registry, and of
`detector_scores` on the wire. `gru` is optional and is `null` when torch is
absent; the remaining weights are renormalised so scores stay comparable.

---

## 2. File layout

```
configs/
  data.yaml               generator config   -> sentinel.config.load_data_config()
  model.yaml              detector config    -> sentinel.config.load_model_config()

data/                     (git-ignored) produced by `sentinel gen`
  events.parquet          EVENT_FIELDS, all splits, sorted by timestamp
  labels.parquet          LABEL_FIELDS, one row per event
  entities.json           entity catalog: id, type, cohort, home geo, cold_start flag
  generation_report.json  realised counts: events/split, episodes/attack type, rates

artifacts/                (git-ignored) produced by `sentinel train`
  profiler.joblib         per-entity + cohort statistics
  isolation_forest.joblib
  sequence_model.joblib   Markov / n-gram transition tables
  graph_detector.joblib
  gru_autoencoder.pt      only when torch is installed
  fusion.json             weights actually used, bias, clip, calibration curve
  calibration.joblib      isotonic mapping from fused logit to risk 0-100
  classifier.joblib       HistGBM attack-type classifier
  thresholds.json         risk threshold per alert budget (0.5% / 1% / 2%)
  metrics.json            the /api/metrics payload, verbatim

reports/                  produced by `sentinel eval` (PNG/PDF git-ignored)
  REPORT.md, pr_curve.png, budget_curve.png, confusion_matrix.png, ...
```

Everything under `data/` and `artifacts/` is reproducible from
`configs/data.yaml` plus its `seed`, and is therefore git-ignored.

---

## 3. REST / SSE API contract

All endpoints are served under the prefix `/api`.

- `GET  /api/health` -> `{status, version, model_loaded, torch_available}`
- `GET  /api/stats` -> `{events_processed, alerts_raised, alerts_by_type: {}, events_per_sec, uptime_s, stream_position, stream_total}`
- `POST /api/score` body = raw event object -> `ScoredEvent`
- `GET  /api/alerts?limit&offset&min_risk&attack_type&entity_id&budget_pct&sort` -> `{alerts: ScoredEvent[], total}`
- `GET  /api/alerts/{event_id}` -> `ScoredEvent` plus `{entity_summary, similar_alerts: ScoredEvent[]}`
- `GET  /api/entities/{entity_id}` -> `{entity_id, entity_type, cohort, first_seen, last_seen, event_count, cold_start, profile_summary: {label, value, cohort_value}[], risk_timeline: {timestamp, risk_score, is_alert}[], activity_by_hour: number[24], top_resources: {resource, count, is_new}[], peer_comparison: {axis, entity, cohort_median}[], drift_state: {drifting, detected_at, adapted}}`
- `POST /api/feedback` body `{event_id, verdict: "true_positive"|"false_positive"|"escalate", note?}` -> `{ok, updated_threshold?}`
- `GET  /api/metrics` -> `{pr_auc, roc_auc, budget_curve: {budget_pct, precision, recall, alerts, analyst_hours}[], per_attack_recall: {attack_type, recall, support, detected}[], confusion_matrix: {labels: string[], matrix: number[][]}, pr_curve: {recall, precision}[], fp_rate_confounders, fp_rate_insider_drift, mttd: {attack_type, mean_events, mean_minutes}[], cold_start: {precision, recall}, post_drift: {precision, recall}, latency_ms: {p50, p95, p99, mean}, ablation: {variant, pr_auc, precision_at_1pct}[]}`
- `GET  /api/stream` -> SSE. Events: `event: alert` with `data` = `ScoredEvent`; `event: stats` with `data` = the `/api/stats` payload; `event: heartbeat`.
- `POST /api/stream/control` body `{action: "start"|"pause"|"reset", speed?: number}` -> `{ok, state}`

`ScoredEvent` JSON shape:

```
{
  event_id, entity_id, entity_type, cohort, timestamp,
  risk_score,            // float 0-100
  risk_band,             // low | medium | high | critical
  is_alert,              // bool, risk above current budget threshold
  predicted_attack_type, // one of ATTACK_TYPES, "normal", or "unknown_novel"
  attack_type_confidence,// 0-1
  classifier_agreement,  // bool: HistGBM and signature matcher agree
  is_novel,              // bool
  detector_scores: { profile, isolation, sequence, graph, gru },   // 0-1 normalised p-value based, gru nullable
  contributions: [ { feature, display_name, value, display_value, contribution, direction, description } ],
  narrative,             // human-readable sentence(s) for the SOC analyst
  counterfactuals: [ { feature, display_name, neutralised_risk, delta } ],
  cold_start,            // bool
  entity_event_count,    // int
  event: { ...all raw AccessEvent fields... },
  ground_truth: { label, is_anomaly } | null   // present only in demo/eval mode
}
```

### 3.1 Points the prose above leaves open

Resolved here so the serving and dashboard agents cannot diverge. All of these
are encoded in `src/sentinel/serving/models.py`.

| Item | Decision |
| --- | --- |
| Timestamp format | ISO-8601, UTC, `Z` suffix: `2026-02-14T09:14:22Z`. Identical at the top level and inside `event`. |
| `contributions[].direction` | `"increases"` or `"decreases"`. |
| `contributions[].value` | `number \| string \| boolean \| null` (a contribution may be over a categorical feature such as `geo_country`); `display_value` is always a preformatted string and is what the UI renders. |
| `contributions[].contribution` | Signed **log-odds** term, not a 0-100 delta. The terms sum exactly to the fused logit. For a 0-100 view, use `counterfactuals[].delta`. |
| `counterfactuals[].delta` | `neutralised_risk - risk_score`, so a factor that drove the alert has a **negative** delta. |
| `sort` query values | `risk_desc` (default), `risk_asc`, `time_desc`, `time_asc`. |
| `limit` / `offset` | `limit` defaults to 50, max 500; `offset` defaults to 0. `total` is the match count **before** limit/offset. |
| `budget_pct` query | One of `0.5`, `1.0`, `2.0`; selects the threshold that decides `is_alert`. Defaults to `alerting.default_budget_pct`. |
| `entity_summary` (alert detail) | `{entity_id, entity_type, cohort, event_count, cold_start, first_seen, last_seen, alert_count, mean_risk}`. |
| `stream/control` -> `state` | `"running"`, `"paused"` or `"stopped"`. `reset` returns `"stopped"`. |
| `activity_by_hour` | Exactly 24 numbers, index = UTC hour. |
| `health.status` | `"ok"` or `"degraded"` (`degraded` = serving without a trained model). |
| Unknown fields | Every model sets `extra="forbid"`. Sending an undocumented field is a 422, not a silent ignore. |
| `ground_truth` | Always present as a key; `null` outside demo/eval mode. |

### 3.2 SSE framing

```
event: alert
data: {...ScoredEvent...}

event: stats
data: {...StatsResponse...}

event: heartbeat
data: {}
```

Heartbeat cadence is `serving.sse_heartbeat_s` (default 15s) so proxies do not
close an idle stream.

---

## 4. Python entry points other phases depend on

```python
from sentinel.schema import (
    ALL_LABELS,
    ATTACK_TYPES,
    ATTACK_STAGES,
    AUTH_METHODS,
    AUTH_RESULTS,
    CONFOUNDER_TYPES,
    DETECTOR_NAMES,
    EDGE_CASE_TYPES,
    ENTITY_TYPES,
    EVENT_FIELDS,
    LABEL_FIELDS,
    PROTOCOLS,
    RESOURCE_TYPES,
    RISK_BANDS,
    SPLITS,
    AccessEvent,
    Label,
    events_arrow_schema,
    labels_arrow_schema,
    is_anomalous_label,
    risk_band,
)
from sentinel.io import read_events, write_events, read_labels, write_labels
from sentinel.config import load_data_config, load_model_config
from sentinel.serving.models import ScoredEvent, MetricsResponse  # ... etc
```

CLI surface (flags are stable; bodies are filled in by later phases):

```
sentinel gen     --config configs/data.yaml [--out DIR] [--seed N]
sentinel train   --config configs/model.yaml [--data-config ...] [--data-dir ...] [--artifacts ...]
sentinel eval    --config configs/model.yaml [--data-dir ...] [--artifacts ...] [--reports ...] [--split test]
sentinel serve   --config configs/model.yaml [--artifacts ...] [--host ...] [--port 8000] [--reload]
sentinel replay  --config configs/model.yaml [--data-dir ...] [--api URL] [--speed 50] [--split test]
sentinel ablate  --config configs/model.yaml [--artifacts ...] [--reports ...] [--holdout-attack TYPE]
```
