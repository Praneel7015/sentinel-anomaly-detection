# sentinel.features — Incremental Feature Pipeline

Leakage-free, past-only feature extraction shared by the batch training path
and the live streaming path.  Both paths run the **same code**; there is no
separate vectorised implementation.

---

## Quick-start

```python
from sentinel.features import batch_features, stream_features, FEATURE_NAMES

# Batch (training / evaluation)
feature_df = batch_features(events_df)          # → pd.DataFrame, columns = FEATURE_NAMES

# Streaming (serving)
for event, fv in stream_features(events_iter):
    arr = fv.to_numpy()                         # shape (35,) float32
    score = my_model.predict(arr)
```

---

## Module layout

| File | Responsibility |
|------|---------------|
| `state.py` | `EntityState`, `GlobalState`, `SourceIPState` — incremental O(1) state |
| `cohort.py` | `CohortState`, Bayesian shrinkage helpers, cold-start gate |
| `graph.py` | `GraphState` — bipartite entity-resource graph degree tracking |
| `sequence.py` | `SequenceModel`, Markov surprisal, GRU tensor encoding |
| `extractors.py` | Feature extractor registry, `FeatureVector`, `extract_all` |
| `pipeline.py` | `stream_features`, `batch_features`, `PipelineState` |
| `__init__.py` | Public API re-exports |

---

## Feature names and meanings

`FEATURE_NAMES` (in stable registry order):

### Temporal features

| # | Name | Meaning | Direction |
|---|------|---------|-----------|
| 1 | `hour_surprisal` | Negative log-2 probability of this UTC hour under the entity's shrunk circadian histogram (bits). High = unusual hour. | ↑ risk |
| 2 | `weekday_surprisal` | Negative log-2 probability of this day-of-week under the entity's shrunk weekday histogram (bits). | ↑ risk |
| 3 | `is_offhours` | 1.0 if event occurred outside 08:00–18:00 UTC, else 0.0. | ↑ risk |
| 4 | `inter_arrival_zscore` | Z-score of the gap since the last event vs the entity's inter-arrival history (Welford). | ↑ risk |
| 5 | `burstiness` | Coefficient of variation (std/mean) of inter-arrival times; high = bursty. | ↑ risk |

### Geo / Network features

| # | Name | Meaning | Direction |
|---|------|---------|-----------|
| 6 | `geo_velocity_kmh` | Implied travel speed (km/h) between this event's location and the previous one. Physically impossible values signal impossible-travel attacks. | ↑ risk |
| 7 | `geo_centroid_distance_km` | Haversine distance (km) from this location to the entity's running geo centroid (home base). | ↑ risk |
| 8 | `new_country_flag` | 1.0 if the `geo_country` has never been seen for this entity, else 0.0. | ↑ risk |
| 9 | `new_subnet_flag` | 1.0 if the source IP's /24 subnet has never been seen for this entity, else 0.0. | ↑ risk |
| 10 | `source_ip_novelty` | 1.0 if the exact source IP has never been seen for this entity (uses subnet_set), else 0.0. | ↑ risk |

### Resource features

| # | Name | Meaning | Direction |
|---|------|---------|-----------|
| 11 | `entity_novel_resource` | 1.0 if this resource has never been accessed by this entity before. | ↑ risk |
| 12 | `cohort_novel_resource` | 1.0 if no peer in this cohort has ever accessed this resource (lateral movement signal). | ↑ risk |
| 13 | `distinct_resources_1h` | Count of access timestamps in the entity's rolling 1-hour window. | ↑ risk |
| 14 | `distinct_resources_24h` | Count of access timestamps in the entity's rolling 24-hour window. | ↑ risk |
| 15 | `resource_js_divergence` | Jensen-Shannon divergence between entity's resource distribution and cohort average. High = unusual resource mix. | ↑ risk |
| 16 | `resource_breadth_expansion_rate` | Distinct resources / event count. High = rapidly expanding access footprint. | ↑ risk |

### Auth features

| # | Name | Meaning | Direction |
|---|------|---------|-----------|
| 17 | `auth_method_novelty` | 1.0 if this auth method (password/token/etc.) has never been used by this entity. | ↑ risk |
| 18 | `entity_failure_count_5m` | Rolling count of auth failures by this entity in the last 5 minutes. | ↑ risk |
| 19 | `entity_failure_count_1h` | Rolling count of auth failures by this entity in the last hour. | ↑ risk |
| 20 | `entity_failure_count_24h` | Rolling count of auth failures by this entity in the last 24 hours. | ↑ risk |
| 21 | `entity_failure_ratio_1h` | Fraction of 1h-window events that were auth failures (entity). | ↑ risk |
| 22 | `ip_failure_count_5m` | Rolling failures from this source IP in the last 5 minutes (credential stuffing signal). | ↑ risk |
| 23 | `ip_failure_count_1h` | Rolling failures from this source IP in the last hour. | ↑ risk |
| 24 | `ip_failure_count_24h` | Rolling failures from this source IP in the last 24 hours. | ↑ risk |
| 25 | `success_after_failures` | 1.0 if this is a successful auth immediately after ≥ N consecutive failures (brute-force signal). | ↑ risk |
| 26 | `distinct_entities_per_ip` | Number of distinct entity IDs that have ever authenticated from this source IP. | ↑ risk |

### Device features

| # | Name | Meaning | Direction |
|---|------|---------|-----------|
| 27 | `fingerprint_unknown` | 1.0 if the device fingerprint has never been seen for this entity. | ↑ risk |
| 28 | `os_mismatch` | 1.0 if the operating system has never been seen for this entity. | ↑ risk |
| 29 | `mac_oui_change` | 1.0 if the MAC OUI (first 3 octets) has never been seen for this entity. | ↑ risk |
| 30 | `protocol_novelty` | 1.0 if the wire protocol has never been used by this entity. | ↑ risk |

### Sequence (command) features

| # | Name | Meaning | Direction |
|---|------|---------|-----------|
| 31 | `command_surprisal` | Length-normalised negative log-probability of the command sequence under the shrunk Markov model (bits/token). 0 for empty sequences. | ↑ risk |
| 32 | `unseen_bigram_count` | Number of command bigrams in this session that have never been seen for this entity. | ↑ risk |
| 33 | `sequence_length_zscore` | Z-score of command sequence length vs cohort distribution. | ↑ risk |

### Volume features

| # | Name | Meaning | Direction |
|---|------|---------|-----------|
| 34 | `bytes_zscore` | Z-score of bytes transferred vs entity EWMA. | ↑ risk |
| 35 | `offhours_bytes_rolling_7d` | Cumulative bytes transferred during off-hours over the past 7 days (low-and-slow exfil signal). | ↑ risk |
| 36 | `transfer_to_duration_ratio` | Bytes per second of session duration; anomalously high = bulk transfer. | ↑ risk |

### Graph features

| # | Name | Meaning | Direction |
|---|------|---------|-----------|
| 37 | `graph_entity_novel_resource` | 1.0 if this is the first time this entity has accessed this resource (graph edge novelty). | ↑ risk |
| 38 | `graph_peer_resource_count` | Number of cohort peers that have ever accessed this resource. Very low = lateral movement. | ↓ risk |
| 39 | `graph_jaccard_vs_cohort` | Jaccard similarity between entity's resource set and cohort centroid. Low = unusual access pattern. | ↓ risk |
| 40 | `graph_entity_degree_deviation` | Z-score of entity's distinct-resource count vs cohort mean. | ↑ risk |

### Cold-start / context features

| # | Name | Meaning | Direction |
|---|------|---------|-----------|
| 41 | `entity_event_count` | Total events seen for this entity so far (cold-start gate denominator). | neutral |
| 42 | `is_cold_start_flag` | 1.0 if entity has seen fewer than `cold_start.min_events` (default 25) events. | neutral |
| 43 | `days_observed` | Days since first event for this entity. | neutral |

---

## Cold-start trustworthiness

Features are either trustworthy for cold-start entities (`cold_start_trustworthy = True`)
or not (`False`).  Features marked `False` are dominated by Bayesian shrinkage toward
the cohort prior when the entity has few events.

**Cold-start trustworthy (flag/binary signals that work from event 1):**

`is_offhours`, `geo_velocity_kmh`, `new_country_flag`, `new_subnet_flag`,
`source_ip_novelty`, `entity_novel_resource`, `cohort_novel_resource`,
`auth_method_novelty`, `entity_failure_count_5m`, `entity_failure_count_1h`,
`entity_failure_count_24h`, `entity_failure_ratio_1h`, `ip_failure_count_5m`,
`ip_failure_count_1h`, `ip_failure_count_24h`, `success_after_failures`,
`distinct_entities_per_ip`, `fingerprint_unknown`, `os_mismatch`,
`mac_oui_change`, `protocol_novelty`, `transfer_to_duration_ratio`,
`graph_entity_novel_resource`, `graph_peer_resource_count`,
`entity_event_count`, `is_cold_start_flag`, `days_observed`

**Not cold-start trustworthy (require history for meaningful z-scores/surprisals):**

`hour_surprisal`, `weekday_surprisal`, `inter_arrival_zscore`, `burstiness`,
`geo_centroid_distance_km`, `distinct_resources_1h`, `distinct_resources_24h`,
`resource_js_divergence`, `resource_breadth_expansion_rate`,
`command_surprisal`, `unseen_bigram_count`, `sequence_length_zscore`,
`bytes_zscore`, `offhours_bytes_rolling_7d`,
`graph_jaccard_vs_cohort`, `graph_entity_degree_deviation`

The scoring layer gates cold-start entities using `is_cold_start_flag` and
widens their alert threshold by `cold_start.provisional_threshold_widening`
(default 1.25×) until `entity_event_count >= cold_start.min_events`.

---

## Batch-equals-stream guarantee

**The batch and stream paths are identical by construction.** `batch_features`
sorts the input DataFrame by timestamp, then drives the exact same
`stream_features` generator row-by-row.  There is no separate vectorised
implementation.

Leakage is structurally impossible because:

1. `stream_features` yields `(event, FeatureVector)` and only then calls
   `state.update(event)` — state is updated **after** features are emitted.
2. `batch_features` inherits this ordering by driving `stream_features`.
3. A unit test (`TestBatchEqualsStream.test_identical_outputs`) asserts
   bit-exact equality between the stream and batch outputs for every row.

A second test (`TestLeakageFree.test_appending_future_does_not_change_past`)
verifies that appending future events does not alter the features of past
events in either the stream or batch path.

---

## Bayesian shrinkage

For any scalar statistic `s`, the shrunk value is:

```
shrunk = w * entity_s + (1 - w) * cohort_s
w = n / (n + k)
```

where `n` = entity event count and `k` = `profile.cohort_shrinkage_k` (default 20).

| entity events | w | interpretation |
|---------------|---|----------------|
| 0 | 0.00 | pure cohort prior |
| 3 | 0.13 | mostly cohort |
| 20 | 0.50 | equal weight |
| 100 | 0.83 | mostly entity |
| 3 000 | 0.99 | nearly pure entity |

This is why cold-start entities are not falsely flagged for unusual hours: their
circadian model is almost entirely the cohort average until they accumulate
sufficient history.

---

## Performance

Measured on the benchmark test (500 events, 10 entities, 3 cohorts, mixed
command sequences, Windows/Python 3.12):

- **~0.37 ms/event average** (≈ 2 700 events/sec)
- Well under the 1 ms real-time budget

All state updates are O(1) (EWMA, Welford online variance, bounded deques).
No full history rescan occurs per event.

---

## State serialisation

`PipelineState` (and all sub-states) implement `to_dict() / from_dict()` for
snapshot/restore.  This is used by the training pipeline to persist the profiler
artifact (`artifacts/profiler.joblib`) and by the serving layer to restore it at
startup.  A round-trip test (`TestDeterminismAndRoundTrip.test_state_roundtrip_produces_identical_subsequent_features`)
asserts that features computed after restoring a serialised state are
bit-identical to those computed from the original live state.
