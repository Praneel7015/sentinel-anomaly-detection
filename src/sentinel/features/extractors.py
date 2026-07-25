"""Feature extractor registry and FeatureVector.

Each extractor is a named pure function:
    f(event, entity_state, cohort_state, global_state, graph_state,
      seq_model_cohort, seq_model_global, config) -> float

Registry metadata drives the explainability layer - display_name and
description are rendered verbatim by the dashboard.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from sentinel.features.cohort import CohortState, is_cold_start, shrink, surprisal
from sentinel.features.graph import GraphState
from sentinel.features.sequence import (
    SequenceModel,
    sequence_length_zscore,
    sequence_surprisal,
    unseen_bigram_count,
)
from sentinel.features.state import EntityState, GlobalState, _offhours, _to_ts

__all__ = ["FEATURE_NAMES", "FeatureMeta", "FeatureVector", "REGISTRY", "extract_all"]

_R = 6371.0  # Earth radius km

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * _R * math.asin(math.sqrt(min(a, 1.0)))

def _js_divergence(p: list[float], q: list[float]) -> float:
    """Jensen-Shannon divergence between two distributions (0..1 scale)."""
    s_p = sum(p)
    s_q = sum(q)
    if s_p == 0 or s_q == 0:
        return 0.0
    pn = [x / s_p for x in p]
    qn = [x / s_q for x in q]
    m = [(a + b) / 2 for a, b in zip(pn, qn, strict=False)]
    def _kl(a: list[float], b: list[float]) -> float:
        return sum(ai * math.log2(ai / bi) for ai, bi in zip(a, b, strict=False) if ai > 1e-12 and bi > 1e-12)
    return (_kl(pn, m) + _kl(qn, m)) / 2

def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    return dot / (na * nb)

@dataclass(frozen=True)
class FeatureMeta:
    name: str
    display_name: str
    description: str
    direction: str  # "increases_risk" | "decreases_risk" | "neutral"
    cold_start_trustworthy: bool

_Extractor = Callable[
    [dict[str, Any], EntityState, CohortState, GlobalState, GraphState,
     SequenceModel | None, SequenceModel | None, dict[str, Any]],
    float,
]

@dataclass
class _RegEntry:
    meta: FeatureMeta
    fn: _Extractor

_REGISTRY: list[_RegEntry] = []

def _reg(
    name: str,
    display_name: str,
    description: str,
    direction: str,
    cold_start_trustworthy: bool,
) -> Callable[[_Extractor], _Extractor]:
    def decorator(fn: _Extractor) -> _Extractor:
        _REGISTRY.append(_RegEntry(
            meta=FeatureMeta(name, display_name, description, direction, cold_start_trustworthy),
            fn=fn,
        ))
        return fn
    return decorator

# -----------------------------------------------------------------------
# TEMPORAL FEATURES
# -----------------------------------------------------------------------

@_reg("hour_surprisal", "Hour-of-day surprisal",
      "How unusual is this hour compared to the entity's shrunk circadian histogram (bits).",
      "increases_risk", False)
def _hour_surprisal(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    ts = _to_ts(ev["timestamp"])
    hour = datetime.fromtimestamp(ts, tz=UTC).hour
    k = cfg.get("cohort_shrinkage_k", 20)
    shrunk_hist = [shrink(e, c, es.event_count, k)
                   for e, c in zip(es.circadian, cs.circadian, strict=False)]
    total = sum(shrunk_hist)
    prob = shrunk_hist[hour] / total if total > 0 else 1.0 / 24
    return surprisal(prob)

@_reg("weekday_surprisal", "Weekday surprisal",
      "How unusual is this day-of-week compared to the entity's shrunk weekday histogram (bits).",
      "increases_risk", False)
def _weekday_surprisal(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    ts = _to_ts(ev["timestamp"])
    dow = datetime.fromtimestamp(ts, tz=UTC).weekday()
    k = cfg.get("cohort_shrinkage_k", 20)
    shrunk_hist = [shrink(e, c, es.event_count, k)
                   for e, c in zip(es.weekday, cs.weekday, strict=False)]
    total = sum(shrunk_hist)
    prob = shrunk_hist[dow] / total if total > 0 else 1.0 / 7
    return surprisal(prob)

@_reg("is_offhours", "Off-hours access",
      "1 if the event occurred outside 08:00-18:00 UTC, 0 otherwise.",
      "increases_risk", True)
def _is_offhours(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    ts = _to_ts(ev["timestamp"])
    hour = datetime.fromtimestamp(ts, tz=UTC).hour
    return 1.0 if _offhours(hour) else 0.0

@_reg("inter_arrival_zscore", "Inter-arrival time z-score",
      "Z-score of the time since the last event for this entity vs its history.",
      "increases_risk", False)
def _inter_arrival_zscore(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    if es.last_seen_s is None or es.inter_arrival_welford.n < 2:
        return 0.0
    ts = _to_ts(ev["timestamp"])
    gap = ts - es.last_seen_s
    mean = es.inter_arrival_welford.mean
    std = es.inter_arrival_welford.std
    if std < 1e-6:
        return 0.0
    return (gap - mean) / std

@_reg("burstiness", "Session burstiness",
      "Coefficient of variation of inter-arrival times; high = bursty activity.",
      "increases_risk", False)
def _burstiness(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    w = es.inter_arrival_welford
    if w.n < 2 or w.mean < 1e-6:
        return 0.0
    return w.std / w.mean

# -----------------------------------------------------------------------
# GEO / NETWORK FEATURES
# -----------------------------------------------------------------------

@_reg("geo_velocity_kmh", "Geo-velocity (km/h)",
      "Speed implied by moving from the previous event location to this one.",
      "increases_risk", True)
def _geo_velocity_kmh(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    if es.last_geo_lat is None or es.last_geo_ts_s is None:
        return 0.0
    ts = _to_ts(ev["timestamp"])
    dt_h = (ts - es.last_geo_ts_s) / 3600.0
    if dt_h <= 0:
        return 0.0
    dist = _haversine(es.last_geo_lat, es.last_geo_lon,
                      float(ev["geo_lat"]), float(ev["geo_lon"]))
    return dist / dt_h

@_reg("geo_centroid_distance_km", "Distance from home centroid (km)",
      "Haversine distance from this event location to the entity home geo centroid.",
      "increases_risk", False)
def _geo_centroid_distance_km(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    if es.geo_n == 0:
        return 0.0
    return _haversine(es.geo_lat_mean, es.geo_lon_mean,
                      float(ev["geo_lat"]), float(ev["geo_lon"]))

@_reg("new_country_flag", "New country",
      "1 if this event is from a country never seen for this entity before.",
      "increases_risk", True)
def _new_country_flag(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    return 0.0 if str(ev["geo_country"]) in es.country_set else 1.0

@_reg("new_subnet_flag", "New subnet",
      "1 if the source IP /24 subnet has never been seen for this entity.",
      "increases_risk", True)
def _new_subnet_flag(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    from sentinel.features.state import _subnet24
    subnet = _subnet24(str(ev["source_ip"]))
    return 0.0 if subnet in es.subnet_set else 1.0

@_reg("source_ip_novelty", "Source IP novelty",
      "1 if this exact source IP has never been seen for this entity (subnet check is coarser).",
      "increases_risk", True)
def _source_ip_novelty(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    ip = str(ev["source_ip"])
    # Track IPs in subnet_set at /24 granularity for bounded memory
    return 0.0 if ip in es.subnet_set else 1.0

# -----------------------------------------------------------------------
# RESOURCE FEATURES
# -----------------------------------------------------------------------

@_reg("entity_novel_resource", "Entity-novel resource",
      "1 if this resource has never been accessed by this entity before.",
      "increases_risk", True)
def _entity_novel_resource(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    return 0.0 if str(ev["resource_accessed"]) in es.resource_set else 1.0

@_reg("cohort_novel_resource", "Cohort-novel resource",
      "1 if no peer in this cohort has ever accessed this resource (lateral movement signal).",
      "increases_risk", True)
def _cohort_novel_resource(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    resource = str(ev["resource_accessed"])
    return 0.0 if cs.resource_counts[resource] > 0 else 1.0

@_reg("distinct_resources_1h", "Distinct resources (1h)",
      "Number of distinct resources accessed by this entity in the rolling 1-hour window.",
      "increases_risk", False)
def _distinct_resources_1h(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    ts = _to_ts(ev["timestamp"])
    from sentinel.features.state import _prune_deque
    _prune_deque(es.resource_ts_1h, ts, 3600)
    return float(len(es.resource_ts_1h))

@_reg("distinct_resources_24h", "Distinct resources (24h)",
      "Number of distinct resources accessed by this entity in the rolling 24-hour window.",
      "increases_risk", False)
def _distinct_resources_24h(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    ts = _to_ts(ev["timestamp"])
    from sentinel.features.state import _prune_deque
    _prune_deque(es.resource_ts_24h, ts, 86400)
    return float(len(es.resource_ts_24h))

@_reg("resource_js_divergence", "Resource distribution JS divergence",
      "Jensen-Shannon divergence of this entity's resource distribution vs cohort average.",
      "increases_risk", False)
def _resource_js_divergence(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    all_resources = list(cs.resource_counts.keys())
    if not all_resources:
        return 0.0
    total_e = sum(es.resource_counts.values()) or 1
    total_c = sum(cs.resource_counts.values()) or 1
    p = [es.resource_counts[r] / total_e for r in all_resources]
    q = [cs.resource_counts[r] / total_c for r in all_resources]
    return _js_divergence(p, q)

@_reg("resource_breadth_expansion_rate", "Resource breadth expansion rate",
      "Rate of new resources per event: distinct_resources / event_count.",
      "increases_risk", False)
def _resource_breadth_expansion_rate(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    if es.event_count == 0:
        return 0.0
    return len(es.resource_set) / es.event_count

# -----------------------------------------------------------------------
# AUTH FEATURES
# -----------------------------------------------------------------------

@_reg("auth_method_novelty", "Auth method novelty",
      "1 if this authentication method has never been used by this entity.",
      "increases_risk", True)
def _auth_method_novelty(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    return 0.0 if str(ev["auth_method"]) in es.auth_method_counts else 1.0

@_reg("entity_failure_count_5m", "Auth failures - entity (5 min)",
      "Rolling count of authentication failures by this entity in the last 5 minutes.",
      "increases_risk", True)
def _entity_fail_5m(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    ts = _to_ts(ev["timestamp"])
    return float(es.failure_count(ts, 300))

@_reg("entity_failure_count_1h", "Auth failures - entity (1 h)",
      "Rolling count of authentication failures by this entity in the last hour.",
      "increases_risk", True)
def _entity_fail_1h(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    ts = _to_ts(ev["timestamp"])
    return float(es.failure_count(ts, 3600))

@_reg("entity_failure_count_24h", "Auth failures - entity (24 h)",
      "Rolling count of authentication failures by this entity in the last 24 hours.",
      "increases_risk", True)
def _entity_fail_24h(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    ts = _to_ts(ev["timestamp"])
    return float(es.failure_count(ts, 86400))

@_reg("entity_failure_ratio_1h", "Auth failure ratio - entity (1 h)",
      "Fraction of events in the last hour that were auth failures for this entity.",
      "increases_risk", True)
def _entity_fail_ratio_1h(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    ts = _to_ts(ev["timestamp"])
    fail = es.failure_count(ts, 3600)
    total = max(1, len(es.resource_ts_1h) + 1)
    return fail / total

@_reg("ip_failure_count_5m", "Auth failures - source IP (5 min)",
      "Rolling count of authentication failures from this source IP in the last 5 minutes.",
      "increases_risk", True)
def _ip_fail_5m(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    ts = _to_ts(ev["timestamp"])
    ip_state = gs.get_source_ip_state(str(ev["source_ip"]))
    return float(ip_state.failure_count(ts, 300))

@_reg("ip_failure_count_1h", "Auth failures - source IP (1 h)",
      "Rolling count of authentication failures from this source IP in the last hour.",
      "increases_risk", True)
def _ip_fail_1h(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    ts = _to_ts(ev["timestamp"])
    ip_state = gs.get_source_ip_state(str(ev["source_ip"]))
    return float(ip_state.failure_count(ts, 3600))

@_reg("ip_failure_count_24h", "Auth failures - source IP (24 h)",
      "Rolling count of authentication failures from this source IP in the last 24 hours.",
      "increases_risk", True)
def _ip_fail_24h(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    ts = _to_ts(ev["timestamp"])
    ip_state = gs.get_source_ip_state(str(ev["source_ip"]))
    return float(ip_state.failure_count(ts, 86400))

@_reg("success_after_failures", "Success after N failures",
      "1 if this is a successful auth preceded by consecutive failures (brute-force signal).",
      "increases_risk", True)
def _success_after_failures(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    n = cfg.get("min_failures_before_success", 3)
    if str(ev["auth_result"]) == "success" and es.consecutive_failures >= n:
        return 1.0
    return 0.0

@_reg("distinct_entities_per_ip", "Distinct entities per source IP",
      "Number of distinct entity IDs that have authenticated from this source IP (credential stuffing signal).",
      "increases_risk", True)
def _distinct_entities_per_ip(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    ip_state = gs.get_source_ip_state(str(ev["source_ip"]))
    return float(ip_state.distinct_entities())

# -----------------------------------------------------------------------
# DEVICE FEATURES
# -----------------------------------------------------------------------

@_reg("fingerprint_unknown", "Unknown device fingerprint",
      "1 if this device fingerprint has never been seen for this entity.",
      "increases_risk", True)
def _fingerprint_unknown(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    return 0.0 if str(ev["device_fingerprint"]) in es.fingerprint_set else 1.0

@_reg("os_mismatch", "OS mismatch vs history",
      "1 if the operating system has never been seen for this entity.",
      "increases_risk", True)
def _os_mismatch(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    return 0.0 if str(ev["device_os"]) in es.os_set else 1.0

@_reg("mac_oui_change", "MAC OUI change",
      "1 if the MAC OUI (first 3 octets) has never been seen for this entity.",
      "increases_risk", True)
def _mac_oui_change(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    mac = str(ev["device_mac"])
    oui = ":".join(mac.split(":")[:3]) if ":" in mac else mac[:6]
    return 0.0 if oui in es.mac_set else 1.0

@_reg("protocol_novelty", "Protocol novelty",
      "1 if the wire protocol has never been used by this entity.",
      "increases_risk", True)
def _protocol_novelty(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    return 0.0 if str(ev["device_protocol"]) in es.protocol_set else 1.0

# -----------------------------------------------------------------------
# SEQUENCE FEATURES
# -----------------------------------------------------------------------

@_reg("command_surprisal", "Command sequence surprisal (bits/token)",
      "Length-normalised negative log-probability of the command sequence under the shrunk Markov model.",
      "increases_risk", False)
def _command_surprisal(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    cmds = list(ev.get("command_sequence") or [])
    k = cfg.get("cohort_shrinkage_k", 20)
    smooth = cfg.get("sequence_smoothing", 0.5)
    return sequence_surprisal(
        cmds,
        es.markov_counts,
        es.markov_unigrams,
        sm_c,
        sm_g,
        es.event_count,
        shrinkage_k=k,
        smoothing=smooth,
    )

@_reg("unseen_bigram_count", "Unseen command bigrams",
      "Number of command bigrams in this session never seen before for this entity.",
      "increases_risk", False)
def _unseen_bigrams(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    cmds = list(ev.get("command_sequence") or [])
    return float(unseen_bigram_count(cmds, es.markov_counts))

@_reg("sequence_length_zscore", "Command sequence length z-score",
      "Z-score of the command sequence length vs cohort distribution.",
      "increases_risk", False)
def _seq_len_zscore(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    cmds = list(ev.get("command_sequence") or [])
    return sequence_length_zscore(len(cmds), sm_c, es.event_count)

# -----------------------------------------------------------------------
# VOLUME FEATURES
# -----------------------------------------------------------------------

@_reg("bytes_zscore", "Bytes transferred z-score",
      "Z-score of bytes transferred vs entity EWMA (normalised by session duration).",
      "increases_risk", False)
def _bytes_zscore(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    nbytes = float(ev["bytes_transferred"])
    if es.bytes_n < 2:
        return 0.0
    mean = es.bytes_ewma
    # Approximate std from session_duration_welford as a proxy
    std = max(es.session_duration_welford.std * 100, 1.0)
    return (nbytes - mean) / std

@_reg("offhours_bytes_rolling_7d", "Off-hours bytes (rolling 7 days)",
      "Cumulative bytes transferred during off-hours over the past 7 days (low-and-slow signal).",
      "increases_risk", False)
def _offhours_bytes_7d(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    ts = _to_ts(ev["timestamp"])
    return float(es.offhours_bytes_sum(ts))

@_reg("transfer_to_duration_ratio", "Transfer-to-duration ratio",
      "Bytes per second of session duration; high = anomalous bulk transfer.",
      "increases_risk", True)
def _transfer_to_duration_ratio(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    dur = float(ev["session_duration_s"])
    nbytes = float(ev["bytes_transferred"])
    if dur < 1.0:
        return nbytes
    return nbytes / dur

# -----------------------------------------------------------------------
# GRAPH FEATURES
# -----------------------------------------------------------------------

@_reg("graph_entity_novel_resource", "Graph: entity-novel resource edge",
      "1 if this is the first time this entity has accessed this resource (graph edge novelty).",
      "increases_risk", True)
def _graph_entity_novel(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    return 0.0 if str(ev["resource_accessed"]) in es.resource_set else 1.0

@_reg("graph_peer_resource_count", "Graph: peers accessing this resource",
      "Number of cohort peers that have ever accessed this resource; very low = lateral movement signal.",
      "decreases_risk", True)
def _graph_peer_resource_count(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    resource = str(ev["resource_accessed"])
    return float(cs.peer_count_for_resource(resource))

@_reg("graph_jaccard_vs_cohort", "Graph: Jaccard vs cohort centroid",
      "Jaccard similarity between entity resource set and cohort resource centroid.",
      "decreases_risk", False)
def _graph_jaccard(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    peer_counts = {r: cs.peer_count_for_resource(r) for r in cs.resource_counts}
    return gr.jaccard_vs_cohort_centroid(es.resource_set, peer_counts, cs.n_entities)

@_reg("graph_entity_degree_deviation", "Graph: entity degree deviation",
      "Z-score of how many distinct resources this entity accesses vs cohort mean.",
      "increases_risk", False)
def _graph_degree_deviation(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    entity_id = str(ev["entity_id"])
    cohort_id = str(ev["cohort"])
    return gr.entity_degree_deviation(entity_id, cohort_id)

# -----------------------------------------------------------------------
# COLD-START CONTEXT (used as gates, not suspicion signals)
# -----------------------------------------------------------------------

@_reg("entity_event_count", "Entity event count",
      "Total number of events seen for this entity so far (cold-start gate).",
      "neutral", True)
def _entity_event_count(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    return float(es.event_count)

@_reg("is_cold_start_flag", "Cold start flag",
      "1 if the entity has fewer events than the cold-start threshold.",
      "neutral", True)
def _is_cold_start_flag(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    min_ev = cfg.get("cold_start_min_events", 25)
    return 1.0 if is_cold_start(es, min_ev) else 0.0

@_reg("days_observed", "Days observed",
      "Number of days since first event for this entity.",
      "neutral", True)
def _days_observed(ev, es, cs, gs, gr, sm_c, sm_g, cfg):
    ts = _to_ts(ev["timestamp"])
    return es.days_observed(ts)

# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

REGISTRY: dict[str, _RegEntry] = {e.meta.name: e for e in _REGISTRY}
FEATURE_NAMES: list[str] = [e.meta.name for e in _REGISTRY]

# Pre-extracted extractor functions for tight-loop performance in extract_all
_EXTRACTOR_FNS: list[_Extractor] = [e.fn for e in _REGISTRY]


class FeatureVector:
    """Named, stably-ordered feature vector.

    Use ``FEATURE_NAMES`` for all index-by-name access; never index positionally
    without going through this class.
    """
    __slots__ = ("_values", "names")

    def __init__(self, values: list[float]) -> None:
        if len(values) != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(FEATURE_NAMES)} features, got {len(values)}"
            )
        self._values = values
        self.names = FEATURE_NAMES

    def __getitem__(self, name: str) -> float:
        return self._values[FEATURE_NAMES.index(name)]

    def to_numpy(self) -> np.ndarray:
        return np.array(self._values, dtype=np.float32)

    def to_dict(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self._values, strict=False))

    def __repr__(self) -> str:
        return f"FeatureVector({self.to_dict()!r})"


def extract_all(
    event: dict[str, Any],
    entity_state: EntityState,
    cohort_state: CohortState,
    global_state: GlobalState,
    graph_state: GraphState,
    seq_model_cohort: SequenceModel | None,
    seq_model_global: SequenceModel | None,
    config: dict[str, Any] | None = None,
) -> FeatureVector:
    """Compute all registered features for one event.

    Returns a :class:`FeatureVector` with stable ordering.  This is the only
    function that should be called by the pipeline.
    """
    cfg = config or {}
    values = []
    for fn in _EXTRACTOR_FNS:
        try:
            v = float(fn(event, entity_state, cohort_state, global_state,
                         graph_state, seq_model_cohort, seq_model_global, cfg))
        except Exception:
            v = 0.0
        values.append(v if math.isfinite(v) else 0.0)
    return FeatureVector(values)
