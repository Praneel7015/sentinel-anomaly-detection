"""Incremental per-entity and global state for past-only feature extraction.

Every statistic here is updated **after** features are emitted, so all
features see strictly past information.  The same state objects serve both
batch training (``batch_features`` drives a streaming loop) and live streaming
(``stream_features``), making leakage structurally impossible.

Key design constraints:
- O(1)-ish updates, bounded memory: EWMA, Welford online variance, bounded
  deques, rolling time windows - never a full history rescan.
- Serialisable: ``to_dict`` / ``from_dict`` for snapshot / restore.
- ``poisoning_guard``: ``update(event, learn=True)`` so the drift agent can
  withhold learning from high-risk events without changing the call-site API.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from datetime import UTC, datetime
from typing import Any

__all__ = ["EntityState", "GlobalState", "SourceIPState"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CIRCADIAN_BUCKETS: int = 24
_WEEKDAY_BUCKETS: int = 7
_LAPLACE_ALPHA: float = 0.1  # Laplace smoothing for histograms

# Default EWMA half-life in days (overridden by the pipeline from model.yaml)
_DEFAULT_HALF_LIFE_DAYS: float = 14.0

# Rolling failure-window horizons in seconds
_FAILURE_WINDOWS_S: tuple[int, ...] = (300, 3600, 86400)  # 5 min, 1 h, 24 h

# Off-hours rolling window for cumulative bytes (seconds)
_OFFHOURS_BYTES_WINDOW_S: int = 7 * 86400  # 7 days

# Max distinct resources tracked in the set (bounded memory)
_MAX_DISTINCT_RESOURCES: int = 10_000

# Max distinct fingerprints tracked per entity
_MAX_FINGERPRINTS: int = 100

# Max entries in rolling deques
_MAX_DEQUE: int = 1_000

# GRU context window length (documented in sequence.py but bounded here)
GRU_WINDOW: int = 16


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _ewma_alpha(half_life_days: float, elapsed_days: float) -> float:
    """Decay factor for time-weighted EWMA over ``elapsed_days``."""
    if elapsed_days <= 0.0 or half_life_days <= 0.0:
        return 1.0
    return 1.0 - math.exp(-elapsed_days * math.log(2) / half_life_days)


def _offhours(hour: int) -> bool:
    """True for hours outside 08:00-18:00 UTC (inclusive start, exclusive end)."""
    return hour < 8 or hour >= 18


# ---------------------------------------------------------------------------
# Welford running mean+variance (online, O(1))
# ---------------------------------------------------------------------------


class _Welford:
    """Welford online algorithm for mean and population variance."""

    __slots__ = ("n", "mean", "_M2")

    def __init__(self) -> None:
        self.n: int = 0
        self.mean: float = 0.0
        self._M2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self._M2 / self.n if self.n >= 2 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.n, "mean": self.mean, "_M2": self._M2}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> _Welford:
        obj = cls()
        obj.n = d["n"]
        obj.mean = d["mean"]
        obj._M2 = d["_M2"]
        return obj


# ---------------------------------------------------------------------------
# SourceIPState  (held inside GlobalState)
# ---------------------------------------------------------------------------


class SourceIPState:
    """Rolling per-source-IP counters: failures, distinct entities attempted."""

    __slots__ = (
        "total_events",
        "failure_deques",
        "entity_set",
        "auth_method_counts",
    )

    def __init__(self) -> None:
        self.total_events: int = 0
        # keyed by window seconds -> deque of (timestamp_s,)
        self.failure_deques: dict[int, deque[float]] = {
            w: deque() for w in _FAILURE_WINDOWS_S
        }
        self.entity_set: set[str] = set()
        self.auth_method_counts: Counter[str] = Counter()

    def update(self, ts: float, entity_id: str, auth_result: str, auth_method: str) -> None:
        self.total_events += 1
        self.entity_set.add(entity_id)
        self.auth_method_counts[auth_method] += 1
        if auth_result == "failure":
            for window_s, dq in self.failure_deques.items():
                dq.append(ts)
                # prune old entries
                cutoff = ts - window_s
                while dq and dq[0] < cutoff:
                    dq.popleft()

    def failure_count(self, ts: float, window_s: int) -> int:
        if window_s not in self.failure_deques:
            return 0
        dq = self.failure_deques[window_s]
        cutoff = ts - window_s
        # deque is time-ordered; prune and count in one pass
        while dq and dq[0] < cutoff:
            dq.popleft()
        return len(dq)

    def distinct_entities(self) -> int:
        return len(self.entity_set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_events": self.total_events,
            "failure_deques": {str(k): list(v) for k, v in self.failure_deques.items()},
            "entity_set": list(self.entity_set),
            "auth_method_counts": dict(self.auth_method_counts),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SourceIPState:
        obj = cls()
        obj.total_events = d["total_events"]
        obj.failure_deques = {
            int(k): deque(v) for k, v in d["failure_deques"].items()
        }
        obj.entity_set = set(d["entity_set"])
        obj.auth_method_counts = Counter(d["auth_method_counts"])
        return obj


# ---------------------------------------------------------------------------
# EntityState
# ---------------------------------------------------------------------------


class EntityState:
    """Incremental per-entity behavioural profile.

    State is updated **after** feature extraction so that every feature for
    event N is computed from events 0..N-1 only.

    ``update(event, learn=True)`` — set ``learn=False`` to skip baseline
    update (poisoning guard plumbing; policy lives in the drift agent).
    """

    def __init__(self, entity_id: str, half_life_days: float = _DEFAULT_HALF_LIFE_DAYS) -> None:
        self.entity_id: str = entity_id
        self.half_life_days: float = half_life_days

        # --- bookkeeping ---
        self.event_count: int = 0
        self.first_seen_s: float | None = None
        self.last_seen_s: float | None = None

        # --- temporal histograms (Laplace-smoothed) ---
        self.circadian: list[float] = [_LAPLACE_ALPHA] * _CIRCADIAN_BUCKETS
        self.weekday: list[float] = [_LAPLACE_ALPHA] * _WEEKDAY_BUCKETS

        # --- geo: centroid + last geo/time for velocity ---
        self.geo_lat_mean: float = 0.0
        self.geo_lon_mean: float = 0.0
        self.geo_n: int = 0
        self.last_geo_lat: float | None = None
        self.last_geo_lon: float | None = None
        self.last_geo_ts_s: float | None = None
        self.country_set: set[str] = set()

        # --- network ---
        self.subnet_set: set[str] = set()  # /24 subnets seen

        # --- resource ---
        self.resource_set: set[str] = set()  # bounded by _MAX_DISTINCT_RESOURCES
        self.resource_counts: Counter[str] = Counter()

        # rolling 1h / 24h resource-access timestamps
        self.resource_ts_1h: deque[float] = deque()  # timestamps of resource accesses
        self.resource_ts_24h: deque[float] = deque()

        # --- auth ---
        self.auth_method_counts: Counter[str] = Counter()
        # rolling failure deques keyed by window seconds
        self.failure_deques: dict[int, deque[float]] = {
            w: deque() for w in _FAILURE_WINDOWS_S
        }
        self.consecutive_failures: int = 0
        self.last_auth_result: str = "success"

        # --- device ---
        self.fingerprint_set: set[str] = set()
        self.os_set: set[str] = set()
        self.mac_set: set[str] = set()  # OUI = first 3 octets
        self.protocol_set: set[str] = set()

        # --- session ---
        self.session_duration_welford: _Welford = _Welford()

        # --- bytes EWMA + rolling off-hours bytes ---
        self.bytes_ewma: float = 0.0
        self.bytes_n: int = 0
        self.offhours_bytes_deque: deque[tuple[float, int]] = deque()  # (ts_s, bytes)

        # --- inter-arrival ---
        self.inter_arrival_welford: _Welford = _Welford()

        # --- command Markov transition counts ---
        # bigrams: (prev_cmd, curr_cmd) -> count
        self.markov_counts: Counter[tuple[str, str]] = Counter()
        self.markov_unigrams: Counter[str] = Counter()

        # rolling recent event deque for GRU window encoding
        # each entry is a compact numeric tuple; see sequence.py for encoding
        self.gru_window: deque[tuple[int, ...]] = deque(maxlen=GRU_WINDOW)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def update(self, event: dict[str, Any], *, learn: bool = True) -> None:
        """Advance entity state with ``event``.

        Call **after** emitting features for this event.

        Args:
            event: a dict with at least the keys from ``AccessEvent`` (or the
                   pandas row dict from ``batch_features``).
            learn: if False, statistics are NOT updated.  The poisoning-guard
                   policy sets this; the API is always the same to callers.
        """
        if not learn:
            self.event_count += 1
            # Still advance bookkeeping so event_count is accurate
            ts = _to_ts(event["timestamp"])
            if self.first_seen_s is None:
                self.first_seen_s = ts
            self.last_seen_s = ts
            return

        ts = _to_ts(event["timestamp"])
        dt = datetime.fromtimestamp(ts, tz=UTC)
        hour = dt.hour
        dow = dt.weekday()  # 0=Mon

        # bookkeeping
        if self.first_seen_s is None:
            self.first_seen_s = ts
        if self.last_seen_s is not None:
            self.inter_arrival_welford.update(ts - self.last_seen_s)
        self.last_seen_s = ts
        self.event_count += 1

        # temporal
        self.circadian[hour] += 1.0
        self.weekday[dow] += 1.0

        # geo
        lat = float(event["geo_lat"])
        lon = float(event["geo_lon"])
        self.geo_n += 1
        self.geo_lat_mean += (lat - self.geo_lat_mean) / self.geo_n
        self.geo_lon_mean += (lon - self.geo_lon_mean) / self.geo_n
        self.last_geo_lat = lat
        self.last_geo_lon = lon
        self.last_geo_ts_s = ts
        self.country_set.add(str(event["geo_country"]))

        # network
        ip = str(event["source_ip"])
        subnet = _subnet24(ip)
        self.subnet_set.add(subnet)

        # resource
        resource = str(event["resource_accessed"])
        if len(self.resource_set) < _MAX_DISTINCT_RESOURCES:
            self.resource_set.add(resource)
        self.resource_counts[resource] += 1
        self.resource_ts_1h.append(ts)
        self.resource_ts_24h.append(ts)
        _prune_deque(self.resource_ts_1h, ts, 3600)
        _prune_deque(self.resource_ts_24h, ts, 86400)

        # auth
        method = str(event["auth_method"])
        result = str(event["auth_result"])
        self.auth_method_counts[method] += 1
        if result == "failure":
            for window_s, dq in self.failure_deques.items():
                dq.append(ts)
                _prune_deque(dq, ts, window_s)
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
        self.last_auth_result = result

        # device
        fp = str(event["device_fingerprint"])
        if len(self.fingerprint_set) < _MAX_FINGERPRINTS:
            self.fingerprint_set.add(fp)
        self.os_set.add(str(event["device_os"]))
        mac = str(event["device_mac"])
        oui = ":".join(mac.split(":")[:3]) if ":" in mac else mac[:6]
        self.mac_set.add(oui)
        self.protocol_set.add(str(event["device_protocol"]))

        # session
        self.session_duration_welford.update(float(event["session_duration_s"]))

        # bytes
        nbytes = int(event["bytes_transferred"])
        if self.bytes_n == 0:
            self.bytes_ewma = float(nbytes)
        else:
            alpha = _ewma_alpha(self.half_life_days, (ts - self.last_seen_s + 1) / 86400)
            self.bytes_ewma = alpha * nbytes + (1 - alpha) * self.bytes_ewma
        self.bytes_n += 1

        if _offhours(hour):
            self.offhours_bytes_deque.append((ts, nbytes))
            _prune_deque_pairs(self.offhours_bytes_deque, ts, _OFFHOURS_BYTES_WINDOW_S)

        # commands (Markov)
        cmds: list[str] = list(event.get("command_sequence") or [])
        for cmd in cmds:
            self.markov_unigrams[cmd] += 1
        for prev_cmd, curr_cmd in zip(cmds, cmds[1:], strict=False):
            self.markov_counts[(prev_cmd, curr_cmd)] += 1

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def failure_count(self, ts: float, window_s: int) -> int:
        """Rolling failure count within ``window_s`` seconds of ``ts``."""
        dq = self.failure_deques.get(window_s)
        if dq is None:
            return 0
        _prune_deque(dq, ts, window_s)
        return len(dq)

    def circadian_prob(self, hour: int) -> float:
        """Laplace-smoothed probability of observing ``hour``."""
        total = sum(self.circadian)
        return self.circadian[hour] / total

    def weekday_prob(self, dow: int) -> float:
        """Laplace-smoothed probability of observing day-of-week ``dow``."""
        total = sum(self.weekday)
        return self.weekday[dow] / total

    def offhours_bytes_sum(self, ts: float) -> int:
        """Sum of bytes transferred during off-hours in the rolling window."""
        _prune_deque_pairs(self.offhours_bytes_deque, ts, _OFFHOURS_BYTES_WINDOW_S)
        return sum(b for _, b in self.offhours_bytes_deque)

    def days_observed(self, ts: float) -> float:
        if self.first_seen_s is None:
            return 0.0
        return (ts - self.first_seen_s) / 86400.0

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "half_life_days": self.half_life_days,
            "event_count": self.event_count,
            "first_seen_s": self.first_seen_s,
            "last_seen_s": self.last_seen_s,
            "circadian": list(self.circadian),
            "weekday": list(self.weekday),
            "geo_lat_mean": self.geo_lat_mean,
            "geo_lon_mean": self.geo_lon_mean,
            "geo_n": self.geo_n,
            "last_geo_lat": self.last_geo_lat,
            "last_geo_lon": self.last_geo_lon,
            "last_geo_ts_s": self.last_geo_ts_s,
            "country_set": list(self.country_set),
            "subnet_set": list(self.subnet_set),
            "resource_set": list(self.resource_set),
            "resource_counts": dict(self.resource_counts),
            "resource_ts_1h": list(self.resource_ts_1h),
            "resource_ts_24h": list(self.resource_ts_24h),
            "auth_method_counts": dict(self.auth_method_counts),
            "failure_deques": {str(k): list(v) for k, v in self.failure_deques.items()},
            "consecutive_failures": self.consecutive_failures,
            "last_auth_result": self.last_auth_result,
            "fingerprint_set": list(self.fingerprint_set),
            "os_set": list(self.os_set),
            "mac_set": list(self.mac_set),
            "protocol_set": list(self.protocol_set),
            "session_duration_welford": self.session_duration_welford.to_dict(),
            "bytes_ewma": self.bytes_ewma,
            "bytes_n": self.bytes_n,
            "offhours_bytes_deque": list(self.offhours_bytes_deque),
            "inter_arrival_welford": self.inter_arrival_welford.to_dict(),
            "markov_counts": {f"{k[0]}\x00{k[1]}": v for k, v in self.markov_counts.items()},
            "markov_unigrams": dict(self.markov_unigrams),
            "gru_window": [list(t) for t in self.gru_window],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EntityState:
        obj = cls.__new__(cls)
        obj.entity_id = d["entity_id"]
        obj.half_life_days = d["half_life_days"]
        obj.event_count = d["event_count"]
        obj.first_seen_s = d["first_seen_s"]
        obj.last_seen_s = d["last_seen_s"]
        obj.circadian = list(d["circadian"])
        obj.weekday = list(d["weekday"])
        obj.geo_lat_mean = d["geo_lat_mean"]
        obj.geo_lon_mean = d["geo_lon_mean"]
        obj.geo_n = d["geo_n"]
        obj.last_geo_lat = d["last_geo_lat"]
        obj.last_geo_lon = d["last_geo_lon"]
        obj.last_geo_ts_s = d["last_geo_ts_s"]
        obj.country_set = set(d["country_set"])
        obj.subnet_set = set(d["subnet_set"])
        obj.resource_set = set(d["resource_set"])
        obj.resource_counts = Counter(d["resource_counts"])
        obj.resource_ts_1h = deque(d["resource_ts_1h"])
        obj.resource_ts_24h = deque(d["resource_ts_24h"])
        obj.auth_method_counts = Counter(d["auth_method_counts"])
        obj.failure_deques = {int(k): deque(v) for k, v in d["failure_deques"].items()}
        obj.consecutive_failures = d["consecutive_failures"]
        obj.last_auth_result = d["last_auth_result"]
        obj.fingerprint_set = set(d["fingerprint_set"])
        obj.os_set = set(d["os_set"])
        obj.mac_set = set(d["mac_set"])
        obj.protocol_set = set(d["protocol_set"])
        obj.session_duration_welford = _Welford.from_dict(d["session_duration_welford"])
        obj.bytes_ewma = d["bytes_ewma"]
        obj.bytes_n = d["bytes_n"]
        obj.offhours_bytes_deque = deque(
            (float(ts), int(b)) for ts, b in d["offhours_bytes_deque"]
        )
        obj.inter_arrival_welford = _Welford.from_dict(d["inter_arrival_welford"])
        obj.markov_counts = Counter(
            {(k.split("\x00")[0], k.split("\x00")[1]): v for k, v in d["markov_counts"].items()}
        )
        obj.markov_unigrams = Counter(d["markov_unigrams"])
        obj.gru_window = deque(
            (tuple(t) for t in d["gru_window"]), maxlen=GRU_WINDOW
        )
        return obj


# ---------------------------------------------------------------------------
# GlobalState
# ---------------------------------------------------------------------------


class GlobalState:
    """Cross-entity structures: per-source-IP counters and resource popularity.

    Shared across all entities processed in one pipeline run.
    """

    def __init__(self) -> None:
        # source-IP state
        self.source_ip_states: dict[str, SourceIPState] = {}
        # global resource popularity: resource -> total access count
        self.resource_global_counts: Counter[str] = Counter()
        # total events processed
        self.total_events: int = 0

    def get_source_ip_state(self, ip: str) -> SourceIPState:
        if ip not in self.source_ip_states:
            self.source_ip_states[ip] = SourceIPState()
        return self.source_ip_states[ip]

    def update(self, event: dict[str, Any]) -> None:
        """Update global counters with ``event``."""
        ip = str(event["source_ip"])
        ts = _to_ts(event["timestamp"])
        entity_id = str(event["entity_id"])
        auth_result = str(event["auth_result"])
        auth_method = str(event["auth_method"])
        resource = str(event["resource_accessed"])

        ip_state = self.get_source_ip_state(ip)
        ip_state.update(ts, entity_id, auth_result, auth_method)
        self.resource_global_counts[resource] += 1
        self.total_events += 1

    def resource_popularity(self, resource: str) -> float:
        """Fraction of all events that accessed ``resource``."""
        if self.total_events == 0:
            return 0.0
        return self.resource_global_counts[resource] / self.total_events

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_ip_states": {
                ip: state.to_dict() for ip, state in self.source_ip_states.items()
            },
            "resource_global_counts": dict(self.resource_global_counts),
            "total_events": self.total_events,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GlobalState:
        obj = cls()
        obj.source_ip_states = {
            ip: SourceIPState.from_dict(sd)
            for ip, sd in d["source_ip_states"].items()
        }
        obj.resource_global_counts = Counter(d["resource_global_counts"])
        obj.total_events = d["total_events"]
        return obj


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_ts(timestamp: Any) -> float:
    """Convert ``timestamp`` (datetime or pandas Timestamp) to POSIX seconds."""
    if isinstance(timestamp, (int, float)):
        return float(timestamp)
    if hasattr(timestamp, "timestamp"):
        return timestamp.timestamp()
    raise TypeError(f"Cannot convert {type(timestamp)} to POSIX seconds")


def _subnet24(ip: str) -> str:
    """Return the /24 prefix string of an IPv4 address."""
    parts = ip.split(".")
    if len(parts) == 4:
        return ".".join(parts[:3])
    return ip  # IPv6 or unusual: keep whole string


def _prune_deque(dq: deque[float], now_ts: float, window_s: int) -> None:
    """Remove entries older than ``window_s`` seconds from the left of ``dq``."""
    cutoff = now_ts - window_s
    while dq and dq[0] < cutoff:
        dq.popleft()


def _prune_deque_pairs(dq: deque[tuple[float, int]], now_ts: float, window_s: int) -> None:
    """Prune ``(ts, value)`` deques by timestamp."""
    cutoff = now_ts - window_s
    while dq and dq[0][0] < cutoff:
        dq.popleft()
