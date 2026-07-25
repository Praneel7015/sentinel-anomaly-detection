"""Cohort-level aggregates, Bayesian shrinkage, and cold-start gating.

A "cohort" is a role group (e.g. ``finance_analyst``, ``plc_gateway``).
``CohortState`` holds pooled statistics for every entity of the same cohort,
in the same incremental form as ``EntityState``.

Shrinkage formula
-----------------
For a scalar statistic ``s``:

    shrunk = w * entity_s + (1 - w) * cohort_s
    w      = n / (n + k)

where ``n`` = entity event count, ``k`` = ``cohort_shrinkage_k`` from
``configs/model.yaml`` (default 20).  An entity with 3 events has
``w ≈ 0.13`` — it is scored almost entirely against its cohort.  An entity
with 3000 events has ``w ≈ 0.99`` — it is scored against itself.

Cold start
----------
``is_cold_start(entity_state)`` returns ``True`` when
``entity_state.event_count < min_events`` (``configs/model.yaml``
``cold_start.min_events``, default 25).  Features marked
``cold_start_trustworthy = False`` in the extractor registry should be gated
(or widened) when the entity is cold.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import UTC
from typing import Any

from sentinel.features.state import (
    _CIRCADIAN_BUCKETS,
    _LAPLACE_ALPHA,
    _WEEKDAY_BUCKETS,
    EntityState,
)

__all__ = [
    "CohortState",
    "shrink",
    "shrink_histogram",
    "is_cold_start",
]

_DEFAULT_K: int = 20
_DEFAULT_MIN_EVENTS: int = 25


class CohortState:
    """Pooled statistics for a role cohort.

    Updated once per entity event via ``update(event)`` — same incremental
    contract as ``EntityState``.
    """

    def __init__(self, cohort_id: str) -> None:
        self.cohort_id: str = cohort_id
        self.event_count: int = 0
        self.entity_ids: set[str] = set()

        # Circadian and weekday histograms (Laplace-smoothed)
        self.circadian: list[float] = [_LAPLACE_ALPHA] * _CIRCADIAN_BUCKETS
        self.weekday: list[float] = [_LAPLACE_ALPHA] * _WEEKDAY_BUCKETS

        # Resource popularity within the cohort
        self.resource_counts: Counter[str] = Counter()

        # Auth method distribution
        self.auth_method_counts: Counter[str] = Counter()

        # Per-resource distinct-entity count (how many peers touched resource X)
        self.resource_entity_sets: dict[str, set[str]] = {}

        # Bytes EWMA (pooled simple mean, not time-weighted)
        self.bytes_total: float = 0.0

        # Session duration running mean + M2 (Welford, pooled)
        self._dur_n: int = 0
        self._dur_mean: float = 0.0
        self._dur_M2: float = 0.0

        # Command Markov transition counts (pooled)
        self.markov_counts: Counter[tuple[str, str]] = Counter()
        self.markov_unigrams: Counter[str] = Counter()

        # Protocol and device-OS distributions
        self.protocol_counts: Counter[str] = Counter()
        self.os_counts: Counter[str] = Counter()

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #

    def update(self, event: dict[str, Any]) -> None:
        """Advance cohort state with one event (call after entity update)."""
        from datetime import datetime

        from sentinel.features.state import _to_ts

        ts = _to_ts(event["timestamp"])
        dt = datetime.fromtimestamp(ts, tz=UTC)
        hour = dt.hour
        dow = dt.weekday()
        entity_id = str(event["entity_id"])

        self.event_count += 1
        self.entity_ids.add(entity_id)

        self.circadian[hour] += 1.0
        self.weekday[dow] += 1.0

        resource = str(event["resource_accessed"])
        self.resource_counts[resource] += 1
        if resource not in self.resource_entity_sets:
            self.resource_entity_sets[resource] = set()
        self.resource_entity_sets[resource].add(entity_id)

        self.auth_method_counts[str(event["auth_method"])] += 1
        self.bytes_total += float(event["bytes_transferred"])

        # Welford for session duration
        dur = float(event["session_duration_s"])
        self._dur_n += 1
        delta = dur - self._dur_mean
        self._dur_mean += delta / self._dur_n
        delta2 = dur - self._dur_mean
        self._dur_M2 += delta * delta2

        cmds: list[str] = list(event.get("command_sequence") or [])
        for cmd in cmds:
            self.markov_unigrams[cmd] += 1
        for prev_cmd, curr_cmd in zip(cmds, cmds[1:], strict=False):
            self.markov_counts[(prev_cmd, curr_cmd)] += 1

        self.protocol_counts[str(event["device_protocol"])] += 1
        self.os_counts[str(event["device_os"])] += 1

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def circadian_prob(self, hour: int) -> float:
        total = sum(self.circadian)
        return self.circadian[hour] / total

    def weekday_prob(self, dow: int) -> float:
        total = sum(self.weekday)
        return self.weekday[dow] / total

    def peer_count_for_resource(self, resource: str) -> int:
        """How many distinct entities in this cohort have ever accessed ``resource``."""
        s = self.resource_entity_sets.get(resource)
        return len(s) if s else 0

    @property
    def n_entities(self) -> int:
        return len(self.entity_ids)

    @property
    def bytes_mean(self) -> float:
        if self.event_count == 0:
            return 0.0
        return self.bytes_total / self.event_count

    @property
    def duration_mean(self) -> float:
        return self._dur_mean

    @property
    def duration_variance(self) -> float:
        if self._dur_n < 2:
            return 0.0
        return self._dur_M2 / self._dur_n

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "cohort_id": self.cohort_id,
            "event_count": self.event_count,
            "entity_ids": list(self.entity_ids),
            "circadian": list(self.circadian),
            "weekday": list(self.weekday),
            "resource_counts": dict(self.resource_counts),
            "auth_method_counts": dict(self.auth_method_counts),
            "resource_entity_sets": {
                r: list(s) for r, s in self.resource_entity_sets.items()
            },
            "bytes_total": self.bytes_total,
            "_dur_n": self._dur_n,
            "_dur_mean": self._dur_mean,
            "_dur_M2": self._dur_M2,
            "markov_counts": {f"{k[0]}\x00{k[1]}": v for k, v in self.markov_counts.items()},
            "markov_unigrams": dict(self.markov_unigrams),
            "protocol_counts": dict(self.protocol_counts),
            "os_counts": dict(self.os_counts),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CohortState:
        obj = cls.__new__(cls)
        obj.cohort_id = d["cohort_id"]
        obj.event_count = d["event_count"]
        obj.entity_ids = set(d["entity_ids"])
        obj.circadian = list(d["circadian"])
        obj.weekday = list(d["weekday"])
        obj.resource_counts = Counter(d["resource_counts"])
        obj.auth_method_counts = Counter(d["auth_method_counts"])
        obj.resource_entity_sets = {
            r: set(s) for r, s in d["resource_entity_sets"].items()
        }
        obj.bytes_total = d["bytes_total"]
        obj._dur_n = d["_dur_n"]
        obj._dur_mean = d["_dur_mean"]
        obj._dur_M2 = d["_dur_M2"]
        obj.markov_counts = Counter(
            {(k.split("\x00")[0], k.split("\x00")[1]): v for k, v in d["markov_counts"].items()}
        )
        obj.markov_unigrams = Counter(d["markov_unigrams"])
        obj.protocol_counts = Counter(d["protocol_counts"])
        obj.os_counts = Counter(d["os_counts"])
        return obj


# ---------------------------------------------------------------------------
# Shrinkage utilities
# ---------------------------------------------------------------------------


def shrink(entity_val: float, cohort_val: float, n: int, k: int = _DEFAULT_K) -> float:
    """Return Bayesian shrinkage blend.

    ``w = n / (n + k)``.  With k=20:
    - n=3  → w≈0.13  (mostly cohort)
    - n=25 → w=0.56  (nearly equal)
    - n=3000 → w≈0.99 (mostly entity)
    """
    w = n / (n + k)
    return w * entity_val + (1.0 - w) * cohort_val


def shrink_histogram(
    entity_hist: list[float],
    cohort_hist: list[float],
    n: int,
    k: int = _DEFAULT_K,
) -> list[float]:
    """Element-wise shrinkage blend of two histograms of the same length."""
    w = n / (n + k)
    return [w * e + (1.0 - w) * c for e, c in zip(entity_hist, cohort_hist, strict=False)]


def is_cold_start(entity_state: EntityState, min_events: int = _DEFAULT_MIN_EVENTS) -> bool:
    """True when the entity has fewer events than ``min_events``.

    This is the single canonical cold-start gate used by both the extractor
    registry and the drift agent.  Extractors with ``cold_start_trustworthy=False``
    should surface ``is_cold_start`` as a flag rather than driving scoring.
    """
    return entity_state.event_count < min_events


def surprisal(prob: float, *, eps: float = 1e-10) -> float:
    """Negative log-2 probability (bits), clamped to avoid log(0)."""
    return -math.log2(max(prob, eps))
