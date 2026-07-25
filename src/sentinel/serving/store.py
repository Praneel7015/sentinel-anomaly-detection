"""In-memory stores for the SENTINEL serving layer.

Four stores, all thread-safe via a single ``threading.Lock`` on mutation paths:

* :class:`AlertStore` – append-only deque of :class:`~sentinel.serving.models.ScoredEvent`,
  capped at ``max_size``, indexed by ``event_id`` and queryable by
  ``entity_id``, ``attack_type``, ``min_risk`` with paginated sorted output.
* :class:`EntityStore` – rolling entity summaries updated on each scored event,
  feeds ``GET /api/entities/{entity_id}``.
* :class:`StatsTracker` – running counters for the ``/api/stats`` endpoint.
* :class:`FeedbackStore` – analyst verdicts; false-positive feedback lowers the
  effective per-entity alert threshold.
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentinel.serving.models import ScoredEvent

__all__ = [
    "AlertStore",
    "EntityStore",
    "FeedbackStore",
    "StatsTracker",
]

_DEFAULT_MAX_ALERTS = 50_000
_EMA_ALPHA = 0.1  # smoothing factor for events-per-second EMA


# --------------------------------------------------------------------------- #
# Entity summary record (internal, not the wire model)
# --------------------------------------------------------------------------- #


@dataclass
class _EntityRecord:
    entity_id: str
    entity_type: str
    cohort: str
    first_seen: datetime
    last_seen: datetime
    event_count: int = 0
    alert_count: int = 0
    total_risk: float = 0.0
    cold_start: bool = True
    activity_by_hour: list[float] = field(default_factory=lambda: [0.0] * 24)
    # resource -> count
    resource_counts: dict[str, int] = field(default_factory=dict)
    # resource -> first-seen timestamp (to mark is_new)
    resource_first_seen: dict[str, datetime] = field(default_factory=dict)
    # risk timeline: list of (timestamp, risk_score, is_alert)
    risk_timeline: list[tuple[datetime, float, bool]] = field(default_factory=list)
    # per-entity effective threshold widening (feedback-driven)
    threshold_widening: float = 1.0
    # drift state
    drifting: bool = False
    drift_detected_at: datetime | None = None
    drift_adapted: bool = False

    @property
    def mean_risk(self) -> float:
        return self.total_risk / self.event_count if self.event_count else 0.0


# --------------------------------------------------------------------------- #
# AlertStore
# --------------------------------------------------------------------------- #


class AlertStore:
    """Append-only ring of :class:`~sentinel.serving.models.ScoredEvent`.

    The internal deque is capped at ``max_size``. Both ``_by_id`` (dict for
    O(1) lookup) and ``_deque`` are kept in sync on every append.
    """

    def __init__(self, max_size: int = _DEFAULT_MAX_ALERTS) -> None:
        self._max_size = max_size
        self._deque: deque[ScoredEvent] = deque()
        self._by_id: dict[str, ScoredEvent] = {}
        self._lock = threading.Lock()

    # -- mutation ------------------------------------------------------------ #

    def append(self, event: ScoredEvent) -> None:
        """Add ``event`` to the store; drops the oldest if over capacity."""
        with self._lock:
            if event.event_id in self._by_id:
                return  # idempotent
            if len(self._deque) >= self._max_size:
                evicted = self._deque.popleft()
                self._by_id.pop(evicted.event_id, None)
            self._deque.append(event)
            self._by_id[event.event_id] = event

    def update(self, event: ScoredEvent) -> None:
        """Replace an existing event (e.g. after feedback re-scores it)."""
        with self._lock:
            if event.event_id not in self._by_id:
                return
            self._by_id[event.event_id] = event
            # rebuild deque entry in-place (preserve order)
            new_deque: deque[ScoredEvent] = deque(
                (event if e.event_id == event.event_id else e) for e in self._deque
            )
            self._deque = new_deque

    # -- queries ------------------------------------------------------------- #

    def get(self, event_id: str) -> ScoredEvent | None:
        return self._by_id.get(event_id)

    def query(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        min_risk: float | None = None,
        attack_type: str | None = None,
        entity_id: str | None = None,
        threshold: float | None = None,
        sort: str = "risk_desc",
    ) -> tuple[list[ScoredEvent], int]:
        """Return ``(page, total)`` matching the given filters.

        ``threshold`` overrides ``is_alert`` filtering: when provided, only
        events with ``risk_score >= threshold`` are included (used by
        ``budget_pct`` query param logic in the app layer).
        """
        with self._lock:
            items: list[ScoredEvent] = list(self._deque)

        # apply filters
        if min_risk is not None:
            items = [e for e in items if e.risk_score >= min_risk]
        if attack_type is not None:
            items = [e for e in items if e.predicted_attack_type == attack_type]
        if entity_id is not None:
            items = [e for e in items if e.entity_id == entity_id]
        if threshold is not None:
            items = [e for e in items if e.risk_score >= threshold]

        # sort
        if sort == "risk_desc":
            items.sort(key=lambda e: e.risk_score, reverse=True)
        elif sort == "risk_asc":
            items.sort(key=lambda e: e.risk_score)
        elif sort == "time_desc":
            items.sort(key=lambda e: e.timestamp, reverse=True)
        elif sort == "time_asc":
            items.sort(key=lambda e: e.timestamp)

        total = len(items)
        page = items[offset : offset + limit]
        return page, total

    def similar(
        self,
        target: ScoredEvent,
        *,
        top_k: int = 5,
    ) -> list[ScoredEvent]:
        """Top-k alerts with the same attack type and entity by risk proximity."""
        with self._lock:
            candidates = [
                e
                for e in self._deque
                if e.event_id != target.event_id
                and e.predicted_attack_type == target.predicted_attack_type
                and e.entity_id == target.entity_id
            ]
        candidates.sort(key=lambda e: abs(e.risk_score - target.risk_score))
        return candidates[:top_k]

    def __len__(self) -> int:
        return len(self._deque)


# --------------------------------------------------------------------------- #
# EntityStore
# --------------------------------------------------------------------------- #


class EntityStore:
    """Per-entity rolling summaries, updated on every :meth:`update` call."""

    def __init__(self) -> None:
        self._records: dict[str, _EntityRecord] = {}
        self._lock = threading.Lock()

    def update(self, event: ScoredEvent, cold_start_min_events: int = 25) -> None:
        """Incorporate a newly scored event into the entity's running state."""
        ts = event.timestamp.replace(tzinfo=UTC) if event.timestamp.tzinfo is None else event.timestamp
        with self._lock:
            rec = self._records.get(event.entity_id)
            if rec is None:
                rec = _EntityRecord(
                    entity_id=event.entity_id,
                    entity_type=event.entity_type,
                    cohort=event.cohort,
                    first_seen=ts,
                    last_seen=ts,
                )
                self._records[event.entity_id] = rec

            rec.last_seen = max(rec.last_seen, ts)
            rec.event_count += 1
            rec.total_risk += event.risk_score
            rec.cold_start = rec.event_count < cold_start_min_events

            if event.is_alert:
                rec.alert_count += 1

            hour = ts.hour
            rec.activity_by_hour[hour] += 1.0

            res = event.event.resource_accessed
            rec.resource_counts[res] = rec.resource_counts.get(res, 0) + 1
            if res not in rec.resource_first_seen:
                rec.resource_first_seen[res] = ts

            # keep timeline bounded at 1000 points
            rec.risk_timeline.append((ts, event.risk_score, event.is_alert))
            if len(rec.risk_timeline) > 1000:
                rec.risk_timeline = rec.risk_timeline[-1000:]

    def get(self, entity_id: str) -> _EntityRecord | None:
        return self._records.get(entity_id)

    def all_ids(self) -> list[str]:
        return list(self._records.keys())


# --------------------------------------------------------------------------- #
# StatsTracker
# --------------------------------------------------------------------------- #


class StatsTracker:
    """Running statistics for ``GET /api/stats``.

    ``events_per_sec`` is an exponential moving average updated on each
    :meth:`record_event` call.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._events_processed: int = 0
        self._alerts_raised: int = 0
        self._alerts_by_type: dict[str, int] = defaultdict(int)
        self._ema_eps: float = 0.0
        self._last_event_time: float | None = None
        self._stream_position: int = 0
        self._stream_total: int = 0

    def record_event(self, event: ScoredEvent) -> None:
        now = time.monotonic()
        with self._lock:
            self._events_processed += 1
            if event.is_alert:
                self._alerts_raised += 1
                self._alerts_by_type[event.predicted_attack_type] += 1

            # EMA of instantaneous event rate
            if self._last_event_time is not None:
                gap = now - self._last_event_time
                if gap > 0:
                    instant_rate = 1.0 / gap
                    self._ema_eps = _EMA_ALPHA * instant_rate + (1 - _EMA_ALPHA) * self._ema_eps
            self._last_event_time = now

    def set_stream_state(self, position: int, total: int) -> None:
        with self._lock:
            self._stream_position = position
            self._stream_total = total

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "events_processed": self._events_processed,
                "alerts_raised": self._alerts_raised,
                "alerts_by_type": dict(self._alerts_by_type),
                "events_per_sec": self._ema_eps,
                "uptime_s": time.monotonic() - self._start_time,
                "stream_position": self._stream_position,
                "stream_total": self._stream_total,
            }

    def reset(self) -> None:
        with self._lock:
            self._start_time = time.monotonic()
            self._events_processed = 0
            self._alerts_raised = 0
            self._alerts_by_type = defaultdict(int)
            self._ema_eps = 0.0
            self._last_event_time = None
            self._stream_position = 0


# --------------------------------------------------------------------------- #
# FeedbackStore
# --------------------------------------------------------------------------- #


@dataclass
class _FeedbackEntry:
    event_id: str
    verdict: str
    note: str | None
    received_at: datetime


class FeedbackStore:
    """Analyst feedback entries with per-entity effective-threshold adjustments."""

    # How much to lower the threshold per false-positive verdict (additive, 0-100 scale).
    _FP_THRESHOLD_DELTA = 5.0

    def __init__(self) -> None:
        self._entries: list[_FeedbackEntry] = []
        self._entity_threshold_lowering: dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def add(
        self,
        event_id: str,
        verdict: str,
        note: str | None,
        entity_id: str | None,
        alert_store: AlertStore,
    ) -> float | None:
        """Store feedback. Returns the new effective threshold if it changed."""
        entry = _FeedbackEntry(
            event_id=event_id,
            verdict=verdict,
            note=note,
            received_at=datetime.now(UTC),
        )
        with self._lock:
            self._entries.append(entry)

            updated: float | None = None
            if verdict == "false_positive" and entity_id:
                self._entity_threshold_lowering[entity_id] += self._FP_THRESHOLD_DELTA
                updated = max(
                    0.0, 40.0 - self._entity_threshold_lowering[entity_id]
                )

            return updated

    def effective_threshold(self, entity_id: str, base: float = 40.0) -> float:
        """Return the adjusted alert threshold for an entity."""
        lowering = self._entity_threshold_lowering.get(entity_id, 0.0)
        return max(0.0, base - lowering)

    def all(self) -> list[_FeedbackEntry]:
        with self._lock:
            return list(self._entries)
