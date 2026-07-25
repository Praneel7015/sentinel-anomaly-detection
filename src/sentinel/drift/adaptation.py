"""Concept drift adaptation.

AdaptiveBaseline wraps EntityState update with EWMA half-life.

update(event, risk_score, learn=True) logic:
- If risk_score < threshold OR analyst verdict == "false_positive" → learn normally.
- If risk_score >= threshold AND no analyst verdict → skip learning (poisoning guard).
- If drift detected → reset entity to cohort prior and restart learning.

analyst_feedback(event_id, verdict) retroactively marks an event and
potentially flips the learning decision for future events from this entity.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sentinel.drift.page_hinkley import PageHinkleyDetector

__all__ = ["AdaptiveBaseline"]

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 65.0  # from model.yaml profile.poisoning_guard.max_risk_for_update


@dataclass
class _EventRecord:
    """Stores enough info to potentially replay learning after analyst feedback."""

    event_id: str
    risk_score: float
    verdict: str | None = None  # "true_positive" | "false_positive" | "escalate" | None
    learned: bool = False


class AdaptiveBaseline:
    """Adaptive baseline controller for one entity.

    Wraps the poisoning guard, EWMA adaptation, and drift response.

    Usage::

        ab = AdaptiveBaseline(entity_id="usr_0001", cohort="finance_analyst")
        should_learn = ab.update(event, risk_score=75.0)
        ab.analyst_feedback("event_abc", verdict="false_positive")
    """

    def __init__(
        self,
        entity_id: str,
        cohort: str,
        max_risk_for_update: float = _DEFAULT_THRESHOLD,
        drift_delta: float = 0.005,
        drift_lambda: float = 50.0,
        drift_min_instances: int = 30,
        fast_half_life_days: float = 2.0,
        recovery_days: float = 7.0,
    ) -> None:
        self.entity_id = entity_id
        self.cohort = cohort
        self.max_risk_for_update = max_risk_for_update
        self.fast_half_life_days = fast_half_life_days
        self.recovery_days = recovery_days

        self.drift_detector = PageHinkleyDetector(
            delta=drift_delta,
            lambda_=drift_lambda,
            min_instances=drift_min_instances,
        )

        # Recent event records (bounded)
        self._recent: dict[str, _EventRecord] = {}
        self._max_records = 500

        # Drift state tracking
        self._drift_detected_at: datetime | None = None
        self._in_fast_rebaseline: bool = False
        self._fast_rebaseline_started: datetime | None = None

    # ---------------------------------------------------------------------- #
    # Update
    # ---------------------------------------------------------------------- #

    def update(
        self,
        event: dict[str, Any],
        risk_score: float,
        entity_state: Any | None = None,
        learn: bool = True,
    ) -> bool:
        """Decide whether to update the entity's baseline.

        Returns:
            True if the entity state should be updated (learn=True was granted).
        """
        event_id = str(event.get("event_id", ""))

        # Record event
        record = _EventRecord(event_id=event_id, risk_score=risk_score)
        self._record_event(event_id, record)

        # Update drift detector
        drift_now = self.drift_detector.update(risk_score)
        if drift_now:
            self._handle_drift(event, entity_state)
            record.learned = False
            return False

        # Poisoning guard: skip learning if risk is high and no FP verdict
        if not learn:
            record.learned = False
            return False

        existing_record = self._recent.get(event_id)
        if existing_record and existing_record.verdict == "false_positive":
            # Analyst already said this is benign → learn
            record.learned = True
            return True

        if risk_score >= self.max_risk_for_update:
            # High risk, no analyst verdict → skip
            logger.debug(
                "Skipping baseline update for %s (risk=%.1f >= threshold=%.1f)",
                self.entity_id,
                risk_score,
                self.max_risk_for_update,
            )
            record.learned = False
            return False

        record.learned = True
        return True

    def _handle_drift(
        self,
        event: dict[str, Any],
        entity_state: Any | None = None,
    ) -> None:
        """Handle detected drift: reset profiler to cohort prior."""
        now = datetime.now(UTC)
        self._drift_detected_at = now
        self._in_fast_rebaseline = True
        self._fast_rebaseline_started = now
        self.drift_detector.acknowledge_rebaseline()

        # Reset entity state to cohort prior if accessible
        if entity_state is not None and hasattr(entity_state, "reset_to_prior"):
            entity_state.reset_to_prior()

        logger.info(
            "Drift detected for entity %s at risk level. Fast rebaseline started.",
            self.entity_id,
        )

    def _record_event(self, event_id: str, record: _EventRecord) -> None:
        if len(self._recent) >= self._max_records:
            # Evict oldest
            oldest = next(iter(self._recent))
            del self._recent[oldest]
        self._recent[event_id] = record

    # ---------------------------------------------------------------------- #
    # Analyst feedback
    # ---------------------------------------------------------------------- #

    def analyst_feedback(self, event_id: str, verdict: str) -> None:
        """Retroactively mark an event with an analyst verdict.

        If verdict is "false_positive" and the event was not learned, mark it
        for future consideration. This doesn't replay learning immediately but
        affects the next threshold check for this entity.

        Args:
            event_id: the event to mark.
            verdict: "true_positive" | "false_positive" | "escalate".
        """
        if event_id in self._recent:
            self._recent[event_id].verdict = verdict
            logger.debug(
                "Analyst feedback for event %s: %s (entity %s)",
                event_id,
                verdict,
                self.entity_id,
            )
        else:
            logger.debug(
                "Event %s not in recent records for entity %s (may have expired)",
                event_id,
                self.entity_id,
            )

    # ---------------------------------------------------------------------- #
    # State
    # ---------------------------------------------------------------------- #

    @property
    def is_drifting(self) -> bool:
        return self._in_fast_rebaseline

    @property
    def drift_detected_at(self) -> datetime | None:
        return self._drift_detected_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "cohort": self.cohort,
            "max_risk_for_update": self.max_risk_for_update,
            "fast_half_life_days": self.fast_half_life_days,
            "recovery_days": self.recovery_days,
            "drift_detector": self.drift_detector.to_dict(),
            "_drift_detected_at": self._drift_detected_at.isoformat()
            if self._drift_detected_at
            else None,
            "_in_fast_rebaseline": self._in_fast_rebaseline,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AdaptiveBaseline:
        obj = cls(
            entity_id=d["entity_id"],
            cohort=d["cohort"],
            max_risk_for_update=d["max_risk_for_update"],
            fast_half_life_days=d["fast_half_life_days"],
            recovery_days=d["recovery_days"],
        )
        obj.drift_detector = PageHinkleyDetector.from_dict(d["drift_detector"])
        if d["_drift_detected_at"]:
            obj._drift_detected_at = datetime.fromisoformat(d["_drift_detected_at"])
        obj._in_fast_rebaseline = d["_in_fast_rebaseline"]
        return obj
