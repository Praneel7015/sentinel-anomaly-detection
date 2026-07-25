"""Locks the wire contract in ``docs/CONTRACTS.md`` against the pydantic models.

If the serving agent or the dashboard agent needs a different shape, this test
is the thing that must change first -- deliberately, and in one place.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from sentinel.schema import ATTACK_TYPES, EVENT_FIELDS, AccessEvent
from sentinel.serving.models import (
    Contribution,
    Counterfactual,
    DetectorScores,
    GroundTruth,
    PredictedAttackType,
    ScoredEvent,
    StreamControlRequest,
)

SCORED_EVENT_KEYS = [
    "event_id",
    "entity_id",
    "entity_type",
    "cohort",
    "timestamp",
    "risk_score",
    "risk_band",
    "is_alert",
    "predicted_attack_type",
    "attack_type_confidence",
    "classifier_agreement",
    "is_novel",
    "detector_scores",
    "contributions",
    "narrative",
    "counterfactuals",
    "cold_start",
    "entity_event_count",
    "event",
    "ground_truth",
]


@pytest.fixture
def raw_event() -> AccessEvent:
    return AccessEvent(
        event_id="0123456789abcdef",
        episode_id="ep_000123",
        entity_id="usr_0001",
        entity_type="user",
        cohort="finance_analyst",
        timestamp=datetime(2026, 2, 14, 9, 14, 22, tzinfo=UTC),
        source_ip="10.2.14.77",
        geo_country="DE",
        geo_city="Berlin",
        geo_lat=52.52,
        geo_lon=13.405,
        resource_accessed="fs:/finance/q3.xlsx",
        resource_type="file",
        auth_method="mfa_push",
        auth_result="success",
        session_duration_s=412.5,
        command_sequence=["open", "read", "export"],
        device_os="Windows",
        device_os_version="11.24H2",
        device_mac="3c:22:fb:11:0a:9e",
        device_protocol="https",
        device_fingerprint="fp_9a1c44e0",
        bytes_transferred=1_048_576,
        split="test",
    )


@pytest.fixture
def scored(raw_event: AccessEvent) -> ScoredEvent:
    return ScoredEvent(
        event_id=raw_event.event_id,
        entity_id=raw_event.entity_id,
        entity_type=raw_event.entity_type,
        cohort=raw_event.cohort,
        timestamp=raw_event.timestamp,
        risk_score=87.4,
        risk_band="critical",
        is_alert=True,
        predicted_attack_type="low_and_slow_exfil",
        attack_type_confidence=0.71,
        classifier_agreement=True,
        is_novel=False,
        detector_scores=DetectorScores(
            profile=0.99, isolation=0.94, sequence=0.62, graph=0.88, gru=None
        ),
        contributions=[
            Contribution(
                feature="offhours_bytes_7d",
                display_name="Off-hours data volume (7d)",
                value=1_048_576.0,
                display_value="1.0 MB",
                contribution=1.84,
                direction="increases",
                description="12x this entity's usual off-hours transfer volume",
            )
        ],
        narrative="usr_0001 moved 1.0 MB to a rarely used share outside working hours.",
        counterfactuals=[
            Counterfactual(
                feature="offhours_bytes_7d",
                display_name="Off-hours data volume (7d)",
                neutralised_risk=41.2,
                delta=-46.2,
            )
        ],
        cold_start=False,
        entity_event_count=1482,
        event=raw_event,
        ground_truth=GroundTruth(label="low_and_slow_exfil", is_anomaly=True),
    )


def test_scored_event_serialises_to_the_documented_shape(scored: ScoredEvent) -> None:
    payload = json.loads(scored.model_dump_json())
    assert list(payload) == SCORED_EVENT_KEYS
    assert list(payload["detector_scores"]) == ["profile", "isolation", "sequence", "graph", "gru"]
    assert payload["detector_scores"]["gru"] is None
    assert list(payload["event"]) == EVENT_FIELDS


def test_timestamps_use_the_same_utc_format_everywhere(scored: ScoredEvent) -> None:
    payload = json.loads(scored.model_dump_json())
    assert payload["timestamp"] == "2026-02-14T09:14:22Z"
    assert payload["event"]["timestamp"] == payload["timestamp"]


def test_predicted_attack_type_covers_every_attack_plus_normal_and_novel() -> None:
    allowed = set(PredictedAttackType.__args__)  # type: ignore[attr-defined]
    assert allowed == {*ATTACK_TYPES, "normal", "unknown_novel"}


def test_ground_truth_is_omittable(raw_event: AccessEvent, scored: ScoredEvent) -> None:
    production = scored.model_copy(update={"ground_truth": None})
    assert json.loads(production.model_dump_json())["ground_truth"] is None


def test_models_reject_unknown_fields(scored: ScoredEvent) -> None:
    payload = json.loads(scored.model_dump_json())
    payload["surprise"] = 1
    with pytest.raises(ValidationError):
        ScoredEvent.model_validate(payload)


def test_stream_control_rejects_a_bad_action() -> None:
    assert StreamControlRequest(action="pause").speed is None
    with pytest.raises(ValidationError):
        StreamControlRequest(action="rewind")
