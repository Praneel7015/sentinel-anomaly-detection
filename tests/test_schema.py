"""Contract tests for the event/label schema and the parquet I/O layer."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from sentinel.io import (
    LabelLeakageError,
    SchemaError,
    read_events,
    read_labels,
    write_events,
    write_labels,
)
from sentinel.schema import (
    ALL_LABELS,
    ATTACK_TYPES,
    EVENT_FIELDS,
    LABEL_FIELDS,
    LABEL_NORMAL,
    AccessEvent,
    Label,
    events_arrow_schema,
    is_anomalous_label,
    labels_arrow_schema,
    risk_band,
)


def make_event(**overrides: object) -> AccessEvent:
    base: dict[str, object] = {
        "event_id": "0123456789abcdef",
        "episode_id": None,
        "entity_id": "usr_0001",
        "entity_type": "user",
        "cohort": "finance_analyst",
        "timestamp": datetime(2026, 2, 14, 9, 14, 22, tzinfo=UTC),
        "source_ip": "10.2.14.77",
        "geo_country": "DE",
        "geo_city": "Berlin",
        "geo_lat": 52.52,
        "geo_lon": 13.405,
        "resource_accessed": "fs:/finance/q3.xlsx",
        "resource_type": "file",
        "auth_method": "mfa_push",
        "auth_result": "success",
        "session_duration_s": 412.5,
        "command_sequence": ["open", "read", "export"],
        "device_os": "Windows",
        "device_os_version": "11.24H2",
        "device_mac": "3c:22:fb:11:0a:9e",
        "device_protocol": "https",
        "device_fingerprint": "fp_9a1c44e0",
        "bytes_transferred": 1_048_576,
        "split": "train",
    }
    base.update(overrides)
    return AccessEvent(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Schema shape
# --------------------------------------------------------------------------- #


def test_event_fields_match_model_and_arrow_schema() -> None:
    assert list(AccessEvent.model_fields) == EVENT_FIELDS
    assert events_arrow_schema().names == EVENT_FIELDS
    assert list(Label.model_fields) == LABEL_FIELDS
    assert labels_arrow_schema().names == LABEL_FIELDS


def test_naive_timestamps_are_interpreted_as_utc() -> None:
    event = make_event(timestamp=datetime(2026, 2, 14, 9, 14, 22))
    assert event.timestamp.tzinfo is not None
    assert event.timestamp.utcoffset() == timedelta(0)


def test_unknown_vocabulary_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="auth_result"):
        make_event(auth_result="maybe")


# --------------------------------------------------------------------------- #
# Anomaly semantics
# --------------------------------------------------------------------------- #


def test_only_attack_types_are_anomalous() -> None:
    for label in ATTACK_TYPES:
        assert is_anomalous_label(label) is True
    for label in (LABEL_NORMAL, "insider_drift", "benign_confounder"):
        assert is_anomalous_label(label) is False


def test_every_label_is_classifiable_and_unknown_labels_raise() -> None:
    for label in ALL_LABELS:
        assert isinstance(is_anomalous_label(label), bool)
    with pytest.raises(ValueError):
        is_anomalous_label("definitely_not_a_label")


def test_label_factory_derives_is_anomaly() -> None:
    assert Label.for_label("a" * 16, "insider_drift", episode_id="ep_1").is_anomaly is False
    assert Label.for_label("b" * 16, "brute_force", episode_id="ep_2").is_anomaly is True
    assert (
        Label.for_label("c" * 16, "benign_confounder", confounder_type="legit_travel").is_anomaly
        is False
    )


# --------------------------------------------------------------------------- #
# Risk bands
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, "low"),
        (39.99, "low"),
        (40.0, "medium"),
        (64.99, "medium"),
        (65.0, "high"),
        (84.99, "high"),
        (85.0, "critical"),
        (100.0, "critical"),
        (-5.0, "low"),
        (120.0, "critical"),
    ],
)
def test_risk_band_boundaries(score: float, expected: str) -> None:
    assert risk_band(score) == expected


# --------------------------------------------------------------------------- #
# Parquet round trip and leakage guard
# --------------------------------------------------------------------------- #


def test_event_round_trips_through_parquet(tmp_path: Path) -> None:
    original = make_event()
    frame = pd.DataFrame([original.to_row()])

    path = write_events(frame, tmp_path / "events.parquet")
    restored_frame = read_events(path)

    assert list(restored_frame.columns) == EVENT_FIELDS
    assert len(restored_frame) == 1

    restored = AccessEvent(**restored_frame.iloc[0].to_dict())
    assert restored.timestamp == original.timestamp
    assert list(restored.command_sequence) == list(original.command_sequence)
    assert restored.bytes_transferred == original.bytes_transferred
    assert restored.model_dump() == original.model_dump()


def test_labels_round_trip_through_parquet(tmp_path: Path) -> None:
    labels = [
        Label.for_label("0" * 16, LABEL_NORMAL),
        Label.for_label(
            "1" * 16, "low_and_slow_exfil", episode_id="ep_7", attack_stage="action_on_objective"
        ),
        Label.for_label(
            "2" * 16, "benign_confounder", episode_id="ep_8", confounder_type="vacation_return"
        ),
    ]
    frame = pd.DataFrame([label.model_dump() for label in labels])

    path = write_labels(frame, tmp_path / "labels.parquet")
    restored = read_labels(path)

    assert list(restored.columns) == LABEL_FIELDS
    assert restored["is_anomaly"].tolist() == [False, True, False]
    assert restored["episode_id"].isna().tolist() == [True, False, False]


def test_write_events_rejects_label_columns(tmp_path: Path) -> None:
    frame = pd.DataFrame([make_event().to_row()])
    frame["label"] = "brute_force"

    with pytest.raises(LabelLeakageError, match="label"):
        write_events(frame, tmp_path / "leaky.parquet")


def test_read_events_rejects_a_file_containing_a_label_column(tmp_path: Path) -> None:
    """A leaky file written by some other tool must still be refused on read."""
    frame = pd.DataFrame([make_event().to_row()])
    frame["label"] = "brute_force"
    frame["is_anomaly"] = True

    path = tmp_path / "leaky.parquet"
    frame.to_parquet(path, index=False)

    with pytest.raises(LabelLeakageError) as excinfo:
        read_events(path)
    assert "label" in str(excinfo.value)
    assert "is_anomaly" in str(excinfo.value)


def test_read_events_rejects_a_file_with_missing_columns(tmp_path: Path) -> None:
    frame = pd.DataFrame([make_event().to_row()]).drop(columns=["bytes_transferred"])
    path = tmp_path / "short.parquet"
    frame.to_parquet(path, index=False)

    with pytest.raises(SchemaError, match="bytes_transferred"):
        read_events(path)
