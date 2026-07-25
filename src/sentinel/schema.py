"""Single source of truth for the SENTINEL event and label schema.

Every other module in the project imports its field names, vocabularies and
Arrow schemas from here. Nothing downstream may redefine a field name locally.

Two rules are load-bearing for the whole project:

1. **Labels never live in the events table.** ``events.parquet`` holds only the
   observable columns in :data:`EVENT_FIELDS`. Ground truth lives in a separate
   ``labels.parquet`` (:data:`LABEL_FIELDS`) joined on ``event_id``. This makes
   "labels are hidden at inference time" a property of the file layout rather
   than of developer discipline. :func:`sentinel.io.read_events` enforces it.

2. **Not every labelled episode is an anomaly.** ``insider_drift`` and
   ``benign_confounder`` are labelled so that we can measure false positives on
   them, but they have ``is_anomaly = False``. Only the six
   :data:`ATTACK_TYPES` are true positives. Use
   :func:`is_anomalous_label` -- never ``label != "normal"``.

Schema extensions beyond the brief (justified in ``docs/CONTRACTS.md``):
``event_id``, ``episode_id``, ``auth_result``, ``bytes_transferred``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "ALL_LABELS",
    "ATTACK_STAGES",
    "ATTACK_TYPES",
    "AUTH_METHODS",
    "AUTH_RESULTS",
    "CONFOUNDER_TYPES",
    "DETECTOR_NAMES",
    "EDGE_CASE_TYPES",
    "EVENT_FIELDS",
    "LABEL_BENIGN_CONFOUNDER",
    "LABEL_FIELDS",
    "LABEL_NORMAL",
    "LABEL_UNKNOWN_NOVEL",
    "ENTITY_TYPES",
    "PREDICTED_TYPES",
    "PROTOCOLS",
    "RESOURCE_TYPES",
    "RISK_BANDS",
    "RISK_BAND_NAMES",
    "RISK_THRESHOLD_CRITICAL",
    "RISK_THRESHOLD_HIGH",
    "RISK_THRESHOLD_MEDIUM",
    "SPLITS",
    "AccessEvent",
    "Label",
    "RiskBand",
    "events_arrow_schema",
    "is_anomalous_label",
    "labels_arrow_schema",
    "risk_band",
]

# --------------------------------------------------------------------------- #
# Vocabularies
# --------------------------------------------------------------------------- #

#: The three kinds of principal that generate access events.
ENTITY_TYPES: list[str] = ["user", "service_account", "edge_device"]

#: How the principal authenticated for this event.
AUTH_METHODS: list[str] = ["password", "token", "certificate", "biometric", "mfa_push"]

#: Outcome of the authentication attempt. SCHEMA EXTENSION: without a failure
#: outcome, brute force and credential stuffing are literally unexpressible.
AUTH_RESULTS: list[str] = ["success", "failure"]

#: Kind of thing that ``resource_accessed`` points at. The ``resource_accessed``
#: string is always ``"<prefix>:<path>"`` where prefix is one of
#: ``fs`` (file), ``api`` (endpoint), ``port`` (port), ``func`` (device_function).
RESOURCE_TYPES: list[str] = ["file", "endpoint", "port", "device_function"]

#: Prefix used in ``resource_accessed`` for each resource type.
RESOURCE_TYPE_PREFIX: dict[str, str] = {
    "file": "fs",
    "endpoint": "api",
    "port": "port",
    "device_function": "func",
}

#: Wire protocol the session ran over.
PROTOCOLS: list[str] = ["https", "ssh", "rdp", "smb", "modbus", "mqtt"]

#: Temporal dataset splits. Ordered chronologically: train < val < test.
SPLITS: list[str] = ["train", "val", "test"]

#: The six injected attack families. These -- and only these -- are anomalies.
ATTACK_TYPES: list[str] = [
    "brute_force",
    "impossible_travel",
    "credential_stuffing",
    "lateral_movement",
    "device_spoofing",
    "low_and_slow_exfil",
]

#: Ambiguous edge cases. Labelled, grouped into episodes, but NOT anomalies:
#: an insider whose role genuinely changed should stop alerting once the drift
#: detector re-baselines them. These exist to measure false-positive tuning.
EDGE_CASE_TYPES: list[str] = ["insider_drift"]

#: Label for ordinary behaviour.
LABEL_NORMAL: str = "normal"

#: Label for a deliberately anomaly-looking but legitimate episode.
LABEL_BENIGN_CONFOUNDER: str = "benign_confounder"

#: Every value the ``label`` column may take.
ALL_LABELS: list[str] = [LABEL_NORMAL, *ATTACK_TYPES, *EDGE_CASE_TYPES, LABEL_BENIGN_CONFOUNDER]

#: Sub-kinds of ``benign_confounder``; recorded in ``Label.confounder_type``.
#: These are the traps: each one trips a naive rule but is entirely legitimate.
CONFOUNDER_TYPES: list[str] = [
    "legit_travel",
    "new_device_enrollment",
    "password_rotation",
    "vacation_return",
    "maintenance_burst",
]

#: Coarse kill-chain position of an event inside an attack episode. Recorded in
#: ``Label.attack_stage`` so mean-time-to-detect can be measured per stage.
ATTACK_STAGES: list[str] = ["recon", "initial_access", "escalation", "action_on_objective"]

#: Label emitted by the serving layer when the supervised classifier and the
#: signature matcher disagree, i.e. "anomalous but we cannot name it".
LABEL_UNKNOWN_NOVEL: str = "unknown_novel"

#: Values ``ScoredEvent.predicted_attack_type`` may take.
PREDICTED_TYPES: list[str] = [LABEL_NORMAL, *ATTACK_TYPES, LABEL_UNKNOWN_NOVEL]

#: Detector names, in fusion order. Keys of ``detector_scores`` on the wire,
#: of the fusion weights in ``configs/model.yaml``, and of the model registry.
#: ``gru`` is optional (null when torch is unavailable).
DETECTOR_NAMES: list[str] = ["profile", "isolation", "sequence", "graph", "gru"]

#: Detectors that must always be present for a score to be valid.
REQUIRED_DETECTORS: list[str] = ["profile", "isolation", "sequence", "graph"]

#: Detectors that may be ``None``.
OPTIONAL_DETECTORS: list[str] = ["gru"]


# --------------------------------------------------------------------------- #
# Anomaly semantics
# --------------------------------------------------------------------------- #


def is_anomalous_label(label: str) -> bool:
    """Return whether ``label`` counts as a positive for detection metrics.

    ``True`` for the six :data:`ATTACK_TYPES` only. ``normal``,
    ``benign_confounder`` and ``insider_drift`` are all negatives -- alerting on
    them is a false positive, which is exactly what those classes are there to
    measure.

    Raises:
        ValueError: if ``label`` is not in :data:`ALL_LABELS`.
    """
    if label not in ALL_LABELS:
        raise ValueError(f"unknown label {label!r}; expected one of {ALL_LABELS}")
    return label in ATTACK_TYPES


# --------------------------------------------------------------------------- #
# Risk bands
# --------------------------------------------------------------------------- #

RISK_THRESHOLD_MEDIUM: float = 40.0
RISK_THRESHOLD_HIGH: float = 65.0
RISK_THRESHOLD_CRITICAL: float = 85.0


@dataclass(frozen=True, slots=True)
class RiskBand:
    """A half-open risk interval ``[lower, upper)`` on the 0-100 risk score."""

    name: str
    lower: float
    upper: float

    def contains(self, score: float) -> bool:
        return self.lower <= score < self.upper


#: Bands over the 0-100 risk score. Half-open on the right, so the boundaries
#: are unambiguous: 40 is medium, 64.9 is medium, 65 is high, 85 is critical.
RISK_BANDS: tuple[RiskBand, ...] = (
    RiskBand("low", 0.0, RISK_THRESHOLD_MEDIUM),
    RiskBand("medium", RISK_THRESHOLD_MEDIUM, RISK_THRESHOLD_HIGH),
    RiskBand("high", RISK_THRESHOLD_HIGH, RISK_THRESHOLD_CRITICAL),
    RiskBand("critical", RISK_THRESHOLD_CRITICAL, float("inf")),
)

RISK_BAND_NAMES: list[str] = [b.name for b in RISK_BANDS]


def risk_band(score: float) -> str:
    """Map a 0-100 risk score onto ``low`` / ``medium`` / ``high`` / ``critical``.

    Scores below 0 clamp to ``low`` and scores above 100 clamp to ``critical``,
    so a mis-calibrated detector can never produce an unrenderable band.
    """
    value = float(score)
    if value >= RISK_THRESHOLD_CRITICAL:
        return "critical"
    for band in RISK_BANDS:
        if band.contains(value):
            return band.name
    # Reachable only for score < 0 or NaN.
    return "low"


# --------------------------------------------------------------------------- #
# Event schema
# --------------------------------------------------------------------------- #

#: Column order of ``events.parquet``. Authoritative -- writers must emit this
#: order and readers may rely on it.
EVENT_FIELDS: list[str] = [
    "event_id",
    "episode_id",
    "entity_id",
    "entity_type",
    "cohort",
    "timestamp",
    "source_ip",
    "geo_country",
    "geo_city",
    "geo_lat",
    "geo_lon",
    "resource_accessed",
    "resource_type",
    "auth_method",
    "auth_result",
    "session_duration_s",
    "command_sequence",
    "device_os",
    "device_os_version",
    "device_mac",
    "device_protocol",
    "device_fingerprint",
    "bytes_transferred",
    "split",
]


class AccessEvent(BaseModel):
    """One observable access event. This is everything a detector may look at.

    Note that ``episode_id`` is observable but carries no ground truth: it only
    says "these events belong to one causally related burst". Baseline events
    have ``episode_id = None``. Confounder episodes have one too, so the field
    cannot be used to shortcut detection.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    event_id: str = Field(description="16 lowercase hex chars, deterministic from the run seed")
    episode_id: str | None = Field(
        default=None,
        description="Groups events of one injected episode (attack, edge case or "
        "confounder). None for ordinary baseline activity.",
    )

    entity_id: str = Field(description="usr_0001 | svc_0001 | dev_0001")
    entity_type: str = Field(description="user | service_account | edge_device")
    cohort: str = Field(description="Role cohort, e.g. finance_analyst, plc_gateway")

    timestamp: datetime = Field(description="Event time, always timezone-aware UTC")

    source_ip: str
    geo_country: str = Field(description="ISO 3166-1 alpha-2")
    geo_city: str
    geo_lat: float = Field(ge=-90.0, le=90.0)
    geo_lon: float = Field(ge=-180.0, le=180.0)

    resource_accessed: str = Field(
        description='"fs:/finance/q3.xlsx" | "api:/v1/payments" | "port:445" | "func:valve_setpoint"'
    )
    resource_type: str = Field(description="file | endpoint | port | device_function")

    auth_method: str = Field(description="password | token | certificate | biometric | mfa_push")
    auth_result: str = Field(description="success | failure  [SCHEMA EXTENSION]")

    session_duration_s: float = Field(ge=0.0)
    command_sequence: list[str] = Field(
        default_factory=list,
        description="Ordered actions taken in the session; empty for non-privileged sessions",
    )

    device_os: str
    device_os_version: str
    device_mac: str
    device_protocol: str = Field(description="https | ssh | rdp | smb | modbus | mqtt")
    device_fingerprint: str = Field(
        description="Stable hash of os|version|mac|protocol; a change means a new device"
    )

    bytes_transferred: int = Field(ge=0, description="[SCHEMA EXTENSION]")

    split: str = Field(description="train | val | test")

    # -- validators ---------------------------------------------------------- #

    @field_validator("timestamp")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        """Naive timestamps are interpreted as UTC; aware ones are converted."""
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @field_validator("entity_type")
    @classmethod
    def _entity_type(cls, value: str) -> str:
        return _one_of(value, ENTITY_TYPES, "entity_type")

    @field_validator("resource_type")
    @classmethod
    def _resource_type(cls, value: str) -> str:
        return _one_of(value, RESOURCE_TYPES, "resource_type")

    @field_validator("auth_method")
    @classmethod
    def _auth_method(cls, value: str) -> str:
        return _one_of(value, AUTH_METHODS, "auth_method")

    @field_validator("auth_result")
    @classmethod
    def _auth_result(cls, value: str) -> str:
        return _one_of(value, AUTH_RESULTS, "auth_result")

    @field_validator("device_protocol")
    @classmethod
    def _protocol(cls, value: str) -> str:
        return _one_of(value, PROTOCOLS, "device_protocol")

    @field_validator("split")
    @classmethod
    def _split(cls, value: str) -> str:
        return _one_of(value, SPLITS, "split")

    # -- helpers ------------------------------------------------------------- #

    def to_row(self) -> dict[str, Any]:
        """Return a plain dict in :data:`EVENT_FIELDS` order, ready for pandas."""
        dumped = self.model_dump()
        return {name: dumped[name] for name in EVENT_FIELDS}


class Label(BaseModel):
    """Ground truth for one event. Lives in ``labels.parquet``, never in events.

    ``is_anomaly`` is redundant with ``label`` by construction (it is derived by
    :func:`is_anomalous_label`) but is materialised so that evaluation code
    cannot accidentally count ``insider_drift`` or ``benign_confounder`` as a
    positive just because they are not ``normal``.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    event_id: str
    episode_id: str | None = Field(
        default=None, description="Matches AccessEvent.episode_id; None for baseline events"
    )
    label: str = Field(description="One of ALL_LABELS")
    is_anomaly: bool = Field(
        description="True only for the six ATTACK_TYPES. insider_drift and "
        "benign_confounder are False -- alerting on them is a false positive."
    )
    confounder_type: str | None = Field(
        default=None, description="One of CONFOUNDER_TYPES when label == 'benign_confounder'"
    )
    attack_stage: str | None = Field(
        default=None, description="One of ATTACK_STAGES for events inside an attack episode"
    )

    @field_validator("label")
    @classmethod
    def _label(cls, value: str) -> str:
        return _one_of(value, ALL_LABELS, "label")

    @field_validator("confounder_type")
    @classmethod
    def _confounder(cls, value: str | None) -> str | None:
        return None if value is None else _one_of(value, CONFOUNDER_TYPES, "confounder_type")

    @field_validator("attack_stage")
    @classmethod
    def _stage(cls, value: str | None) -> str | None:
        return None if value is None else _one_of(value, ATTACK_STAGES, "attack_stage")

    @classmethod
    def for_label(
        cls,
        event_id: str,
        label: str,
        *,
        episode_id: str | None = None,
        confounder_type: str | None = None,
        attack_stage: str | None = None,
    ) -> Label:
        """Build a :class:`Label` with ``is_anomaly`` derived, never hand-set."""
        return cls(
            event_id=event_id,
            episode_id=episode_id,
            label=label,
            is_anomaly=is_anomalous_label(label),
            confounder_type=confounder_type,
            attack_stage=attack_stage,
        )


#: Column order of ``labels.parquet``.
LABEL_FIELDS: list[str] = [
    "event_id",
    "episode_id",
    "label",
    "is_anomaly",
    "confounder_type",
    "attack_stage",
]

#: Label columns that must never appear in ``events.parquet``. ``event_id`` and
#: ``episode_id`` are excluded because they are legitimate join keys.
LABEL_ONLY_FIELDS: list[str] = [f for f in LABEL_FIELDS if f not in {"event_id", "episode_id"}]


# --------------------------------------------------------------------------- #
# Arrow schemas
# --------------------------------------------------------------------------- #

_TIMESTAMP_TYPE = pa.timestamp("us", tz="UTC")


def events_arrow_schema() -> pa.Schema:
    """Arrow schema for ``events.parquet``, in :data:`EVENT_FIELDS` order."""
    return pa.schema(
        [
            pa.field("event_id", pa.string(), nullable=False),
            pa.field("episode_id", pa.string(), nullable=True),
            pa.field("entity_id", pa.string(), nullable=False),
            pa.field("entity_type", pa.string(), nullable=False),
            pa.field("cohort", pa.string(), nullable=False),
            pa.field("timestamp", _TIMESTAMP_TYPE, nullable=False),
            pa.field("source_ip", pa.string(), nullable=False),
            pa.field("geo_country", pa.string(), nullable=False),
            pa.field("geo_city", pa.string(), nullable=False),
            pa.field("geo_lat", pa.float64(), nullable=False),
            pa.field("geo_lon", pa.float64(), nullable=False),
            pa.field("resource_accessed", pa.string(), nullable=False),
            pa.field("resource_type", pa.string(), nullable=False),
            pa.field("auth_method", pa.string(), nullable=False),
            pa.field("auth_result", pa.string(), nullable=False),
            pa.field("session_duration_s", pa.float64(), nullable=False),
            pa.field("command_sequence", pa.list_(pa.string()), nullable=False),
            pa.field("device_os", pa.string(), nullable=False),
            pa.field("device_os_version", pa.string(), nullable=False),
            pa.field("device_mac", pa.string(), nullable=False),
            pa.field("device_protocol", pa.string(), nullable=False),
            pa.field("device_fingerprint", pa.string(), nullable=False),
            pa.field("bytes_transferred", pa.int64(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
        ]
    )


def labels_arrow_schema() -> pa.Schema:
    """Arrow schema for ``labels.parquet``, in :data:`LABEL_FIELDS` order."""
    return pa.schema(
        [
            pa.field("event_id", pa.string(), nullable=False),
            pa.field("episode_id", pa.string(), nullable=True),
            pa.field("label", pa.string(), nullable=False),
            pa.field("is_anomaly", pa.bool_(), nullable=False),
            pa.field("confounder_type", pa.string(), nullable=True),
            pa.field("attack_stage", pa.string(), nullable=True),
        ]
    )


def _one_of(value: str, allowed: list[str], field_name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{field_name}={value!r} is not one of {allowed}")
    return value
