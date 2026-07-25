"""Pydantic response/request models for the SENTINEL REST + SSE API.

This module is the wire contract between the FastAPI service and the React
dashboard. Both sides import (or mirror) these shapes, so neither can drift.
The prose version, including the endpoint table, lives in ``docs/CONTRACTS.md``
and must be kept in sync with this file.

Conventions:

* Every model forbids extra fields. Adding a field to the wire format means
  editing this file, which means the change is reviewed rather than absorbed.
* Timestamps are timezone-aware UTC and serialise to ISO-8601 with a ``Z``
  suffix, e.g. ``2026-02-14T09:14:22Z``.
* ``risk_score`` is 0-100. Detector scores and confidences are 0-1.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from sentinel.schema import AccessEvent

__all__ = [
    "AlertDetailResponse",
    "AlertSort",
    "AlertsResponse",
    "AblationRow",
    "BudgetPoint",
    "ColdStartMetrics",
    "ConfusionMatrix",
    "Contribution",
    "ContributionDirection",
    "Counterfactual",
    "DetectorScores",
    "DriftState",
    "EntityDetailResponse",
    "EntitySummary",
    "FeedbackRequest",
    "FeedbackResponse",
    "FeedbackVerdict",
    "GeneralisationResult",
    "GroundTruth",
    "HealthResponse",
    "LatencyStats",
    "MetricsResponse",
    "MttdEntry",
    "PeerComparison",
    "PerAttackRecall",
    "PrPoint",
    "PredictedAttackType",
    "ProfileSummaryItem",
    "ResourceUsage",
    "RiskBandName",
    "RiskTimelinePoint",
    "ScoredEvent",
    "StatsResponse",
    "StreamAction",
    "StreamControlRequest",
    "StreamControlResponse",
    "StreamState",
    "SubgroupMetrics",
]

# --------------------------------------------------------------------------- #
# Shared aliases
# --------------------------------------------------------------------------- #


def _to_utc(value: datetime) -> datetime:
    """Naive input is read as UTC; aware input is converted to UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


#: Always timezone-aware UTC. Pydantic serialises these to ISO-8601 with a ``Z``
#: suffix (``2026-02-14T09:14:22Z``), matching ``AccessEvent.timestamp`` exactly,
#: so the browser's ``new Date(...)`` is unambiguous.
UtcDatetime = Annotated[datetime, AfterValidator(_to_utc)]

RiskBandName = Literal["low", "medium", "high", "critical"]

PredictedAttackType = Literal[
    "normal",
    "brute_force",
    "impossible_travel",
    "credential_stuffing",
    "lateral_movement",
    "device_spoofing",
    "low_and_slow_exfil",
    "unknown_novel",
]

ContributionDirection = Literal["increases", "decreases"]

FeedbackVerdict = Literal["true_positive", "false_positive", "escalate"]

StreamAction = Literal["start", "pause", "reset"]
StreamState = Literal["running", "paused", "stopped"]

#: Accepted values of the ``sort`` query parameter on ``GET /api/alerts``.
AlertSort = Literal["risk_desc", "risk_asc", "time_desc", "time_asc"]


class _Model(BaseModel):
    """Base: strict on input, stable on output."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# --------------------------------------------------------------------------- #
# ScoredEvent and its parts
# --------------------------------------------------------------------------- #


class DetectorScores(_Model):
    """Per-detector normalised anomaly scores in 0-1 (1 - p_value).

    ``gru`` is ``null`` when the optional torch detector is not loaded; the
    fusion weights are renormalised in that case so scores stay comparable.
    """

    profile: float = Field(ge=0.0, le=1.0)
    isolation: float = Field(ge=0.0, le=1.0)
    sequence: float = Field(ge=0.0, le=1.0)
    graph: float = Field(ge=0.0, le=1.0)
    gru: float | None = Field(default=None, ge=0.0, le=1.0)


class Contribution(_Model):
    """One additive term of the fused log-odds score.

    Because fusion is a weighted sum in log-odds space, ``contribution`` is the
    exact term this feature added to the logit -- not a SHAP approximation.
    """

    feature: str = Field(description="Machine name, e.g. 'geo_velocity_kmh'")
    display_name: str = Field(description="Analyst-facing label, e.g. 'Travel speed'")
    value: float | str | bool | None = Field(description="Raw feature value")
    display_value: str = Field(description="Formatted value, e.g. '1,840 km/h'")
    contribution: float = Field(description="Signed log-odds contribution")
    direction: ContributionDirection
    description: str = Field(description="One clause explaining why this is unusual")


class Counterfactual(_Model):
    """What the risk would have been with one factor forced to its normal value."""

    feature: str
    display_name: str
    neutralised_risk: float = Field(ge=0.0, le=100.0)
    delta: float = Field(
        description="neutralised_risk - risk_score; negative means it drove the alert"
    )


class GroundTruth(_Model):
    """Present only in demo/eval mode. Never populated in a production path."""

    label: str
    is_anomaly: bool


class ScoredEvent(_Model):
    """The canonical alert object. Returned by /score, /alerts and the SSE stream."""

    event_id: str
    entity_id: str
    entity_type: str
    cohort: str
    timestamp: UtcDatetime

    risk_score: float = Field(ge=0.0, le=100.0)
    risk_band: RiskBandName
    is_alert: bool = Field(description="risk_score is above the current alert-budget threshold")

    predicted_attack_type: PredictedAttackType
    attack_type_confidence: float = Field(ge=0.0, le=1.0)
    classifier_agreement: bool = Field(
        description="HistGBM and the transparent signature matcher agree"
    )
    is_novel: bool = Field(description="Anomalous but unattributable to a known family")

    detector_scores: DetectorScores
    contributions: list[Contribution] = Field(default_factory=list)
    narrative: str = Field(description="Human-readable summary for the SOC analyst")
    counterfactuals: list[Counterfactual] = Field(default_factory=list)

    cold_start: bool = Field(description="Entity had fewer than cold_start.min_events of history")
    entity_event_count: int = Field(ge=0, description="Events seen for this entity before now")

    event: AccessEvent = Field(description="The raw, unmodified access event")
    ground_truth: GroundTruth | None = None


# --------------------------------------------------------------------------- #
# GET /api/health, /api/stats
# --------------------------------------------------------------------------- #


class HealthResponse(_Model):
    status: Literal["ok", "degraded"]
    version: str
    model_loaded: bool
    torch_available: bool


class StatsResponse(_Model):
    events_processed: int
    alerts_raised: int
    alerts_by_type: dict[str, int] = Field(
        default_factory=dict, description="predicted_attack_type -> count"
    )
    events_per_sec: float
    uptime_s: float
    stream_position: int
    stream_total: int


# --------------------------------------------------------------------------- #
# GET /api/alerts
# --------------------------------------------------------------------------- #


class AlertsResponse(_Model):
    alerts: list[ScoredEvent]
    total: int = Field(description="Matching alerts before limit/offset")


class EntitySummary(_Model):
    """Compact entity context attached to an alert detail response."""

    entity_id: str
    entity_type: str
    cohort: str
    event_count: int
    cold_start: bool
    first_seen: UtcDatetime | None = None
    last_seen: UtcDatetime | None = None
    alert_count: int = 0
    mean_risk: float = 0.0


class AlertDetailResponse(ScoredEvent):
    """``GET /api/alerts/{event_id}``: a ScoredEvent plus investigation context."""

    entity_summary: EntitySummary
    similar_alerts: list[ScoredEvent] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# GET /api/entities/{entity_id}
# --------------------------------------------------------------------------- #


class ProfileSummaryItem(_Model):
    """One row of the "this entity vs its cohort" table."""

    label: str
    value: float | str
    cohort_value: float | str


class RiskTimelinePoint(_Model):
    timestamp: UtcDatetime
    risk_score: float
    is_alert: bool


class ResourceUsage(_Model):
    resource: str
    count: int
    is_new: bool = Field(description="First seen for this entity within the observed window")


class PeerComparison(_Model):
    axis: str = Field(description="Radar/bar axis name, e.g. 'off-hours %'")
    entity: float
    cohort_median: float


class DriftState(_Model):
    drifting: bool
    detected_at: UtcDatetime | None = None
    adapted: bool = False


class EntityDetailResponse(_Model):
    entity_id: str
    entity_type: str
    cohort: str
    first_seen: UtcDatetime | None = None
    last_seen: UtcDatetime | None = None
    event_count: int
    cold_start: bool
    profile_summary: list[ProfileSummaryItem] = Field(default_factory=list)
    risk_timeline: list[RiskTimelinePoint] = Field(default_factory=list)
    activity_by_hour: list[float] = Field(
        default_factory=lambda: [0.0] * 24,
        min_length=24,
        max_length=24,
        description="Event counts bucketed by UTC hour 0..23",
    )
    top_resources: list[ResourceUsage] = Field(default_factory=list)
    peer_comparison: list[PeerComparison] = Field(default_factory=list)
    drift_state: DriftState


# --------------------------------------------------------------------------- #
# POST /api/feedback
# --------------------------------------------------------------------------- #


class FeedbackRequest(_Model):
    event_id: str
    verdict: FeedbackVerdict
    note: str | None = None


class FeedbackResponse(_Model):
    ok: bool
    updated_threshold: float | None = Field(
        default=None, description="New alert threshold if the feedback moved it"
    )


# --------------------------------------------------------------------------- #
# GET /api/metrics
# --------------------------------------------------------------------------- #


class BudgetPoint(_Model):
    budget_pct: float
    precision: float
    recall: float
    alerts: int
    analyst_hours: float


class PerAttackRecall(_Model):
    attack_type: str
    recall: float
    support: int = Field(description="Ground-truth episodes/events of this type")
    detected: int


class ConfusionMatrix(_Model):
    labels: list[str]
    matrix: list[list[int]] = Field(description="Row = true label, column = predicted label")


class PrPoint(_Model):
    recall: float
    precision: float


class MttdEntry(_Model):
    """Mean time to detect, from the first event of an episode to its first alert."""

    attack_type: str
    mean_events: float
    mean_minutes: float


class SubgroupMetrics(_Model):
    precision: float
    recall: float
    n_events: int = 0


#: Cold-start and post-drift subgroups share the precision/recall shape.
ColdStartMetrics = SubgroupMetrics


class LatencyStats(_Model):
    p50: float
    p95: float
    p99: float
    mean: float


class AblationRow(_Model):
    variant: str = Field(description="e.g. 'no_graph', 'profile_only', 'full'")
    pr_auc: float
    precision_at_1pct: float


class GeneralisationResult(_Model):
    held_out_attack: str
    unsupervised_recall: float
    n_events: int = 0


class MetricsResponse(_Model):
    pr_auc: float
    roc_auc: float
    budget_curve: list[BudgetPoint] = Field(default_factory=list)
    per_attack_recall: list[PerAttackRecall] = Field(default_factory=list)
    confusion_matrix: ConfusionMatrix
    pr_curve: list[PrPoint] = Field(default_factory=list)
    fp_rate_confounders: float
    fp_rate_insider_drift: float
    mttd: list[MttdEntry] = Field(default_factory=list)
    cold_start: SubgroupMetrics
    post_drift: SubgroupMetrics
    latency_ms: LatencyStats
    ablation: list[AblationRow] = Field(default_factory=list)
    generalisation: GeneralisationResult | None = None
    n_test_events: int = 0
    n_test_anomalies: int = 0
    anomaly_rate_pct: float = 0.0
    threshold_at_1pct: float = 0.0


# --------------------------------------------------------------------------- #
# POST /api/stream/control
# --------------------------------------------------------------------------- #


class StreamControlRequest(_Model):
    action: StreamAction
    speed: float | None = Field(default=None, gt=0.0, description="Events per second")


class StreamControlResponse(_Model):
    ok: bool
    state: StreamState


def scored_event_json_schema() -> dict[str, Any]:
    """JSON Schema for :class:`ScoredEvent`; handy for generating TS types."""
    return ScoredEvent.model_json_schema()
