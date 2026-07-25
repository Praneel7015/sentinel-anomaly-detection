"""Single entry point for event scoring.

score_event() extracts features, scores all detectors, fuses, calibrates,
classifies, generates attribution + narrative + counterfactuals, updates drift
detectors, and returns a fully populated ScoredEvent.

This is what the FastAPI /score endpoint calls.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sentinel.explain.attribution import attribute_contributions
from sentinel.explain.counterfactual import compute_counterfactuals
from sentinel.explain.narrative import build_narrative
from sentinel.features.cohort import is_cold_start
from sentinel.features.extractors import FeatureVector, extract_all
from sentinel.features.pipeline import PipelineState
from sentinel.models.classifier import AttackClassifier
from sentinel.models.registry import ModelRegistry
from sentinel.schema import LABEL_NORMAL, risk_band
from sentinel.serving.models import GroundTruth, ScoredEvent

__all__ = ["score_event"]

logger = logging.getLogger(__name__)

_DEFAULT_ALERT_THRESHOLD = 65.0


def score_event(
    event: Any,  # AccessEvent or dict
    pipeline_state: PipelineState,
    registry: ModelRegistry,
    calibrator: Any | None = None,  # IsotonicCalibrator (optional)
    classifier: AttackClassifier | None = None,
    alert_threshold: float = _DEFAULT_ALERT_THRESHOLD,
    top_k_contributions: int = 6,
    counterfactual_top_k: int = 3,
    label: str | None = None,  # for demo/eval ground truth
) -> ScoredEvent:
    """Score one event end-to-end.

    Args:
        event: AccessEvent pydantic model or equivalent dict.
        pipeline_state: current PipelineState (updated inside this call).
        registry: fitted ModelRegistry.
        calibrator: optional IsotonicCalibrator for probability calibration.
        classifier: optional AttackClassifier.
        alert_threshold: risk score above which is_alert = True.
        top_k_contributions: max contributions to return.
        counterfactual_top_k: max counterfactuals to return.
        label: ground truth label (demo/eval mode only).

    Returns:
        Fully populated ScoredEvent.
    """
    # --- Normalise event to dict ---
    if hasattr(event, "model_dump"):
        event_dict = event.model_dump()
    elif hasattr(event, "to_dict") and hasattr(event, "iloc"):
        # pd.Series from a DataFrame row
        event_dict = event.to_dict()
    elif hasattr(event, "__dict__"):
        event_dict = vars(event)
    else:
        event_dict = dict(event)

    entity_id = str(event_dict["entity_id"])
    cohort = str(event_dict["cohort"])

    # --- Get state objects (pre-update) ---
    entity_state = pipeline_state.get_entity_state(entity_id)
    cohort_state = pipeline_state.get_cohort_state(cohort)
    entity_event_count = entity_state.event_count
    cold = is_cold_start(entity_state)

    # --- Extract features ---
    seq_model_cohort = pipeline_state.get_cohort_seq_model(cohort)
    fv = extract_all(
        event_dict,
        entity_state,
        cohort_state,
        pipeline_state.global_state,
        pipeline_state.graph_state,
        seq_model_cohort,
        pipeline_state.seq_model_global,
        pipeline_state.config,
    )

    # --- Score all detectors ---
    detector_scores = registry.score_all(
        fv,
        entity_id=entity_id,
        cohort=cohort,
        entity_state=entity_state,
    )

    # --- Fuse into risk score ---
    risk_score, detector_contributions = registry.fusion.fuse(detector_scores)

    # --- Calibrate (optional) ---
    if calibrator is not None:
        calibrated_prob = calibrator.calibrate(risk_score)
        # Keep raw risk score; calibrated prob is for future use
    else:
        calibrated_prob = risk_score / 100.0

    risk_score = float(max(0.0, min(100.0, risk_score)))

    # --- Classify attack type ---
    if classifier is not None:
        attack_type, attack_confidence, agreement, is_novel = classifier.predict(fv)
    else:
        attack_type = LABEL_NORMAL
        attack_confidence = 1.0 - calibrated_prob
        agreement = True
        is_novel = False

    # --- Attribution ---
    profiler_z = registry.get_profiler_scores(fv, entity_id=entity_id, cohort=cohort)
    isolation_imps = registry.isolation.feature_importances(cohort)

    contributions = attribute_contributions(
        fv=fv,
        detector_contributions=detector_contributions,
        profiler_z_scores=profiler_z,
        isolation_importances=isolation_imps,
        top_k=top_k_contributions,
    )

    # --- Narrative ---
    narrative = build_narrative(contributions, risk_score)

    # --- Counterfactuals ---
    counterfactuals = compute_counterfactuals(
        fv=fv,
        contributions=contributions,
        risk_score=risk_score,
        profiler=registry.profiler,
        fusion=registry.fusion,
        entity_id=entity_id,
        cohort=cohort,
        top_k=counterfactual_top_k,
    )

    # --- Update pipeline state (after scoring, before returning) ---
    pipeline_state.update(event_dict)

    # --- Update drift detectors (if registry has them) ---
    # Drift updates happen in the serving layer; here we just flag cold start

    # --- Build ScoredEvent ---
    from sentinel.schema import AccessEvent as _AE

    if isinstance(event, _AE):
        raw_event = event
    else:
        raw_event = _AE(**{
            k: v for k, v in event_dict.items()
            if k in _AE.model_fields
        })

    ground_truth: GroundTruth | None = None
    if label is not None:
        from sentinel.schema import ATTACK_TYPES as _AT, is_anomalous_label as _ial
        _safe_label = label if label in ("normal", "benign_confounder", "insider_drift", *_AT) else "normal"
        ground_truth = GroundTruth(
            label=_safe_label,
            is_anomaly=_ial(_safe_label),
        )

    return ScoredEvent(
        event_id=str(event_dict["event_id"]),
        entity_id=entity_id,
        entity_type=str(event_dict["entity_type"]),
        cohort=cohort,
        timestamp=event_dict["timestamp"],
        risk_score=risk_score,
        risk_band=risk_band(risk_score),  # type: ignore[arg-type]
        is_alert=risk_score >= alert_threshold,
        predicted_attack_type=attack_type,  # type: ignore[arg-type]
        attack_type_confidence=float(attack_confidence),
        classifier_agreement=agreement,
        is_novel=is_novel,
        detector_scores=detector_scores,
        contributions=contributions,
        narrative=narrative,
        counterfactuals=counterfactuals,
        cold_start=cold,
        entity_event_count=entity_event_count,
        event=raw_event,
        ground_truth=ground_truth,
    )
