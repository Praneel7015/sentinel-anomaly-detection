"""Feature pipeline - the two entry points, one code path.

stream_features(events_iterable)
    Yields (event_dict, FeatureVector) one at a time, updating state after
    emitting.  The canonical real-time path.

batch_features(df)
    Returns a feature DataFrame aligned to the input, implemented by driving
    the exact same streaming loop (sorted by timestamp).  There is deliberately
    no separate vectorised implementation; leakage cannot sneak in through a
    different code path.

Leakage guarantee
-----------------
State is updated AFTER the feature vector for each event is emitted.
Features for event N are computed from events 0..N-1 only.
A unit test asserts batch == stream bit-for-bit.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import pandas as pd

from sentinel.features.cohort import CohortState
from sentinel.features.extractors import (
    FEATURE_NAMES,
    FeatureVector,
    extract_all,
)
from sentinel.features.graph import GraphState
from sentinel.features.sequence import SequenceModel
from sentinel.features.state import EntityState, GlobalState

__all__ = ["stream_features", "batch_features", "PipelineState"]


class PipelineState:
    """All mutable state for one pipeline run.

    Holds entity states, cohort states, global state, and sequence models.
    Serialisable via ``to_dict`` / ``from_dict`` for snapshot/restore.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = config or {}
        self.entity_states: dict[str, EntityState] = {}
        self.cohort_states: dict[str, CohortState] = {}
        self.global_state: GlobalState = GlobalState()
        self.graph_state: GraphState = GraphState()
        self.seq_model_global: SequenceModel = SequenceModel(
            smoothing=self.config.get("sequence_smoothing", 0.5)
        )
        # per-cohort sequence models
        self.seq_models_cohort: dict[str, SequenceModel] = {}

    def get_entity_state(self, entity_id: str) -> EntityState:
        if entity_id not in self.entity_states:
            hl = self.config.get("ewma_half_life_days", 14.0)
            self.entity_states[entity_id] = EntityState(entity_id, half_life_days=hl)
        return self.entity_states[entity_id]

    def get_cohort_state(self, cohort_id: str) -> CohortState:
        if cohort_id not in self.cohort_states:
            self.cohort_states[cohort_id] = CohortState(cohort_id)
        return self.cohort_states[cohort_id]

    def get_cohort_seq_model(self, cohort_id: str) -> SequenceModel:
        if cohort_id not in self.seq_models_cohort:
            self.seq_models_cohort[cohort_id] = SequenceModel(
                smoothing=self.config.get("sequence_smoothing", 0.5)
            )
        return self.seq_models_cohort[cohort_id]

    def update(self, event: dict[str, Any], *, learn: bool = True) -> None:
        """Update all state objects with one event (called AFTER feature extraction)."""
        entity_id = str(event["entity_id"])
        cohort_id = str(event["cohort"])
        resource = str(event["resource_accessed"])

        es = self.get_entity_state(entity_id)
        cs = self.get_cohort_state(cohort_id)
        sm_c = self.get_cohort_seq_model(cohort_id)

        # graph update: pass resource set BEFORE adding current resource
        self.graph_state.update(entity_id, cohort_id, resource, set(es.resource_set))

        # entity update
        es.update(event, learn=learn)

        # cohort and global updates (always learn - poisoning guard is per-entity)
        cs.update(event)
        self.global_state.update(event)

        # sequence model updates
        cmds = list(event.get("command_sequence") or [])
        sm_c.update(cmds)
        self.seq_model_global.update(cmds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "entity_states": {k: v.to_dict() for k, v in self.entity_states.items()},
            "cohort_states": {k: v.to_dict() for k, v in self.cohort_states.items()},
            "global_state": self.global_state.to_dict(),
            "graph_state": self.graph_state.to_dict(),
            "seq_model_global": self.seq_model_global.to_dict(),
            "seq_models_cohort": {k: v.to_dict() for k, v in self.seq_models_cohort.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineState:
        obj = cls.__new__(cls)
        obj.config = d["config"]
        obj.entity_states = {k: EntityState.from_dict(v) for k, v in d["entity_states"].items()}
        obj.cohort_states = {k: CohortState.from_dict(v) for k, v in d["cohort_states"].items()}
        obj.global_state = GlobalState.from_dict(d["global_state"])
        obj.graph_state = GraphState.from_dict(d["graph_state"])
        obj.seq_model_global = SequenceModel.from_dict(d["seq_model_global"])
        obj.seq_models_cohort = {
            k: SequenceModel.from_dict(v) for k, v in d["seq_models_cohort"].items()
        }
        return obj


def stream_features(
    events_iterable: Iterable[dict[str, Any]],
    state: PipelineState | None = None,
    config: dict[str, Any] | None = None,
) -> Iterator[tuple[dict[str, Any], FeatureVector]]:
    """Yield (event, FeatureVector) for each event in ``events_iterable``.

    State is updated AFTER each yield so features are strictly past-only.
    Events must arrive in non-decreasing timestamp order.

    Args:
        events_iterable: any iterable of event dicts with AccessEvent fields.
        state: a ``PipelineState`` to resume from (e.g. for val/test scoring
               after fitting on train).  Created fresh if None.
        config: extra config overrides (merged with state.config).
    """
    if state is None:
        state = PipelineState(config or {})
    elif config:
        state.config = {**state.config, **config}

    for event in events_iterable:
        entity_id = str(event["entity_id"])
        cohort_id = str(event["cohort"])

        es = state.get_entity_state(entity_id)
        cs = state.get_cohort_state(cohort_id)
        sm_c = state.get_cohort_seq_model(cohort_id)

        fv = extract_all(
            event,
            es,
            cs,
            state.global_state,
            state.graph_state,
            sm_c,
            state.seq_model_global,
            state.config,
        )

        yield event, fv

        # Update state AFTER emitting (leakage-free guarantee)
        state.update(event)


def batch_features(
    df: pd.DataFrame,
    state: PipelineState | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Return a feature DataFrame aligned to ``df``.

    Implemented by driving the exact same streaming loop (sorted by timestamp).
    No separate vectorised implementation exists so leakage has nowhere to hide.

    The input DataFrame must contain at minimum the columns of AccessEvent.
    The output DataFrame has the same index as the input and columns =
    FEATURE_NAMES.

    Args:
        df: event DataFrame, may be unsorted.
        state: pipeline state to resume from.
        config: config overrides.

    Returns:
        pd.DataFrame with columns ``FEATURE_NAMES`` and same index as ``df``.
    """
    # Sort by timestamp for streaming correctness, remember original order
    sorted_df = df.sort_values("timestamp", kind="stable")

    # Convert to list of dicts once (much faster than iterrows per-row)
    sorted_records: list[dict[str, Any]] = sorted_df.to_dict(orient="records")
    sorted_indices = sorted_df.index.tolist()

    feature_matrix: list[list[float]] = []
    for _event, fv in stream_features(iter(sorted_records), state=state, config=config):
        feature_matrix.append(fv.to_numpy().tolist())

    result = pd.DataFrame(
        feature_matrix,
        index=sorted_indices,
        columns=FEATURE_NAMES,
    )
    # Restore original order
    return result.loc[df.index]
