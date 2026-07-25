"""Incremental, past-only feature extraction shared by batch and streaming paths.

Public API
----------
- :class:`~sentinel.features.state.EntityState`
- :class:`~sentinel.features.state.GlobalState`
- :class:`~sentinel.features.cohort.CohortState`
- :func:`~sentinel.features.cohort.shrink`
- :func:`~sentinel.features.cohort.is_cold_start`
- :class:`~sentinel.features.graph.GraphState`
- :class:`~sentinel.features.sequence.SequenceModel`
- :data:`~sentinel.features.extractors.FEATURE_NAMES`
- :class:`~sentinel.features.extractors.FeatureVector`
- :func:`~sentinel.features.extractors.extract_all`
- :func:`~sentinel.features.pipeline.stream_features`
- :func:`~sentinel.features.pipeline.batch_features`
- :class:`~sentinel.features.pipeline.PipelineState`
"""

from sentinel.features.cohort import CohortState, is_cold_start, shrink
from sentinel.features.extractors import FEATURE_NAMES, REGISTRY, FeatureVector, extract_all
from sentinel.features.graph import GraphState
from sentinel.features.pipeline import PipelineState, batch_features, stream_features
from sentinel.features.sequence import GRU_FEATURE_DIM, GRU_WINDOW, SequenceModel
from sentinel.features.state import EntityState, GlobalState, SourceIPState

__all__ = [
    "EntityState",
    "GlobalState",
    "SourceIPState",
    "CohortState",
    "shrink",
    "is_cold_start",
    "GraphState",
    "SequenceModel",
    "GRU_WINDOW",
    "GRU_FEATURE_DIM",
    "FEATURE_NAMES",
    "FeatureVector",
    "extract_all",
    "REGISTRY",
    "stream_features",
    "batch_features",
    "PipelineState",
]
