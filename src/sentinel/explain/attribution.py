"""Exact contribution decomposition.

Given detector_contributions: dict[str, float] from fusion and per-feature
z-scores from the profiler, decompose the risk score into per-feature
contributions.

Strategy:
- profiler → features by z-score magnitude
- isolation → top features by expected feature importance from the forest
- sequence → sequence-related features
- graph → graph-specific features
- gru → distributes remaining contribution uniformly across all features

Output: list of Contribution objects from serving/models.py.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentinel.models.isolation import IsolationForestDetector

from sentinel.features.extractors import FEATURE_NAMES, REGISTRY, FeatureVector
from sentinel.serving.models import Contribution

__all__ = ["attribute_contributions"]

# Feature groups by detector
_SEQUENCE_FEATURES = frozenset([
    "command_surprisal",
    "unseen_bigram_count",
    "sequence_length_zscore",
])

_GRAPH_FEATURES = frozenset([
    "graph_entity_novel_resource",
    "graph_peer_resource_count",
    "graph_jaccard_vs_cohort",
    "graph_entity_degree_deviation",
])

_PROFILER_FEATURES = frozenset(
    fname for fname in FEATURE_NAMES
    if fname not in _SEQUENCE_FEATURES and fname not in _GRAPH_FEATURES
)


def attribute_contributions(
    fv: FeatureVector,
    detector_contributions: dict[str, float],
    profiler_z_scores: dict[str, float],
    isolation_importances: dict[str, float] | None = None,
    top_k: int = 6,
) -> list[Contribution]:
    """Decompose detector log-odds contributions into per-feature contributions.

    Args:
        fv: the feature vector for the current event.
        detector_contributions: {detector_name: log_odds_contribution} from fusion.
        profiler_z_scores: {feature_name: z_score} from StatisticalProfiler.score().
        isolation_importances: {feature_name: importance} from IsolationForestDetector.
        top_k: number of top contributions to return.

    Returns:
        Sorted list of Contribution objects, highest |contribution| first.
    """
    feature_contributions: dict[str, float] = {f: 0.0 for f in FEATURE_NAMES}

    # --- Profile detector contribution ---
    profile_logodds = detector_contributions.get("profile", 0.0)
    if profile_logodds != 0.0:
        # Distribute proportionally by |z-score|
        z_abs = {
            f: abs(profiler_z_scores.get(f, 0.0))
            for f in _PROFILER_FEATURES
        }
        z_total = sum(z_abs.values())
        if z_total > 1e-12:
            for f in _PROFILER_FEATURES:
                feature_contributions[f] += profile_logodds * (z_abs[f] / z_total)

    # --- Isolation detector contribution ---
    isolation_logodds = detector_contributions.get("isolation", 0.0)
    if isolation_logodds != 0.0:
        imps = isolation_importances or {}
        imp_total = sum(imps.values())
        if imp_total > 1e-12:
            for f in FEATURE_NAMES:
                feature_contributions[f] += isolation_logodds * (imps.get(f, 0.0) / imp_total)
        else:
            # Uniform over all features
            n = len(FEATURE_NAMES)
            for f in FEATURE_NAMES:
                feature_contributions[f] += isolation_logodds / n

    # --- Sequence detector contribution ---
    sequence_logodds = detector_contributions.get("sequence", 0.0)
    if sequence_logodds != 0.0:
        seq_feats = list(_SEQUENCE_FEATURES)
        per_feat = sequence_logodds / max(len(seq_feats), 1)
        for f in seq_feats:
            feature_contributions[f] += per_feat

    # --- Graph detector contribution ---
    graph_logodds = detector_contributions.get("graph", 0.0)
    if graph_logodds != 0.0:
        graph_feats = list(_GRAPH_FEATURES)
        per_feat = graph_logodds / max(len(graph_feats), 1)
        for f in graph_feats:
            feature_contributions[f] += per_feat

    # --- GRU contribution ---
    gru_logodds = detector_contributions.get("gru", 0.0)
    if gru_logodds != 0.0:
        n = len(FEATURE_NAMES)
        for f in FEATURE_NAMES:
            feature_contributions[f] += gru_logodds / n

    # --- Build Contribution objects ---
    contributions: list[Contribution] = []
    for fname, logodds in feature_contributions.items():
        if abs(logodds) < 1e-12:
            continue

        val = float(fv[fname])
        meta = REGISTRY.get(fname)

        if meta is not None:
            display_name = meta.meta.display_name
            description = meta.meta.description
            direction_hint = meta.meta.direction
        else:
            display_name = fname
            description = fname
            direction_hint = "increases_risk"

        # Direction: positive logodds increases risk, negative decreases
        direction: str = "increases" if logodds > 0 else "decreases"

        display_value = _format_value(fname, val)

        contributions.append(
            Contribution(
                feature=fname,
                display_name=display_name,
                value=val,
                display_value=display_value,
                contribution=logodds,
                direction=direction,  # type: ignore[arg-type]
                description=description,
            )
        )

    # Sort by absolute contribution, descending
    contributions.sort(key=lambda c: abs(c.contribution), reverse=True)
    return contributions[:top_k]


def _format_value(fname: str, val: float) -> str:
    """Format a feature value for display."""
    if fname == "geo_velocity_kmh":
        return f"{val:,.0f} km/h"
    if fname in ("entity_failure_count_5m", "entity_failure_count_1h",
                 "entity_failure_count_24h", "ip_failure_count_5m",
                 "ip_failure_count_1h", "ip_failure_count_24h"):
        return f"{int(val)} failures"
    if fname in ("offhours_bytes_rolling_7d", "bytes_zscore"):
        return f"{val:.1f}"
    if fname in ("is_offhours", "new_country_flag", "new_subnet_flag",
                 "entity_novel_resource", "cohort_novel_resource",
                 "fingerprint_unknown", "os_mismatch", "mac_oui_change",
                 "success_after_failures", "protocol_novelty",
                 "auth_method_novelty", "graph_entity_novel_resource",
                 "is_cold_start_flag"):
        return "yes" if val > 0.5 else "no"
    if "zscore" in fname or "z_score" in fname:
        return f"{val:+.2f}σ"
    if fname in ("geo_centroid_distance_km",):
        return f"{val:,.0f} km"
    if fname in ("transfer_to_duration_ratio",):
        return f"{val:,.0f} B/s"
    return f"{val:.3g}"
