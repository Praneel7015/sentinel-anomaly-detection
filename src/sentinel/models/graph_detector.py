"""Graph anomaly detector.

Uses the four graph features from FeatureVector:
  - graph_entity_novel_resource  (novel edge: 1 or 0)
  - graph_peer_resource_count    (how many peers touched this resource)
  - graph_jaccard_vs_cohort      (resource-set similarity to cohort)
  - graph_entity_degree_deviation (z-score of distinct-resource count)

Computes a cohort-normalised anomaly score in [0, 1].

The score is a weighted combination of:
1. Novel-resource flag (novel = anomalous)
2. Low peer resource count (fewer peers = more anomalous)
3. Low Jaccard similarity (diverging from cohort = anomalous)
4. High absolute degree deviation (outlier in resource breadth = anomalous)

These are normalised against cohort distributions fitted during training.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sentinel.features.extractors import FeatureVector

__all__ = ["GraphDetector"]

_GRAPH_FEATURES = [
    "graph_entity_novel_resource",
    "graph_peer_resource_count",
    "graph_jaccard_vs_cohort",
    "graph_entity_degree_deviation",
]

# Weights for each sub-score (must sum to 1)
_WEIGHTS = np.array([0.35, 0.25, 0.20, 0.20], dtype=np.float64)


class GraphDetector:
    """Cohort-normalised graph anomaly detector.

    Usage::

        det = GraphDetector()
        det.fit(feature_df)
        score = det.score(fv, cohort="finance_analyst")
    """

    def __init__(self) -> None:
        # cohort_id -> per-feature (mean, std) for normalisation
        self._cohort_stats: dict[str, dict[str, tuple[float, float]]] = {}
        # global stats for fallback
        self._global_stats: dict[str, tuple[float, float]] = {}
        self._fitted = False

    # ---------------------------------------------------------------------- #
    # Fitting
    # ---------------------------------------------------------------------- #

    def fit(self, feature_df: pd.DataFrame, split: str = "train") -> None:
        """Fit cohort-level distributions for graph features.

        Args:
            feature_df: DataFrame with graph feature columns, ``cohort``,
                        and ``split`` columns.
            split: filter rows by this split value.
        """
        if "split" in feature_df.columns:
            df = feature_df[feature_df["split"] == split].copy()
        else:
            df = feature_df.copy()

        if df.empty:
            self._fitted = True
            return

        # Global stats
        for feat in _GRAPH_FEATURES:
            if feat in df.columns:
                vals = df[feat].values.astype(np.float64)
                vals = vals[np.isfinite(vals)]
                self._global_stats[feat] = (
                    float(vals.mean()) if len(vals) > 0 else 0.0,
                    float(vals.std()) if len(vals) > 1 else 1.0,
                )

        # Per-cohort stats
        if "cohort" in df.columns:
            for cohort, group in df.groupby("cohort"):
                cohort_id = str(cohort)
                self._cohort_stats[cohort_id] = {}
                for feat in _GRAPH_FEATURES:
                    if feat in group.columns:
                        vals = group[feat].values.astype(np.float64)
                        vals = vals[np.isfinite(vals)]
                        if len(vals) == 0:
                            mean, std = self._global_stats.get(feat, (0.0, 1.0))
                        else:
                            mean = float(vals.mean())
                            std = float(vals.std()) if len(vals) > 1 else 1.0
                        self._cohort_stats[cohort_id][feat] = (mean, max(std, 1e-6))

        self._fitted = True

    # ---------------------------------------------------------------------- #
    # Scoring
    # ---------------------------------------------------------------------- #

    def score(self, fv: FeatureVector, cohort: str | None = None) -> float:
        """Return cohort-normalised anomaly score in [0, 1].

        Combines four graph features into a single score.
        """
        stats = {}
        if cohort and cohort in self._cohort_stats:
            stats = self._cohort_stats[cohort]

        sub_scores = np.zeros(4, dtype=np.float64)

        # 1. Novel resource flag: 1 = novel = anomalous
        novel = float(fv["graph_entity_novel_resource"])
        sub_scores[0] = max(0.0, min(1.0, novel))

        # 2. Peer resource count: low count = anomalous
        #    Normalise: score = 1 - sigmoid(count / mean_count)
        peer_count = float(fv["graph_peer_resource_count"])
        feat = "graph_peer_resource_count"
        mean_count, std_count = stats.get(feat, self._global_stats.get(feat, (10.0, 5.0)))
        if mean_count > 0:
            ratio = peer_count / max(mean_count, 1.0)
            # Low ratio -> high anomaly: use 1 - clipped_ratio
            sub_scores[1] = max(0.0, min(1.0, 1.0 - min(ratio, 2.0) / 2.0))
        else:
            sub_scores[1] = 0.5

        # 3. Jaccard vs cohort: low similarity = anomalous
        jaccard = float(fv["graph_jaccard_vs_cohort"])
        feat = "graph_jaccard_vs_cohort"
        mean_j, std_j = stats.get(feat, self._global_stats.get(feat, (0.5, 0.2)))
        # Convert to z-score then to probability
        if std_j > 1e-9:
            z = (mean_j - jaccard) / std_j  # negative jaccard deviation = anomaly
            sub_scores[2] = max(0.0, min(1.0, (z + 3.0) / 6.0))
        else:
            sub_scores[2] = 0.5

        # 4. Degree deviation: high absolute z-score = anomalous
        deg_dev = float(fv["graph_entity_degree_deviation"])
        # Already a z-score; convert to [0,1]: clip to [-3,3] then rescale
        sub_scores[3] = max(0.0, min(1.0, (abs(deg_dev) + 3.0) / 6.0 - 0.5))

        return float(np.dot(_WEIGHTS, sub_scores))

    # ---------------------------------------------------------------------- #
    # Persistence
    # ---------------------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return {
            "cohort_stats": {
                cid: {f: list(t) for f, t in stats.items()}
                for cid, stats in self._cohort_stats.items()
            },
            "global_stats": {f: list(t) for f, t in self._global_stats.items()},
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GraphDetector:
        obj = cls()
        obj._cohort_stats = {
            cid: {f: tuple(t) for f, t in stats.items()}  # type: ignore[misc]
            for cid, stats in d["cohort_stats"].items()
        }
        obj._global_stats = {
            f: tuple(t) for f, t in d["global_stats"].items()  # type: ignore[misc]
        }
        obj._fitted = d["fitted"]
        return obj
