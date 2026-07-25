"""Markov/n-gram sequence detector.

Wraps the surprisal computed by features/sequence.py into a per-cohort
distribution and returns a rank-normalised anomaly score in [0, 1].

During fit(), surprisal values for all training events are collected per
cohort. At score time, the surprisal for the new event is rank-transformed
against this empirical distribution: percentile rank in [0, 1].
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sentinel.features.extractors import FeatureVector

__all__ = ["MarkovDetector"]


class MarkovDetector:
    """Per-cohort empirical surprisal distribution.

    Fits from the ``command_surprisal`` feature in the feature DataFrame
    (pre-computed during batch_features). At score time, looks up where the
    new surprisal sits in the cohort distribution.

    Usage::

        det = MarkovDetector()
        det.fit(feature_df)
        score = det.score(fv)  # uses command_surprisal feature
    """

    def __init__(self) -> None:
        # cohort_id -> sorted array of training surprisal values
        self._distributions: dict[str, np.ndarray] = {}
        # global fallback distribution
        self._global_dist: np.ndarray | None = None
        self._fitted = False

    # ---------------------------------------------------------------------- #
    # Fitting
    # ---------------------------------------------------------------------- #

    def fit(self, feature_df: pd.DataFrame, split: str = "train") -> None:
        """Fit surprisal distributions from training data.

        Args:
            feature_df: DataFrame with ``command_surprisal``, ``cohort``,
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

        # Drop non-finite values
        valid_mask = np.isfinite(df["command_surprisal"].values)
        df_valid = df[valid_mask]

        all_vals: list[float] = []

        if "cohort" in df_valid.columns:
            for cohort, group in df_valid.groupby("cohort"):
                vals = group["command_surprisal"].values.astype(np.float64)
                self._distributions[str(cohort)] = np.sort(vals)
                all_vals.extend(vals.tolist())
        else:
            all_vals = df_valid["command_surprisal"].values.tolist()

        if all_vals:
            self._global_dist = np.sort(np.array(all_vals, dtype=np.float64))

        self._fitted = True

    # ---------------------------------------------------------------------- #
    # Scoring
    # ---------------------------------------------------------------------- #

    def score(self, fv: FeatureVector, cohort: str | None = None) -> float:
        """Return rank-normalised anomaly score in [0, 1].

        Higher = more anomalous (higher surprisal than most training events).
        Returns 0.5 (neutral) if no training data is available.
        """
        surprisal = float(fv["command_surprisal"])

        # Choose distribution: cohort-specific, or global fallback
        dist = None
        if cohort is not None:
            dist = self._distributions.get(cohort)
        if dist is None:
            dist = self._global_dist
        if dist is None or len(dist) == 0:
            return 0.5

        # Rank transform: fraction of training values <= surprisal
        rank = float(np.searchsorted(dist, surprisal, side="right")) / len(dist)
        return float(max(0.0, min(1.0, rank)))

    # ---------------------------------------------------------------------- #
    # Persistence
    # ---------------------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return {
            "distributions": {
                k: v.tolist() for k, v in self._distributions.items()
            },
            "global_dist": self._global_dist.tolist() if self._global_dist is not None else None,
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MarkovDetector:
        obj = cls()
        obj._distributions = {
            k: np.array(v, dtype=np.float64)
            for k, v in d["distributions"].items()
        }
        obj._global_dist = (
            np.array(d["global_dist"], dtype=np.float64)
            if d["global_dist"] is not None
            else None
        )
        obj._fitted = d["fitted"]
        return obj
