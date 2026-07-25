"""Per-entity statistical profiler with cohort shrinkage.

Fits a Welford mean/variance profile per feature per entity, shrunk toward
its cohort prior (the mean of all entities in the same cohort).

Scoring returns:
- A z-score per feature (signed, clipped to ±5)
- A combined Mahalanobis-like statistic (mean of squared z-scores, sqrt)

Cold-start entities with few observations are automatically shrunk heavily
toward the cohort prior, providing reasonable scores with no history.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sentinel.features.extractors import FEATURE_NAMES, FeatureVector

__all__ = ["StatisticalProfiler"]

_CLIP = 5.0  # z-score clip bound


@dataclass
class _WelfordAccum:
    """Online mean/variance accumulator (Welford algorithm)."""

    n: int = 0
    mean: float = 0.0
    M2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    @property
    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        return self.M2 / self.n


@dataclass
class _EntityProfile:
    """Welford accumulators for each feature for one entity."""

    accums: list[_WelfordAccum] = field(
        default_factory=lambda: [_WelfordAccum() for _ in FEATURE_NAMES]
    )


class StatisticalProfiler:
    """Shrunk per-entity statistical profiler.

    Fits mean/variance per feature per entity, shrunk toward cohort prior.
    At score time returns per-feature z-scores and a combined score.

    Usage::

        profiler = StatisticalProfiler(cohort_shrinkage_k=20, min_variance=1e-6)
        profiler.fit(feature_df, split="train")
        scores = profiler.score(fv, entity_id="usr_0001", cohort="finance_analyst")
    """

    def __init__(
        self,
        cohort_shrinkage_k: int = 20,
        min_variance: float = 1e-6,
    ) -> None:
        self.cohort_shrinkage_k = cohort_shrinkage_k
        self.min_variance = min_variance

        # entity_id -> per-feature Welford accumulators
        self._entity_profiles: dict[str, _EntityProfile] = {}

        # cohort_id -> per-feature Welford accumulators (pooled)
        self._cohort_profiles: dict[str, _EntityProfile] = {}

        # entity_id -> event count (for shrinkage weight)
        self._entity_counts: dict[str, int] = {}

        # cohort_id -> list of entity_ids (for building cohort prior)
        self._cohort_entities: dict[str, list[str]] = {}

        self._fitted = False

    # ---------------------------------------------------------------------- #
    # Fitting
    # ---------------------------------------------------------------------- #

    def fit(self, feature_df: pd.DataFrame, split: str = "train") -> None:
        """Fit Welford profiles from a feature DataFrame.

        Args:
            feature_df: DataFrame with FEATURE_NAMES columns plus
                        ``entity_id``, ``cohort``, and ``split`` columns.
            split: only rows where ``feature_df["split"] == split`` are used.
        """
        if "split" in feature_df.columns:
            df = feature_df[feature_df["split"] == split].copy()
        else:
            df = feature_df.copy()

        if df.empty:
            self._fitted = True
            return

        for row in df.itertuples(index=False):
            entity_id = str(row.entity_id)
            cohort = str(row.cohort)

            if entity_id not in self._entity_profiles:
                self._entity_profiles[entity_id] = _EntityProfile()
                self._entity_counts[entity_id] = 0
            if cohort not in self._cohort_profiles:
                self._cohort_profiles[cohort] = _EntityProfile()
                self._cohort_entities.setdefault(cohort, [])
            if entity_id not in self._cohort_entities.get(cohort, []):
                self._cohort_entities.setdefault(cohort, []).append(entity_id)

            ep = self._entity_profiles[entity_id]
            cp = self._cohort_profiles[cohort]

            for i, fname in enumerate(FEATURE_NAMES):
                val = float(getattr(row, fname, 0.0))
                if math.isfinite(val):
                    ep.accums[i].update(val)
                    cp.accums[i].update(val)

            self._entity_counts[entity_id] += 1

        self._fitted = True

    # ---------------------------------------------------------------------- #
    # Scoring
    # ---------------------------------------------------------------------- #

    def score(
        self,
        fv: FeatureVector,
        entity_id: str,
        cohort: str,
    ) -> dict[str, float]:
        """Score a feature vector against the entity's shrunk profile.

        Returns a dict with:
        - one key per feature name: signed z-score clipped to [-5, +5]
        - ``"mahalanobis"`` : sqrt of mean squared z-score (overall anomaly level)
        """
        k = self.cohort_shrinkage_k
        ep = self._entity_profiles.get(entity_id)
        cp = self._cohort_profiles.get(cohort)
        n = self._entity_counts.get(entity_id, 0)

        w = n / (n + k)  # shrinkage weight toward entity

        result: dict[str, float] = {}
        sq_sum = 0.0

        for i, fname in enumerate(FEATURE_NAMES):
            val = float(fv[fname])

            # Shrunk mean
            e_mean = ep.accums[i].mean if ep else 0.0
            c_mean = cp.accums[i].mean if cp else 0.0
            shrunk_mean = w * e_mean + (1.0 - w) * c_mean

            # Shrunk variance (take max to avoid zero)
            e_var = ep.accums[i].variance if ep else 0.0
            c_var = cp.accums[i].variance if cp else 0.0
            shrunk_var = w * e_var + (1.0 - w) * c_var
            shrunk_var = max(shrunk_var, self.min_variance)

            z = (val - shrunk_mean) / math.sqrt(shrunk_var)
            z_clipped = max(-_CLIP, min(_CLIP, z))
            result[fname] = z_clipped
            sq_sum += z_clipped ** 2

        result["mahalanobis"] = math.sqrt(sq_sum / max(len(FEATURE_NAMES), 1))
        return result

    # ---------------------------------------------------------------------- #
    # Persistence helpers
    # ---------------------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        def _prof_to_list(prof: _EntityProfile) -> list[dict[str, Any]]:
            return [{"n": a.n, "mean": a.mean, "M2": a.M2} for a in prof.accums]

        return {
            "cohort_shrinkage_k": self.cohort_shrinkage_k,
            "min_variance": self.min_variance,
            "entity_profiles": {
                k: _prof_to_list(v) for k, v in self._entity_profiles.items()
            },
            "cohort_profiles": {
                k: _prof_to_list(v) for k, v in self._cohort_profiles.items()
            },
            "entity_counts": dict(self._entity_counts),
            "cohort_entities": {k: list(v) for k, v in self._cohort_entities.items()},
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StatisticalProfiler:
        obj = cls(
            cohort_shrinkage_k=d["cohort_shrinkage_k"],
            min_variance=d["min_variance"],
        )

        def _list_to_prof(lst: list[dict[str, Any]]) -> _EntityProfile:
            prof = _EntityProfile.__new__(_EntityProfile)
            prof.accums = [
                _WelfordAccum(n=a["n"], mean=a["mean"], M2=a["M2"]) for a in lst
            ]
            return prof

        obj._entity_profiles = {k: _list_to_prof(v) for k, v in d["entity_profiles"].items()}
        obj._cohort_profiles = {k: _list_to_prof(v) for k, v in d["cohort_profiles"].items()}
        obj._entity_counts = dict(d["entity_counts"])
        obj._cohort_entities = {k: list(v) for k, v in d["cohort_entities"].items()}
        obj._fitted = d["fitted"]
        return obj

    # Expose raw cohort mean/std for a feature (used by counterfactual module)
    def cohort_median_approx(self, cohort: str, feature: str) -> float:
        """Approximate cohort median as the cohort mean (close enough for counterfactuals)."""
        cp = self._cohort_profiles.get(cohort)
        if cp is None:
            return 0.0
        idx = FEATURE_NAMES.index(feature)
        return cp.accums[idx].mean
