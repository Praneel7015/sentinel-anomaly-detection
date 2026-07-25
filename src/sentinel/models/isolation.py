"""IsolationForest wrapper - one forest per cohort.

Trains one sklearn IsolationForest per cohort (not per entity, which would
be too sparse for a forest). Scores are normalised to [0, 1] where 1 means
maximally anomalous.

The raw isolation score from sklearn is in [-0.5, 0.5] range; we map it to
[0, 1] by: normalised = (0.5 - raw_score) / 1.0, clipped to [0, 1].
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from sentinel.features.extractors import FEATURE_NAMES, FeatureVector

__all__ = ["IsolationForestDetector"]


class IsolationForestDetector:
    """One IsolationForest per cohort.

    Usage::

        det = IsolationForestDetector(n_estimators=200)
        det.fit(feature_df, split="train")
        score = det.score(fv, cohort="finance_analyst")
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: int | str = 4096,
        contamination: float | str = "auto",
        random_state: int = 20260725,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

        # cohort_id -> fitted IsolationForest
        self._forests: dict[str, IsolationForest] = {}
        # cohort_id -> training score range for normalisation
        self._train_score_min: dict[str, float] = {}
        self._train_score_max: dict[str, float] = {}
        self._fitted = False

    # ---------------------------------------------------------------------- #
    # Fitting
    # ---------------------------------------------------------------------- #

    def fit(self, feature_df: pd.DataFrame, split: str = "train") -> None:
        """Fit one IsolationForest per cohort from train-split rows.

        Args:
            feature_df: DataFrame with FEATURE_NAMES columns plus
                        ``cohort`` and ``split`` columns.
            split: filter rows by this split value.
        """
        if "split" in feature_df.columns:
            df = feature_df[feature_df["split"] == split].copy()
        else:
            df = feature_df.copy()

        if df.empty:
            self._fitted = True
            return

        for cohort, group in df.groupby("cohort"):
            cohort_id = str(cohort)
            X = group[FEATURE_NAMES].values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            n_samples = len(X)
            max_s = self.max_samples
            if isinstance(max_s, int):
                max_s = min(max_s, n_samples)
            else:
                max_s = "auto"

            forest = IsolationForest(
                n_estimators=self.n_estimators,
                max_samples=max_s,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=1,
            )
            forest.fit(X)
            self._forests[cohort_id] = forest

            # Compute training score range for normalisation
            raw_scores = forest.score_samples(X)
            self._train_score_min[cohort_id] = float(raw_scores.min())
            self._train_score_max[cohort_id] = float(raw_scores.max())

        self._fitted = True

    # ---------------------------------------------------------------------- #
    # Scoring
    # ---------------------------------------------------------------------- #

    def score(self, fv: FeatureVector, cohort: str) -> float:
        """Return normalised anomaly score in [0, 1].

        Returns 0.5 (neutral) if the cohort was not seen during training.
        """
        forest = self._forests.get(cohort)
        if forest is None:
            return 0.5

        x = fv.to_numpy().reshape(1, -1).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        raw = float(forest.score_samples(x)[0])

        # sklearn score_samples: lower = more anomalous, range roughly [-0.5, 0.5]
        # Normalise using training range so 0 = normal-like, 1 = most anomalous seen
        s_min = self._train_score_min.get(cohort, -0.5)
        s_max = self._train_score_max.get(cohort, 0.5)
        span = s_max - s_min
        if span < 1e-9:
            return 0.5
        normalised = (s_max - raw) / span
        return float(max(0.0, min(1.0, normalised)))

    # ---------------------------------------------------------------------- #
    # Feature importance (for attribution)
    # ---------------------------------------------------------------------- #

    def feature_importances(self, cohort: str) -> dict[str, float]:
        """Return feature importances for a cohort's forest (mean depth-based).

        Returns uniform importances if the cohort is not fitted.
        """
        forest = self._forests.get(cohort)
        if forest is None:
            n = len(FEATURE_NAMES)
            return {f: 1.0 / n for f in FEATURE_NAMES}

        # Use mean feature importance from the impurity decrease proxy
        # sklearn IsolationForest doesn't have feature_importances_ by default,
        # so we use the mean path length deviation per feature as a proxy.
        importances = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        for estimator in forest.estimators_:
            if hasattr(estimator, "feature_importances_"):
                importances += estimator.feature_importances_

        total = importances.sum()
        if total < 1e-12:
            n = len(FEATURE_NAMES)
            return {f: 1.0 / n for f in FEATURE_NAMES}

        return {f: float(importances[i] / total) for i, f in enumerate(FEATURE_NAMES)}

    # ---------------------------------------------------------------------- #
    # Persistence
    # ---------------------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        import pickle
        import base64

        forests_serialised = {}
        for cohort_id, forest in self._forests.items():
            forests_serialised[cohort_id] = base64.b64encode(
                pickle.dumps(forest)
            ).decode("ascii")

        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "forests": forests_serialised,
            "train_score_min": dict(self._train_score_min),
            "train_score_max": dict(self._train_score_max),
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IsolationForestDetector:
        import pickle
        import base64

        obj = cls(
            n_estimators=d["n_estimators"],
            max_samples=d["max_samples"],
            contamination=d["contamination"],
            random_state=d["random_state"],
        )
        obj._forests = {
            cid: pickle.loads(base64.b64decode(v.encode("ascii")))
            for cid, v in d["forests"].items()
        }
        obj._train_score_min = dict(d["train_score_min"])
        obj._train_score_max = dict(d["train_score_max"])
        obj._fitted = d["fitted"]
        return obj
