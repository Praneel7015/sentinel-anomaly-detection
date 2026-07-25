"""Additive log-odds fusion.

Core design:
    1. Each detector score → rank-based p-value using a fitted streaming
       quantile reservoir over the training distribution.
    2. log-odds = log(p / (1 - p + ε))
    3. Weighted sum of log-odds terms (weights from configs/model.yaml)
    4. sigmoid(bias + weighted_sum) * 100

Explainability is exact: the contribution of detector d to the final logit is
exactly w_d * logit_d. There are no approximations.

Also exposes:
    calibrate(scores, labels)  - fit isotonic calibration
    fit_quantiles(train_scores_df)  - fit rank transform reservoirs
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from sentinel.serving.models import DetectorScores

__all__ = ["LogOddsFusion", "detector_logodds"]

_EPS = 1e-7
_DEFAULT_WEIGHTS: dict[str, float] = {
    "profile": 1.00,
    "isolation": 0.85,
    "sequence": 0.80,
    "graph": 0.70,
    "gru": 0.60,
}
_DEFAULT_BIAS = -3.0
_DEFAULT_LOGIT_CLIP = 6.0


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class _QuantileReservoir:
    """Fixed-size random reservoir for streaming quantile estimation.

    The reservoir holds up to ``max_size`` samples uniformly at random.
    ``rank(x)`` returns the fraction of training samples <= x.
    """

    def __init__(self, max_size: int = 20_000) -> None:
        self.max_size = max_size
        self._data: list[float] = []
        self._sorted: np.ndarray | None = None
        self._dirty = True
        self._n_seen = 0

    def update(self, x: float) -> None:
        self._n_seen += 1
        if len(self._data) < self.max_size:
            self._data.append(x)
        else:
            # Reservoir sampling
            idx = int(np.random.randint(0, self._n_seen))
            if idx < self.max_size:
                self._data[idx] = x
        self._dirty = True

    def rank(self, x: float, min_window: int = 500, warmup_pvalue: float = 0.5) -> float:
        """Return fraction of stored samples <= x (empirical CDF)."""
        if len(self._data) < min_window:
            return warmup_pvalue
        if self._dirty:
            self._sorted = np.sort(np.array(self._data, dtype=np.float64))
            self._dirty = False
        assert self._sorted is not None
        return float(np.searchsorted(self._sorted, x, side="right")) / len(self._sorted)

    def bulk_update(self, values: np.ndarray) -> None:
        """Efficiently load training values into the reservoir."""
        for v in values:
            self.update(float(v))

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_size": self.max_size,
            "data": list(self._data),
            "n_seen": self._n_seen,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> _QuantileReservoir:
        obj = cls(max_size=d["max_size"])
        obj._data = list(d["data"])
        obj._n_seen = d["n_seen"]
        obj._dirty = True
        return obj


class LogOddsFusion:
    """Additive log-odds fusion of detector scores into a 0-100 risk score.

    Usage::

        fusion = LogOddsFusion(weights=..., bias=-3.0)
        fusion.fit_quantiles(train_scores_df)
        risk, contributions = fusion.fuse(detector_scores)
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        bias: float = _DEFAULT_BIAS,
        logit_clip: float = _DEFAULT_LOGIT_CLIP,
        quantile_window: int = 20_000,
        min_window_for_pvalue: int = 500,
        warmup_pvalue: float = 0.5,
        redistribute_missing_weight: bool = True,
    ) -> None:
        self.weights = dict(weights or _DEFAULT_WEIGHTS)
        self.bias = bias
        self.logit_clip = logit_clip
        self.quantile_window = quantile_window
        self.min_window_for_pvalue = min_window_for_pvalue
        self.warmup_pvalue = warmup_pvalue
        self.redistribute_missing_weight = redistribute_missing_weight

        # detector_name -> quantile reservoir
        self._reservoirs: dict[str, _QuantileReservoir] = {}
        for name in self.weights:
            self._reservoirs[name] = _QuantileReservoir(max_size=quantile_window)

        self._fitted = False

    # ---------------------------------------------------------------------- #
    # Fitting
    # ---------------------------------------------------------------------- #

    def fit_quantiles(self, train_scores_df: pd.DataFrame) -> None:
        """Fit quantile reservoirs from training detector scores.

        Args:
            train_scores_df: DataFrame with columns matching detector names
                             (profile, isolation, sequence, graph, gru).
                             Only non-null values are used.
        """
        for name in self.weights:
            if name in train_scores_df.columns:
                vals = train_scores_df[name].dropna().values.astype(np.float64)
                vals = vals[np.isfinite(vals)]
                self._reservoirs[name].bulk_update(vals)
        self._fitted = True

    # ---------------------------------------------------------------------- #
    # Fusing
    # ---------------------------------------------------------------------- #

    def fuse(
        self,
        detector_scores: DetectorScores,
    ) -> tuple[float, dict[str, float]]:
        """Compute risk score and per-detector log-odds contributions.

        Args:
            detector_scores: DetectorScores pydantic object.

        Returns:
            (risk_score, contributions) where:
            - risk_score is in [0, 100]
            - contributions is {detector_name: log_odds_contribution}
              that sum to the total logit (minus the bias)
        """
        scores_dict: dict[str, float | None] = {
            "profile": detector_scores.profile,
            "isolation": detector_scores.isolation,
            "sequence": detector_scores.sequence,
            "graph": detector_scores.graph,
            "gru": detector_scores.gru,
        }

        # Determine active detectors and weights
        active_weights: dict[str, float] = {}
        for name, w in self.weights.items():
            s = scores_dict.get(name)
            if s is not None:
                active_weights[name] = w

        total_w = sum(active_weights.values())
        if total_w < 1e-12:
            # No active detectors: return neutral score
            return 50.0, {}

        # If redistribute_missing_weight, scale up active weights proportionally
        if self.redistribute_missing_weight:
            full_w = sum(self.weights.values())
            scale = full_w / total_w if total_w > 0 else 1.0
        else:
            scale = 1.0

        contributions: dict[str, float] = {}
        logit = self.bias

        for name, w in active_weights.items():
            raw_score = scores_dict[name]
            assert raw_score is not None

            # Convert raw score (0-1) to p-value via quantile rank
            reservoir = self._reservoirs.get(name)
            if reservoir is not None:
                p = reservoir.rank(
                    raw_score,
                    min_window=self.min_window_for_pvalue,
                    warmup_pvalue=self.warmup_pvalue,
                )
            else:
                p = raw_score  # fallback: treat score as p-value

            # Clip p away from 0 and 1 to avoid ±inf log-odds
            p = max(_EPS, min(1.0 - _EPS, p))

            # Log-odds of this detector
            lo = math.log(p / (1.0 - p))

            # Clip individual logit terms to prevent saturation
            lo = max(-self.logit_clip, min(self.logit_clip, lo))

            effective_w = w * scale
            contrib = effective_w * lo
            contributions[name] = contrib
            logit += contrib

        risk = _sigmoid(logit) * 100.0
        return float(risk), contributions

    # ---------------------------------------------------------------------- #
    # Online update
    # ---------------------------------------------------------------------- #

    def update_quantile(self, name: str, score: float) -> None:
        """Update the reservoir for a single detector with a new score observation."""
        if name in self._reservoirs:
            self._reservoirs[name].update(score)

    # ---------------------------------------------------------------------- #
    # Persistence
    # ---------------------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "bias": self.bias,
            "logit_clip": self.logit_clip,
            "quantile_window": self.quantile_window,
            "min_window_for_pvalue": self.min_window_for_pvalue,
            "warmup_pvalue": self.warmup_pvalue,
            "redistribute_missing_weight": self.redistribute_missing_weight,
            "reservoirs": {k: v.to_dict() for k, v in self._reservoirs.items()},
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LogOddsFusion:
        obj = cls(
            weights=d["weights"],
            bias=d["bias"],
            logit_clip=d["logit_clip"],
            quantile_window=d["quantile_window"],
            min_window_for_pvalue=d["min_window_for_pvalue"],
            warmup_pvalue=d["warmup_pvalue"],
            redistribute_missing_weight=d["redistribute_missing_weight"],
        )
        obj._reservoirs = {
            k: _QuantileReservoir.from_dict(v) for k, v in d["reservoirs"].items()
        }
        obj._fitted = d["fitted"]
        return obj


def detector_logodds(
    score: float,
    reservoir: _QuantileReservoir,
    min_window: int = 500,
    warmup_pvalue: float = 0.5,
    logit_clip: float = _DEFAULT_LOGIT_CLIP,
) -> float:
    """Convert a single detector score to its log-odds term."""
    p = reservoir.rank(score, min_window=min_window, warmup_pvalue=warmup_pvalue)
    p = max(_EPS, min(1.0 - _EPS, p))
    lo = math.log(p / (1.0 - p))
    return max(-logit_clip, min(logit_clip, lo))
