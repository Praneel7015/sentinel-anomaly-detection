"""Isotonic calibration wrapper.

Wraps sklearn.isotonic.IsotonicRegression to map raw 0-100 risk scores to
calibrated probabilities in [0, 1].
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression

__all__ = ["IsotonicCalibrator"]


class IsotonicCalibrator:
    """Maps raw risk scores to calibrated probabilities.

    Usage::

        cal = IsotonicCalibrator()
        cal.fit(raw_scores, binary_labels)  # binary_labels: 0/1
        calibrated = cal.calibrate(score)   # float in [0, 1]
    """

    def __init__(self) -> None:
        self._iso: IsotonicRegression | None = None
        self._fitted = False
        # Fallback scale when not fitted: identity / 100
        self._raw_min = 0.0
        self._raw_max = 100.0

    def fit(self, raw_scores: np.ndarray, binary_labels: np.ndarray) -> None:
        """Fit isotonic regression.

        Args:
            raw_scores: 1-D array of raw risk scores (0-100).
            binary_labels: 1-D array of 0/1 ground truth labels.
        """
        raw_scores = np.asarray(raw_scores, dtype=np.float64).ravel()
        binary_labels = np.asarray(binary_labels, dtype=np.float64).ravel()

        if len(raw_scores) == 0:
            return

        self._raw_min = float(raw_scores.min())
        self._raw_max = float(raw_scores.max())

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_scores, binary_labels)
        self._iso = iso
        self._fitted = True

    def calibrate(self, score: float) -> float:
        """Return calibrated probability in [0, 1].

        Falls back to score/100 when not fitted.
        """
        if self._iso is None:
            return float(max(0.0, min(1.0, score / 100.0)))
        return float(self._iso.predict([score])[0])

    def to_dict(self) -> dict[str, Any]:
        import pickle
        import base64

        return {
            "fitted": self._fitted,
            "raw_min": self._raw_min,
            "raw_max": self._raw_max,
            "iso": base64.b64encode(pickle.dumps(self._iso)).decode("ascii")
            if self._iso is not None
            else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IsotonicCalibrator:
        import pickle
        import base64

        obj = cls()
        obj._fitted = d["fitted"]
        obj._raw_min = d["raw_min"]
        obj._raw_max = d["raw_max"]
        if d["iso"] is not None:
            obj._iso = pickle.loads(base64.b64decode(d["iso"].encode("ascii")))
        return obj
