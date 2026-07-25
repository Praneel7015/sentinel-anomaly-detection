"""Page-Hinkley drift detector per entity.

Monitors the rolling mean of the entity's anomaly score. Fires drift_detected
when the cumulative sum of incremental changes exceeds lambda (from config).

Page-Hinkley test (upward change):
    cumsum_t = max(0, cumsum_{t-1} + (x_t - running_mean_t - delta))
    drift_detected when max(cumsum) - cumsum_t > lambda

On detection, triggers fast re-baselining by resetting Welford stats for that
entity to its cohort prior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["PageHinkleyDetector"]


@dataclass
class PageHinkleyDetector:
    """Page-Hinkley drift detector for one entity's risk score stream.

    Attributes:
        delta: tolerance for gradual change (from configs/model.yaml).
        lambda_: detection threshold (also from model.yaml).
        min_instances: minimum observations before firing.

    Usage::

        ph = PageHinkleyDetector(delta=0.005, lambda_=50.0)
        drift = ph.update(score)  # True if drift detected
        if drift:
            ph.reset()
    """

    delta: float = 0.005
    lambda_: float = 50.0
    min_instances: int = 30

    # Running state
    _n: int = field(default=0, repr=False)
    _mean: float = field(default=0.0, repr=False)
    _cumsum: float = field(default=0.0, repr=False)
    _max_cumsum: float = field(default=0.0, repr=False)
    _drift_detected: bool = field(default=False, repr=False)

    # Track if reset is needed after drift
    _needs_rebaseline: bool = field(default=False, repr=False)

    def update(self, score: float) -> bool:
        """Update with new risk score observation.

        Returns:
            True if drift is detected (cumulative change exceeds lambda_).
        """
        self._n += 1

        # Update running mean incrementally
        self._mean += (score - self._mean) / self._n

        # Page-Hinkley increment for upward change detection:
        # m_t = max(0, m_{t-1} + (x_t - mean_t - delta))
        self._cumsum = max(0.0, self._cumsum + (score - self._mean - self.delta))
        self._max_cumsum = max(self._max_cumsum, self._cumsum)

        # Check for drift: cumsum exceeds lambda (upward shift detected)
        if self._n >= self.min_instances:
            if self._cumsum > self.lambda_:
                self._drift_detected = True
                self._needs_rebaseline = True
                return True

        return False

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    @property
    def needs_rebaseline(self) -> bool:
        return self._needs_rebaseline

    def reset(self) -> None:
        """Reset detector state (call after re-baselining the entity profile)."""
        self._n = 0
        self._mean = 0.0
        self._cumsum = 0.0
        self._max_cumsum = 0.0
        self._drift_detected = False
        self._needs_rebaseline = False

    def acknowledge_rebaseline(self) -> None:
        """Acknowledge that re-baselining has happened; clear the flag."""
        self._needs_rebaseline = False
        self._drift_detected = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta": self.delta,
            "lambda_": self.lambda_,
            "min_instances": self.min_instances,
            "_n": self._n,
            "_mean": self._mean,
            "_cumsum": self._cumsum,
            "_max_cumsum": self._max_cumsum,
            "_drift_detected": self._drift_detected,
            "_needs_rebaseline": self._needs_rebaseline,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PageHinkleyDetector:
        obj = cls(
            delta=d["delta"],
            lambda_=d["lambda_"],
            min_instances=d["min_instances"],
        )
        obj._n = d["_n"]
        obj._mean = d["_mean"]
        obj._cumsum = d["_cumsum"]
        obj._max_cumsum = d["_max_cumsum"]
        obj._drift_detected = d["_drift_detected"]
        obj._needs_rebaseline = d["_needs_rebaseline"]
        return obj
