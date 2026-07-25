"""Model registry - loads/saves all detectors and runs coordinated scoring.

fit_all(events_df, labels_df, feature_df) trains every detector.
score_all(fv, entity_id, cohort, entity_state) -> DetectorScores.
save(path) / load(path) persist to artifacts/model_registry.pkl.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from sentinel.features.extractors import FEATURE_NAMES, FeatureVector
from sentinel.models.calibration import IsotonicCalibrator
from sentinel.models.fusion import LogOddsFusion
from sentinel.models.graph_detector import GraphDetector
from sentinel.models.isolation import IsolationForestDetector
from sentinel.models.markov import MarkovDetector
from sentinel.models.profiler import StatisticalProfiler
from sentinel.models.torch_gru import GRUAutoencoder
from sentinel.serving.models import DetectorScores

__all__ = ["ModelRegistry"]

logger = logging.getLogger(__name__)

_ARTIFACTS_DIR = Path("artifacts")


class ModelRegistry:
    """Coordinates training and scoring of all detectors.

    Usage::

        registry = ModelRegistry.from_config(cfg)
        registry.fit_all(events_df, labels_df, feature_df)
        scores = registry.score_all(fv, entity_id, cohort, entity_state)
        registry.save()
    """

    def __init__(
        self,
        profiler: StatisticalProfiler | None = None,
        isolation: IsolationForestDetector | None = None,
        markov: MarkovDetector | None = None,
        graph: GraphDetector | None = None,
        gru: GRUAutoencoder | None = None,
        fusion: LogOddsFusion | None = None,
        calibrator: IsotonicCalibrator | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        cfg = config or {}

        self.profiler = profiler or StatisticalProfiler(
            cohort_shrinkage_k=cfg.get("profile", {}).get("cohort_shrinkage_k", 20),
            min_variance=cfg.get("profile", {}).get("min_variance", 1e-6),
        )
        self.isolation = isolation or IsolationForestDetector(
            n_estimators=cfg.get("isolation", {}).get("n_estimators", 200),
            max_samples=cfg.get("isolation", {}).get("max_samples", 4096),
            contamination=cfg.get("isolation", {}).get("contamination", "auto"),
            random_state=cfg.get("isolation", {}).get("random_state", 20260725),
        )
        self.markov = markov or MarkovDetector()
        self.graph = graph or GraphDetector()
        self.gru = gru or GRUAutoencoder(
            hidden_size=cfg.get("gru", {}).get("hidden_size", 64),
            num_layers=cfg.get("gru", {}).get("num_layers", 1),
            learning_rate=cfg.get("gru", {}).get("learning_rate", 1e-3),
        )
        self.fusion = fusion or LogOddsFusion(
            weights=cfg.get("fusion", {}).get("weights"),
            bias=cfg.get("fusion", {}).get("bias", -3.0),
            logit_clip=cfg.get("fusion", {}).get("logit_clip", 6.0),
        )
        self.calibrator = calibrator or IsotonicCalibrator()
        self._gru_epochs: int = cfg.get("gru", {}).get("epochs", 8)
        self._gru_batch_size: int = cfg.get("gru", {}).get("batch_size", 256)

    # ---------------------------------------------------------------------- #
    # Training
    # ---------------------------------------------------------------------- #

    def fit_all(
        self,
        events_df: pd.DataFrame,
        labels_df: pd.DataFrame | None,
        feature_df: pd.DataFrame,
    ) -> None:
        """Fit every detector on the train split.

        Args:
            events_df: raw event DataFrame (schema.AccessEvent rows).
            labels_df: label DataFrame (schema.Label rows), may be None.
            feature_df: pre-computed feature DataFrame from batch_features().
        """
        logger.info("Fitting statistical profiler...")
        self.profiler.fit(feature_df, split="train")

        logger.info("Fitting IsolationForest detectors...")
        self.isolation.fit(feature_df, split="train")

        logger.info("Fitting Markov detector...")
        self.markov.fit(feature_df, split="train")

        logger.info("Fitting graph detector...")
        self.graph.fit(feature_df, split="train")

        logger.info("Fitting GRU autoencoder...")
        try:
            self.gru.fit(
                feature_df,
                labels_df,
                epochs=self._gru_epochs,
                batch_size=self._gru_batch_size,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("GRU training failed (non-fatal): %s", exc)

        # Compute training scores for fusion quantile fitting
        logger.info("Fitting fusion quantiles...")
        self._fit_fusion_quantiles(feature_df)

        logger.info("All detectors fitted.")

    def _fit_fusion_quantiles(self, feature_df: pd.DataFrame) -> None:
        """Score all training events through each detector to calibrate quantiles."""
        profile_scores: list[float] = []
        isolation_scores: list[float] = []
        markov_scores: list[float] = []
        graph_scores: list[float] = []

        train_df = (
            feature_df[feature_df["split"] == "train"]
            if "split" in feature_df.columns
            else feature_df
        )

        for row in train_df.itertuples(index=False):
            vals = [float(getattr(row, fname, 0.0)) for fname in FEATURE_NAMES]
            fv = FeatureVector(vals)

            entity_id = str(getattr(row, "entity_id", "unknown"))
            cohort = str(getattr(row, "cohort", "default"))

            prof_scores = self.profiler.score(fv, entity_id=entity_id, cohort=cohort)
            profile_scores.append(prof_scores.get("mahalanobis", 0.0))
            isolation_scores.append(self.isolation.score(fv, cohort=cohort))
            markov_scores.append(self.markov.score(fv, cohort=cohort))
            graph_scores.append(self.graph.score(fv, cohort=cohort))

        train_scores_df = pd.DataFrame(
            {
                "profile": profile_scores,
                "isolation": isolation_scores,
                "sequence": markov_scores,
                "graph": graph_scores,
            }
        )
        self.fusion.fit_quantiles(train_scores_df)

    # ---------------------------------------------------------------------- #
    # Scoring
    # ---------------------------------------------------------------------- #

    def score_all(
        self,
        fv: FeatureVector,
        entity_id: str,
        cohort: str,
        entity_state: Any | None = None,
    ) -> DetectorScores:
        """Score an event through all detectors.

        Returns a DetectorScores pydantic object.
        """
        # Profile detector: use mahalanobis as the 0-1 score
        prof_result = self.profiler.score(fv, entity_id=entity_id, cohort=cohort)
        mahal = prof_result.get("mahalanobis", 0.0)
        # Normalise mahalanobis to [0, 1] using a rough scale (expected max ~5)
        profile_score = float(max(0.0, min(1.0, mahal / 5.0)))

        isolation_score = self.isolation.score(fv, cohort=cohort)
        sequence_score = self.markov.score(fv, cohort=cohort)
        graph_score = self.graph.score(fv, cohort=cohort)
        gru_score = self.gru.score(fv, entity_state=entity_state)

        return DetectorScores(
            profile=profile_score,
            isolation=isolation_score,
            sequence=sequence_score,
            graph=graph_score,
            gru=gru_score,
        )

    def get_profiler_scores(
        self,
        fv: FeatureVector,
        entity_id: str,
        cohort: str,
    ) -> dict[str, float]:
        """Return per-feature z-scores from the profiler (used by attribution)."""
        return self.profiler.score(fv, entity_id=entity_id, cohort=cohort)

    # ---------------------------------------------------------------------- #
    # Save / load
    # ---------------------------------------------------------------------- #

    def save(self, path: str | Path | None = None) -> None:
        """Persist the registry to ``artifacts/model_registry.pkl``."""
        dest = Path(path) if path else _ARTIFACTS_DIR / "model_registry.pkl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "profiler": self.profiler.to_dict(),
            "isolation": self.isolation.to_dict(),
            "markov": self.markov.to_dict(),
            "graph": self.graph.to_dict(),
            "gru": self.gru.to_dict(),
            "fusion": self.fusion.to_dict(),
            "calibrator": self.calibrator.to_dict(),
        }
        with open(dest, "wb") as f:
            pickle.dump(payload, f)
        # Also save GRU weights separately
        self.gru.save()
        logger.info("Registry saved to %s", dest)

    @classmethod
    def load(cls, path: str | Path | None = None) -> ModelRegistry:
        """Load registry from ``artifacts/model_registry.pkl``."""
        src = Path(path) if path else _ARTIFACTS_DIR / "model_registry.pkl"
        with open(src, "rb") as f:
            payload = pickle.load(f)

        registry = cls.__new__(cls)
        registry.profiler = StatisticalProfiler.from_dict(payload["profiler"])
        registry.isolation = IsolationForestDetector.from_dict(payload["isolation"])
        registry.markov = MarkovDetector.from_dict(payload["markov"])
        registry.graph = GraphDetector.from_dict(payload["graph"])
        registry.gru = GRUAutoencoder.from_dict(payload["gru"])
        registry.fusion = LogOddsFusion.from_dict(payload["fusion"])
        registry.calibrator = IsotonicCalibrator.from_dict(payload["calibrator"])
        registry._gru_epochs = 8
        registry._gru_batch_size = 256

        # Try loading GRU weights
        registry.gru.load()
        logger.info("Registry loaded from %s", src)
        return registry

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ModelRegistry:
        """Construct a registry from a model.yaml config dict."""
        return cls(config=config)
