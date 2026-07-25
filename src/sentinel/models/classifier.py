"""Attack-type classifier.

Two parallel components:
1. Signature matcher: transparent if/elif rules over feature values.
   Covers all 6 attack types with at least 2 rules each.
   Confidence degrades with partial matches.

2. HistGradientBoostingClassifier: trained on anomaly-labelled rows only.
   Uses FeatureVector.to_numpy() as input.

When both agree → classifier_agreement = True.
When they disagree → label = "unknown_novel", is_novel = True.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

from sentinel.features.extractors import FEATURE_NAMES, FeatureVector
from sentinel.schema import ATTACK_TYPES, LABEL_NORMAL, LABEL_UNKNOWN_NOVEL

__all__ = ["AttackClassifier"]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Signature rules
# --------------------------------------------------------------------------- #

def _signature_match(fv: FeatureVector) -> tuple[str, float]:
    """Transparent rule-based signature matcher.

    Returns (attack_type_or_normal, confidence) in [0, 1].
    Confidence < 0.35 → unknown_novel at the end.
    """
    # --- brute_force ---
    # Strong: many failures from same IP + entity, then success
    if (
        fv["ip_failure_count_5m"] > 15
        and fv["entity_failure_count_5m"] > 5
    ):
        conf = min(1.0, 0.7 + 0.01 * (fv["ip_failure_count_5m"] - 15))
        return "brute_force", conf

    # Moderate: success after many entity failures
    if fv["success_after_failures"] > 0 and fv["entity_failure_count_5m"] > 8:
        return "brute_force", 0.75

    # --- credential_stuffing ---
    # Many distinct entities from same IP, some failures
    if (
        fv["distinct_entities_per_ip"] > 5
        and fv["ip_failure_count_1h"] > 10
    ):
        conf = min(1.0, 0.65 + 0.02 * (fv["distinct_entities_per_ip"] - 5))
        return "credential_stuffing", conf

    # IP hitting many entities with high failure rate
    if (
        fv["distinct_entities_per_ip"] > 8
        and fv["ip_failure_count_5m"] > 5
    ):
        return "credential_stuffing", 0.72

    # --- impossible_travel ---
    # Very high geo-velocity (> 1000 km/h implies physical impossibility)
    if fv["geo_velocity_kmh"] > 1000.0:
        conf = min(1.0, 0.70 + 0.0001 * (fv["geo_velocity_kmh"] - 1000))
        return "impossible_travel", conf

    # New country + high velocity
    if fv["new_country_flag"] > 0 and fv["geo_velocity_kmh"] > 500.0:
        return "impossible_travel", 0.68

    # --- lateral_movement ---
    # Many distinct resources + cohort-novel resource
    if (
        fv["cohort_novel_resource"] > 0
        and fv["distinct_resources_1h"] > 8
    ):
        conf = min(1.0, 0.65 + 0.02 * fv["distinct_resources_1h"])
        return "lateral_movement", conf

    # Graph: peer count very low + novel resource edge
    if (
        fv["graph_peer_resource_count"] < 2
        and fv["graph_entity_novel_resource"] > 0
        and fv["entity_novel_resource"] > 0
    ):
        return "lateral_movement", 0.68

    # --- device_spoofing ---
    # New fingerprint + OS mismatch + MAC OUI change
    if (
        fv["fingerprint_unknown"] > 0
        and fv["os_mismatch"] > 0
        and fv["mac_oui_change"] > 0
    ):
        return "device_spoofing", 0.85

    # New fingerprint + protocol novelty
    if fv["fingerprint_unknown"] > 0 and fv["protocol_novelty"] > 0:
        return "device_spoofing", 0.65

    # --- low_and_slow_exfil ---
    # High off-hours bytes + high transfer ratio + resource breadth expansion
    if (
        fv["offhours_bytes_rolling_7d"] > 50_000
        and fv["transfer_to_duration_ratio"] > 500
    ):
        conf = min(1.0, 0.60 + 0.0001 * fv["offhours_bytes_rolling_7d"] / 1000)
        return "low_and_slow_exfil", conf

    # Anomalous bytes z-score + off-hours
    if fv["bytes_zscore"] > 3.0 and fv["is_offhours"] > 0:
        return "low_and_slow_exfil", 0.62

    # Default: normal (no rule fired)
    return LABEL_NORMAL, 0.5


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #


class AttackClassifier:
    """Dual-mode attack-type classifier.

    Usage::

        clf = AttackClassifier()
        clf.fit(feature_df, labels_df)
        attack_type, confidence, agreement, is_novel = clf.predict(fv)
    """

    def __init__(
        self,
        max_iter: int = 300,
        learning_rate: float = 0.08,
        max_depth: int = 6,
        min_confidence: float = 0.35,
        random_state: int = 20260725,
    ) -> None:
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_confidence = min_confidence
        self.random_state = random_state

        self._model: HistGradientBoostingClassifier | None = None
        self._encoder: LabelEncoder | None = None
        self._fitted = False

    # ---------------------------------------------------------------------- #
    # Fitting
    # ---------------------------------------------------------------------- #

    def fit(self, feature_df: pd.DataFrame, labels_df: pd.DataFrame) -> None:
        """Train HistGBM on anomaly-labelled rows (train split only).

        Args:
            feature_df: DataFrame with FEATURE_NAMES columns, ``entity_id``,
                        ``event_id``, and ``split`` columns.
            labels_df: DataFrame with ``event_id`` and ``label`` columns.
        """
        if labels_df is None or len(labels_df) == 0:
            logger.warning("No labels provided; HistGBM not trained")
            self._fitted = True
            return

        # Merge labels
        df = feature_df.merge(
            labels_df[["event_id", "label"]], on="event_id", how="left"
        )
        if "label" not in df.columns:
            logger.warning("label column missing after merge")
            self._fitted = True
            return

        # Filter: train split + anomaly labels only
        mask = df["split"] == "train"
        anomaly_mask = df["label"].isin(ATTACK_TYPES)
        train_anom = df[mask & anomaly_mask]

        if len(train_anom) < 2:
            logger.warning(
                "Fewer than 2 anomaly rows in train split; HistGBM not trained"
            )
            self._fitted = True
            return

        X = train_anom[FEATURE_NAMES].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y_raw = train_anom["label"].values

        enc = LabelEncoder()
        y = enc.fit_transform(y_raw)

        model = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            class_weight="balanced",
            random_state=self.random_state,
        )
        model.fit(X, y)

        self._model = model
        self._encoder = enc
        self._fitted = True
        logger.info(
            "HistGBM fitted on %d anomaly events, classes: %s",
            len(train_anom),
            list(enc.classes_),
        )

    # ---------------------------------------------------------------------- #
    # Prediction
    # ---------------------------------------------------------------------- #

    def predict(
        self, fv: FeatureVector
    ) -> tuple[str, float, bool, bool]:
        """Predict attack type using both signature and HistGBM.

        Returns:
            (attack_type, confidence, classifier_agreement, is_novel)
        """
        # Signature matcher
        sig_label, sig_conf = _signature_match(fv)

        # HistGBM prediction
        gbm_label, gbm_conf = self._predict_gbm(fv)

        # Resolution logic
        if gbm_label is None:
            # Only signature available
            if sig_label == LABEL_NORMAL or sig_conf < self.min_confidence:
                return LABEL_NORMAL, sig_conf, True, False
            return sig_label, sig_conf, True, False

        # Both available
        if sig_label == gbm_label:
            # Agreement
            agreement = True
            best_conf = max(sig_conf, gbm_conf)
            label = sig_label
            is_novel = False
        else:
            # Disagreement
            agreement = False
            # If one says normal and the other says attack, check confidence
            if sig_label == LABEL_NORMAL and gbm_conf >= self.min_confidence:
                # GBM detects something the rules miss
                label = gbm_label
                best_conf = gbm_conf
                is_novel = False
            elif gbm_label == LABEL_NORMAL and sig_conf >= self.min_confidence:
                # Rules detect something GBM misses
                label = sig_label
                best_conf = sig_conf
                is_novel = False
            else:
                # Two attack types disagree → novel
                label = LABEL_UNKNOWN_NOVEL
                best_conf = max(sig_conf, gbm_conf) * 0.7  # reduced confidence
                is_novel = True

        if best_conf < self.min_confidence and label != LABEL_NORMAL:
            label = LABEL_UNKNOWN_NOVEL
            is_novel = True

        return label, float(best_conf), agreement, is_novel

    def _predict_gbm(
        self, fv: FeatureVector
    ) -> tuple[str | None, float]:
        """Run HistGBM and return (label, confidence). None if not fitted."""
        if self._model is None or self._encoder is None:
            return None, 0.0

        x = fv.to_numpy().reshape(1, -1).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        proba = self._model.predict_proba(x)[0]
        best_idx = int(np.argmax(proba))
        best_conf = float(proba[best_idx])
        label = str(self._encoder.classes_[best_idx])
        return label, best_conf

    # ---------------------------------------------------------------------- #
    # Persistence
    # ---------------------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        import pickle
        import base64

        return {
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_confidence": self.min_confidence,
            "random_state": self.random_state,
            "fitted": self._fitted,
            "model": base64.b64encode(pickle.dumps(self._model)).decode("ascii")
            if self._model is not None
            else None,
            "encoder": base64.b64encode(pickle.dumps(self._encoder)).decode("ascii")
            if self._encoder is not None
            else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AttackClassifier:
        import pickle
        import base64

        obj = cls(
            max_iter=d["max_iter"],
            learning_rate=d["learning_rate"],
            max_depth=d["max_depth"],
            min_confidence=d["min_confidence"],
            random_state=d["random_state"],
        )
        obj._fitted = d["fitted"]
        if d["model"] is not None:
            obj._model = pickle.loads(base64.b64decode(d["model"].encode("ascii")))
        if d["encoder"] is not None:
            obj._encoder = pickle.loads(base64.b64decode(d["encoder"].encode("ascii")))
        return obj
