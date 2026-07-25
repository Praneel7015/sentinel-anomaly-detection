"""Tests for the detector layer, fusion, classifier, drift, and scorer.

All tests use hand-constructed FeatureVector fixtures - no parquet files required.
"""
from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from sentinel.features.extractors import FEATURE_NAMES, FeatureVector

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_fv(overrides: dict[str, float] | None = None) -> FeatureVector:
    """Build a benign-looking FeatureVector, optionally with overrides."""
    values = {f: 0.0 for f in FEATURE_NAMES}
    # Reasonable benign defaults
    values["entity_event_count"] = 100.0
    values["graph_peer_resource_count"] = 15.0
    values["graph_jaccard_vs_cohort"] = 0.7
    values["days_observed"] = 30.0
    if overrides:
        values.update(overrides)
    return FeatureVector([values[f] for f in FEATURE_NAMES])


def _make_feature_df(
    n: int = 50,
    n_anomaly: int = 5,
    cohort: str = "finance_analyst",
    entity_id: str = "usr_0001",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a minimal feature DataFrame + labels DataFrame for fitting."""
    rng = np.random.default_rng(42)
    records = []
    labels = []
    for i in range(n):
        is_anom = i < n_anomaly
        row = {f: float(rng.normal(0, 1)) for f in FEATURE_NAMES}
        row["entity_id"] = entity_id
        row["cohort"] = cohort
        row["split"] = "train"
        row["event_id"] = f"evt_{i:04d}"

        if is_anom:
            # Make anomalous rows stand out
            row["geo_velocity_kmh"] = 5000.0
            row["entity_failure_count_5m"] = 20.0
            row["ip_failure_count_5m"] = 30.0

        records.append(row)
        labels.append({
            "event_id": f"evt_{i:04d}",
            "label": "brute_force" if is_anom else "normal",
        })

    return pd.DataFrame(records), pd.DataFrame(labels)


# ---------------------------------------------------------------------------
# 1. Statistical Profiler
# ---------------------------------------------------------------------------


class TestStatisticalProfiler:
    def test_higher_z_for_ood(self) -> None:
        from sentinel.models.profiler import StatisticalProfiler

        feature_df, _ = _make_feature_df(n=100, n_anomaly=0)
        profiler = StatisticalProfiler()
        profiler.fit(feature_df, split="train")

        # Benign-ish vector
        fv_normal = _make_fv({"geo_velocity_kmh": 0.0})
        # Clearly anomalous vector
        fv_anom = _make_fv({"geo_velocity_kmh": 5000.0})

        scores_normal = profiler.score(fv_normal, "usr_0001", "finance_analyst")
        scores_anom = profiler.score(fv_anom, "usr_0001", "finance_analyst")

        assert abs(scores_anom["geo_velocity_kmh"]) > abs(scores_normal["geo_velocity_kmh"])
        assert scores_anom["mahalanobis"] >= scores_normal["mahalanobis"]

    def test_cold_start_shrinks_to_cohort(self) -> None:
        from sentinel.models.profiler import StatisticalProfiler

        feature_df, _ = _make_feature_df(n=50)
        profiler = StatisticalProfiler(cohort_shrinkage_k=20)
        profiler.fit(feature_df, split="train")

        # Entity not in training → falls back entirely to cohort prior
        fv = _make_fv()
        result = profiler.score(fv, "unseen_entity", "finance_analyst")
        assert "mahalanobis" in result
        assert math.isfinite(result["mahalanobis"])

    def test_serialization_roundtrip(self) -> None:
        from sentinel.models.profiler import StatisticalProfiler

        feature_df, _ = _make_feature_df(n=30)
        profiler = StatisticalProfiler()
        profiler.fit(feature_df, split="train")

        d = profiler.to_dict()
        loaded = StatisticalProfiler.from_dict(d)

        fv = _make_fv()
        s1 = profiler.score(fv, "usr_0001", "finance_analyst")
        s2 = loaded.score(fv, "usr_0001", "finance_analyst")
        assert abs(s1["mahalanobis"] - s2["mahalanobis"]) < 1e-9


# ---------------------------------------------------------------------------
# 2. IsolationForest Detector
# ---------------------------------------------------------------------------


class TestIsolationForestDetector:
    def test_fit_and_score(self) -> None:
        from sentinel.models.isolation import IsolationForestDetector

        feature_df, _ = _make_feature_df(n=100)
        det = IsolationForestDetector(n_estimators=50, random_state=42)
        det.fit(feature_df, split="train")

        fv = _make_fv()
        score = det.score(fv, cohort="finance_analyst")
        assert 0.0 <= score <= 1.0

    def test_unknown_cohort_returns_neutral(self) -> None:
        from sentinel.models.isolation import IsolationForestDetector

        det = IsolationForestDetector()
        fv = _make_fv()
        score = det.score(fv, cohort="never_seen_cohort")
        assert score == 0.5

    def test_feature_importances(self) -> None:
        from sentinel.models.isolation import IsolationForestDetector

        feature_df, _ = _make_feature_df(n=60)
        det = IsolationForestDetector(n_estimators=30, random_state=42)
        det.fit(feature_df, split="train")

        imps = det.feature_importances("finance_analyst")
        assert set(imps.keys()) == set(FEATURE_NAMES)
        assert abs(sum(imps.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 3. Markov Detector
# ---------------------------------------------------------------------------


class TestMarkovDetector:
    def test_fit_and_score(self) -> None:
        from sentinel.models.markov import MarkovDetector

        feature_df, _ = _make_feature_df(n=80)
        det = MarkovDetector()
        det.fit(feature_df, split="train")

        fv_low = _make_fv({"command_surprisal": 0.0})
        fv_high = _make_fv({"command_surprisal": 20.0})

        score_low = det.score(fv_low, cohort="finance_analyst")
        score_high = det.score(fv_high, cohort="finance_analyst")

        assert score_low < score_high
        assert 0.0 <= score_low <= 1.0
        assert 0.0 <= score_high <= 1.0

    def test_no_data_returns_neutral(self) -> None:
        from sentinel.models.markov import MarkovDetector

        det = MarkovDetector()  # not fitted
        fv = _make_fv()
        score = det.score(fv, cohort="x")
        assert score == 0.5


# ---------------------------------------------------------------------------
# 4. Graph Detector
# ---------------------------------------------------------------------------


class TestGraphDetector:
    def test_novel_resource_increases_score(self) -> None:
        from sentinel.models.graph_detector import GraphDetector

        feature_df, _ = _make_feature_df(n=80)
        det = GraphDetector()
        det.fit(feature_df, split="train")

        fv_known = _make_fv({
            "graph_entity_novel_resource": 0.0,
            "graph_peer_resource_count": 20.0,
            "graph_jaccard_vs_cohort": 0.8,
            "graph_entity_degree_deviation": 0.0,
        })
        fv_novel = _make_fv({
            "graph_entity_novel_resource": 1.0,
            "graph_peer_resource_count": 0.0,
            "graph_jaccard_vs_cohort": 0.0,
            "graph_entity_degree_deviation": 3.0,
        })

        score_known = det.score(fv_known, cohort="finance_analyst")
        score_novel = det.score(fv_novel, cohort="finance_analyst")
        assert score_novel > score_known


# ---------------------------------------------------------------------------
# 5. Fusion
# ---------------------------------------------------------------------------


class TestLogOddsFusion:
    def _make_registry_fusion(self) -> "LogOddsFusion":  # noqa: F821
        from sentinel.models.fusion import LogOddsFusion

        fusion = LogOddsFusion(bias=-3.0)
        # Seed reservoirs with training data so rank is meaningful
        rng = np.random.default_rng(0)
        for name in ["profile", "isolation", "sequence", "graph"]:
            vals = rng.uniform(0, 1, size=600)
            for v in vals:
                fusion._reservoirs[name].update(float(v))
        fusion._fitted = True
        return fusion

    def test_risk_in_range(self) -> None:
        from sentinel.serving.models import DetectorScores

        fusion = self._make_registry_fusion()
        ds = DetectorScores(profile=0.5, isolation=0.5, sequence=0.5, graph=0.5)
        risk, _ = fusion.fuse(ds)
        assert 0.0 <= risk <= 100.0

    def test_monotone_in_profile_score(self) -> None:
        """Risk should increase as the profile score increases."""
        from sentinel.serving.models import DetectorScores

        fusion = self._make_registry_fusion()
        risks = []
        for profile_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            ds = DetectorScores(profile=profile_val, isolation=0.5, sequence=0.5, graph=0.5)
            r, _ = fusion.fuse(ds)
            risks.append(r)

        # Monotonically non-decreasing
        for i in range(len(risks) - 1):
            assert risks[i] <= risks[i + 1] + 0.1  # allow tiny float drift

    def test_contributions_sum_to_logit(self) -> None:
        """Sum of contributions + bias ≈ logit of the normalised risk score."""
        import math

        from sentinel.models.fusion import _sigmoid
        from sentinel.serving.models import DetectorScores

        fusion = self._make_registry_fusion()
        ds = DetectorScores(profile=0.8, isolation=0.7, sequence=0.6, graph=0.9)
        risk, contribs = fusion.fuse(ds)

        # logit of the output probability
        p = risk / 100.0
        if p <= 0 or p >= 1:
            return  # degenerate case, skip
        logit_out = math.log(p / (1 - p))

        # Contributions + bias should sum to the logit
        total_contrib = sum(contribs.values()) + fusion.bias
        assert abs(total_contrib - logit_out) < 0.5  # within half a logit unit

    def test_gru_none_uses_redistribution(self) -> None:
        from sentinel.serving.models import DetectorScores

        fusion = self._make_registry_fusion()
        ds_with = DetectorScores(profile=0.7, isolation=0.6, sequence=0.5, graph=0.4, gru=0.8)
        ds_without = DetectorScores(profile=0.7, isolation=0.6, sequence=0.5, graph=0.4, gru=None)
        risk_with, _ = fusion.fuse(ds_with)
        risk_without, _ = fusion.fuse(ds_without)
        # Both should produce valid scores
        assert 0.0 <= risk_with <= 100.0
        assert 0.0 <= risk_without <= 100.0


# ---------------------------------------------------------------------------
# 6. Calibration
# ---------------------------------------------------------------------------


class TestIsotonicCalibrator:
    def test_fit_and_calibrate(self) -> None:
        from sentinel.models.calibration import IsotonicCalibrator

        cal = IsotonicCalibrator()
        scores = np.array([10.0, 30.0, 50.0, 70.0, 90.0] * 20)
        labels = np.array([0, 0, 0, 1, 1] * 20)
        cal.fit(scores, labels)

        # High scores should have higher calibrated probability
        assert cal.calibrate(80.0) > cal.calibrate(20.0)

    def test_unfitted_fallback(self) -> None:
        from sentinel.models.calibration import IsotonicCalibrator

        cal = IsotonicCalibrator()
        assert abs(cal.calibrate(50.0) - 0.5) < 0.01


# ---------------------------------------------------------------------------
# 7. Classifier
# ---------------------------------------------------------------------------


class TestAttackClassifier:
    def test_brute_force_signature(self) -> None:
        from sentinel.models.classifier import AttackClassifier

        clf = AttackClassifier()
        fv = _make_fv({
            "ip_failure_count_5m": 20.0,
            "entity_failure_count_5m": 10.0,
        })
        label, conf, agreement, is_novel = clf.predict(fv)
        assert label == "brute_force"
        assert conf >= 0.5

    def test_impossible_travel_signature(self) -> None:
        from sentinel.models.classifier import AttackClassifier

        clf = AttackClassifier()
        fv = _make_fv({"geo_velocity_kmh": 2000.0})
        label, conf, agreement, is_novel = clf.predict(fv)
        assert label == "impossible_travel"
        assert conf >= 0.5

    def test_device_spoofing_signature(self) -> None:
        from sentinel.models.classifier import AttackClassifier

        clf = AttackClassifier()
        fv = _make_fv({
            "fingerprint_unknown": 1.0,
            "os_mismatch": 1.0,
            "mac_oui_change": 1.0,
        })
        label, conf, agreement, is_novel = clf.predict(fv)
        assert label == "device_spoofing"

    def test_lateral_movement_signature(self) -> None:
        from sentinel.models.classifier import AttackClassifier

        clf = AttackClassifier()
        fv = _make_fv({
            "cohort_novel_resource": 1.0,
            "distinct_resources_1h": 15.0,
        })
        label, conf, agreement, is_novel = clf.predict(fv)
        assert label == "lateral_movement"

    def test_credential_stuffing_signature(self) -> None:
        from sentinel.models.classifier import AttackClassifier

        clf = AttackClassifier()
        fv = _make_fv({
            "distinct_entities_per_ip": 10.0,
            "ip_failure_count_1h": 25.0,
        })
        label, conf, agreement, is_novel = clf.predict(fv)
        assert label == "credential_stuffing"

    def test_low_and_slow_exfil_signature(self) -> None:
        from sentinel.models.classifier import AttackClassifier

        clf = AttackClassifier()
        fv = _make_fv({
            "offhours_bytes_rolling_7d": 100_000.0,
            "transfer_to_duration_ratio": 1000.0,
        })
        label, conf, agreement, is_novel = clf.predict(fv)
        assert label == "low_and_slow_exfil"

    def test_gbm_fits_and_predicts(self) -> None:
        from sentinel.models.classifier import AttackClassifier

        feature_df, labels_df = _make_feature_df(n=200, n_anomaly=30)
        clf = AttackClassifier(max_iter=10)
        clf.fit(feature_df, labels_df)

        fv = _make_fv({"geo_velocity_kmh": 5000.0})
        label, conf, _, _ = clf.predict(fv)
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# 8. Page-Hinkley Drift Detector
# ---------------------------------------------------------------------------


class TestPageHinkleyDetector:
    def test_no_drift_on_stable_stream(self) -> None:
        from sentinel.drift.page_hinkley import PageHinkleyDetector

        ph = PageHinkleyDetector(delta=0.005, lambda_=50.0, min_instances=10)
        for _ in range(50):
            fired = ph.update(0.2)
            assert not fired

    def test_fires_after_sustained_drift(self) -> None:
        from sentinel.drift.page_hinkley import PageHinkleyDetector

        ph = PageHinkleyDetector(delta=0.005, lambda_=5.0, min_instances=5)

        # Baseline: stable low scores
        for _ in range(10):
            ph.update(0.1)

        # Drift: sudden sustained high scores
        fired = False
        for _ in range(200):
            if ph.update(0.9):
                fired = True
                break

        assert fired, "Should have detected drift on sustained high scores"

    def test_reset_clears_state(self) -> None:
        from sentinel.drift.page_hinkley import PageHinkleyDetector

        ph = PageHinkleyDetector(delta=0.005, lambda_=5.0, min_instances=5)
        # Trigger drift
        for _ in range(10):
            ph.update(0.1)
        for _ in range(200):
            ph.update(0.99)

        ph.reset()
        assert ph._n == 0
        assert ph._cumsum == 0.0
        assert not ph.drift_detected


# ---------------------------------------------------------------------------
# 9. Scorer (integration)
# ---------------------------------------------------------------------------


def _make_access_event() -> dict:
    """Build a minimal valid access event dict."""
    return {
        "event_id": "abc123def456789a",
        "episode_id": None,
        "entity_id": "usr_0001",
        "entity_type": "user",
        "cohort": "finance_analyst",
        "timestamp": datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC),
        "source_ip": "10.1.2.3",
        "geo_country": "US",
        "geo_city": "New York",
        "geo_lat": 40.71,
        "geo_lon": -74.01,
        "resource_accessed": "api:/v1/reports",
        "resource_type": "endpoint",
        "auth_method": "password",
        "auth_result": "success",
        "session_duration_s": 120.0,
        "command_sequence": [],
        "device_os": "Windows",
        "device_os_version": "11",
        "device_mac": "aa:bb:cc:dd:ee:ff",
        "device_protocol": "https",
        "device_fingerprint": "fp_abc",
        "bytes_transferred": 1024,
        "split": "test",
    }


class TestScorer:
    def _make_minimal_registry(self) -> "ModelRegistry":  # noqa: F821
        from sentinel.models.registry import ModelRegistry

        reg = ModelRegistry()
        # Fit on minimal data so scoring doesn't fail
        feature_df, labels_df = _make_feature_df(n=60, n_anomaly=5)
        reg.fit_all(
            pd.DataFrame(),  # events_df not required for minimal test
            labels_df,
            feature_df,
        )
        return reg

    def test_score_event_returns_valid_scored_event(self) -> None:
        from sentinel.features.pipeline import PipelineState
        from sentinel.models.classifier import AttackClassifier
        from sentinel.models.scorer import score_event

        pipeline_state = PipelineState()
        registry = self._make_minimal_registry()
        classifier = AttackClassifier()

        event = _make_access_event()
        scored = score_event(
            event,
            pipeline_state=pipeline_state,
            registry=registry,
            classifier=classifier,
        )

        assert 0.0 <= scored.risk_score <= 100.0
        assert scored.narrative != ""
        assert scored.entity_id == "usr_0001"
        assert scored.detector_scores.profile is not None
        assert scored.risk_band in ("low", "medium", "high", "critical")

    def test_score_event_all_fields_present(self) -> None:
        from sentinel.features.pipeline import PipelineState
        from sentinel.models.scorer import score_event

        pipeline_state = PipelineState()
        registry = self._make_minimal_registry()

        event = _make_access_event()
        scored = score_event(event, pipeline_state=pipeline_state, registry=registry)

        assert scored.entity_event_count >= 0
        assert isinstance(scored.contributions, list)
        assert isinstance(scored.counterfactuals, list)
        assert isinstance(scored.cold_start, bool)
        assert isinstance(scored.is_alert, bool)
        assert isinstance(scored.is_novel, bool)
        assert isinstance(scored.classifier_agreement, bool)


# ---------------------------------------------------------------------------
# 10. GRU Autoencoder
# ---------------------------------------------------------------------------


class TestGRUAutoencoder:
    def test_score_returns_none_when_not_fitted(self) -> None:
        from sentinel.models.torch_gru import GRUAutoencoder

        gru = GRUAutoencoder()
        fv = _make_fv()
        result = gru.score(fv)
        assert result is None

    def test_torch_available_and_fits(self) -> None:
        from sentinel.models.torch_gru import TORCH_AVAILABLE, GRUAutoencoder

        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")

        feature_df, labels_df = _make_feature_df(n=50, n_anomaly=5)
        gru = GRUAutoencoder(hidden_size=8, num_layers=1)
        gru.fit(feature_df, labels_df, epochs=2, batch_size=8)

        fv = _make_fv()
        score = gru.score(fv)
        assert score is not None
        assert 0.0 <= score <= 1.0

    def test_graceful_degradation_no_torch(self) -> None:
        """If torch is not available, score returns None without raising."""
        import sentinel.models.torch_gru as gru_mod

        original = gru_mod.TORCH_AVAILABLE
        gru_mod.TORCH_AVAILABLE = False
        try:
            gru = gru_mod.GRUAutoencoder()
            fv = _make_fv()
            result = gru.score(fv)
            assert result is None
        finally:
            gru_mod.TORCH_AVAILABLE = original


# ---------------------------------------------------------------------------
# 11. Attribution
# ---------------------------------------------------------------------------


class TestAttribution:
    def test_contributions_have_valid_direction(self) -> None:
        from sentinel.explain.attribution import attribute_contributions

        fv = _make_fv({"geo_velocity_kmh": 5000.0})
        detector_contribs = {"profile": 2.0, "isolation": 1.0, "sequence": 0.5, "graph": 0.3}
        z_scores = {f: 0.0 for f in FEATURE_NAMES}
        z_scores["geo_velocity_kmh"] = 4.5

        contribs = attribute_contributions(fv, detector_contribs, z_scores, top_k=6)

        assert len(contribs) > 0
        for c in contribs:
            assert c.direction in ("increases", "decreases")
            assert c.feature in FEATURE_NAMES
            assert isinstance(c.contribution, float)

    def test_top_k_limiting(self) -> None:
        from sentinel.explain.attribution import attribute_contributions

        fv = _make_fv()
        detector_contribs = {"profile": 3.0, "isolation": 1.0, "sequence": 0.5, "graph": 0.3}
        z_scores = {f: float(i) for i, f in enumerate(FEATURE_NAMES)}

        contribs = attribute_contributions(fv, detector_contribs, z_scores, top_k=3)
        assert len(contribs) <= 3


# ---------------------------------------------------------------------------
# 12. Narrative
# ---------------------------------------------------------------------------


class TestNarrative:
    def test_narrative_length(self) -> None:
        from sentinel.explain.attribution import attribute_contributions
        from sentinel.explain.narrative import build_narrative

        fv = _make_fv({"geo_velocity_kmh": 5000.0})
        detector_contribs = {"profile": 2.0, "isolation": 1.0, "sequence": 0.5, "graph": 0.3}
        z_scores = {f: 0.0 for f in FEATURE_NAMES}
        z_scores["geo_velocity_kmh"] = 4.5

        contribs = attribute_contributions(fv, detector_contribs, z_scores)
        narrative = build_narrative(contribs, risk_score=78.0)

        assert len(narrative) <= 250  # allow slight slack
        assert isinstance(narrative, str)
        assert len(narrative) > 0

    def test_empty_contributions(self) -> None:
        from sentinel.explain.narrative import build_narrative

        narrative = build_narrative([], risk_score=42.0)
        assert "42" in narrative
