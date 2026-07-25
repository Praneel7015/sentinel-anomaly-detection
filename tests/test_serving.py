"""FastAPI endpoint tests for the SENTINEL serving layer.

Uses :class:`fastapi.testclient.TestClient` (synchronous) against the real
FastAPI application wired with the stub scorer.  No trained model is required.

Coverage targets from the task:
* GET  /api/health        → 200 + required fields
* POST /api/score         → ScoredEvent with all required fields, risk in [0,100]
* GET  /api/alerts        → paginated; min_risk filter; budget_pct changes flagged set
* POST /api/feedback      → {ok: true}
* GET  /api/entities/{id} → expected shape after scoring 3 events for that entity
* GET  /api/metrics       → 200 even without eval data
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from sentinel.serving.app import (
    _alert_store,
    _entity_store,
    _feedback_store,
    _stats,
    app,
)

# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #

_BASE_EVENT: dict = {
    "event_id": "deadbeefdeadbeef",
    "episode_id": None,
    "entity_id": "usr_test01",
    "entity_type": "user",
    "cohort": "finance_analyst",
    "timestamp": "2026-02-14T09:00:00Z",
    "source_ip": "10.0.0.1",
    "geo_country": "US",
    "geo_city": "New York",
    "geo_lat": 40.71,
    "geo_lon": -74.01,
    "resource_accessed": "fs:/finance/q3.xlsx",
    "resource_type": "file",
    "auth_method": "mfa_push",
    "auth_result": "success",
    "session_duration_s": 120.0,
    "command_sequence": ["read"],
    "device_os": "Windows",
    "device_os_version": "11",
    "device_mac": "aa:bb:cc:dd:ee:ff",
    "device_protocol": "https",
    "device_fingerprint": "fp_abc123",
    "bytes_transferred": 4096,
    "split": "test",
}


def _make_event(**overrides) -> dict:
    ev = dict(_BASE_EVENT)
    ev.update(overrides)
    return ev


@pytest.fixture(autouse=True)
def _reset_stores():
    """Reset all in-memory stores between tests."""
    _alert_store._deque.clear()
    _alert_store._by_id.clear()
    _entity_store._records.clear()
    _stats.reset()
    _feedback_store._entries.clear()
    _feedback_store._entity_threshold_lowering.clear()
    yield


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


# --------------------------------------------------------------------------- #
# GET /api/health
# --------------------------------------------------------------------------- #


def test_health_returns_200(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200


def test_health_has_required_fields(client: TestClient) -> None:
    data = client.get("/api/health").json()
    assert "status" in data
    assert "version" in data
    assert "model_loaded" in data
    assert "torch_available" in data
    assert data["status"] in ("ok", "degraded")
    assert isinstance(data["model_loaded"], bool)
    assert isinstance(data["torch_available"], bool)


# --------------------------------------------------------------------------- #
# GET /api/stats
# --------------------------------------------------------------------------- #


def test_stats_returns_200(client: TestClient) -> None:
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    for key in ("events_processed", "alerts_raised", "alerts_by_type", "events_per_sec", "uptime_s"):
        assert key in data


# --------------------------------------------------------------------------- #
# POST /api/score
# --------------------------------------------------------------------------- #


def test_score_returns_scored_event(client: TestClient) -> None:
    resp = client.post("/api/score", json=_make_event())
    assert resp.status_code == 200
    data = resp.json()
    for key in (
        "event_id", "entity_id", "entity_type", "cohort", "timestamp",
        "risk_score", "risk_band", "is_alert",
        "predicted_attack_type", "attack_type_confidence",
        "classifier_agreement", "is_novel",
        "detector_scores", "contributions", "narrative", "counterfactuals",
        "cold_start", "entity_event_count", "event", "ground_truth",
    ):
        assert key in data, f"Missing key: {key}"


def test_score_risk_in_valid_range(client: TestClient) -> None:
    data = client.post("/api/score", json=_make_event()).json()
    assert 0.0 <= data["risk_score"] <= 100.0


def test_score_risk_band_matches_score(client: TestClient) -> None:
    data = client.post("/api/score", json=_make_event()).json()
    score = data["risk_score"]
    band = data["risk_band"]
    if score < 40:
        assert band == "low"
    elif score < 65:
        assert band == "medium"
    elif score < 85:
        assert band == "high"
    else:
        assert band == "critical"


def test_score_detector_scores_present(client: TestClient) -> None:
    data = client.post("/api/score", json=_make_event()).json()
    ds = data["detector_scores"]
    for name in ("profile", "isolation", "sequence", "graph"):
        assert name in ds
        assert 0.0 <= ds[name] <= 1.0
    assert "gru" in ds  # may be None


def test_score_rejects_extra_fields(client: TestClient) -> None:
    bad = _make_event()
    bad["not_a_field"] = "oops"
    resp = client.post("/api/score", json=bad)
    assert resp.status_code == 422


def test_score_appended_to_alert_store(client: TestClient) -> None:
    client.post("/api/score", json=_make_event(event_id="aabbccddeeff0011"))
    assert _alert_store.get("aabbccddeeff0011") is not None


# --------------------------------------------------------------------------- #
# GET /api/alerts
# --------------------------------------------------------------------------- #


def _score_n_events(client: TestClient, n: int, entity_id: str = "usr_test01") -> list[dict]:
    results = []
    for i in range(n):
        eid = f"{i:016x}"
        resp = client.post("/api/score", json=_make_event(event_id=eid, entity_id=entity_id))
        assert resp.status_code == 200
        results.append(resp.json())
    return results


def test_alerts_returns_paginated_result(client: TestClient) -> None:
    _score_n_events(client, 5)
    resp = client.get("/api/alerts?limit=3&offset=0")
    assert resp.status_code == 200
    data = resp.json()
    assert "alerts" in data and "total" in data
    assert len(data["alerts"]) <= 3
    assert data["total"] >= 0


def test_alerts_total_is_count_before_pagination(client: TestClient) -> None:
    _score_n_events(client, 6)
    full = client.get("/api/alerts?limit=500").json()
    paged = client.get("/api/alerts?limit=2&offset=0").json()
    assert paged["total"] == full["total"]
    assert len(paged["alerts"]) <= 2


def test_alerts_min_risk_filter(client: TestClient) -> None:
    _score_n_events(client, 20)
    threshold = 60.0
    data = client.get(f"/api/alerts?min_risk={threshold}&limit=500").json()
    for alert in data["alerts"]:
        assert alert["risk_score"] >= threshold, f"Got {alert['risk_score']} < {threshold}"


def test_alerts_entity_filter(client: TestClient) -> None:
    _score_n_events(client, 3, entity_id="usr_alpha")
    _score_n_events(client, 3, entity_id="usr_beta")
    data = client.get("/api/alerts?entity_id=usr_alpha&limit=500").json()
    for alert in data["alerts"]:
        assert alert["entity_id"] == "usr_alpha"


def test_alerts_attack_type_filter(client: TestClient) -> None:
    _score_n_events(client, 15)
    all_data = client.get("/api/alerts?limit=500").json()
    if not all_data["alerts"]:
        pytest.skip("no alerts in store")

    attack_type = all_data["alerts"][0]["predicted_attack_type"]
    filtered = client.get(f"/api/alerts?attack_type={attack_type}&limit=500").json()
    for alert in filtered["alerts"]:
        assert alert["predicted_attack_type"] == attack_type


def test_alerts_budget_pct_changes_flagged_set(client: TestClient) -> None:
    """Higher budget_pct → lower threshold → more events pass the threshold."""
    _score_n_events(client, 30)
    low_budget = client.get("/api/alerts?budget_pct=0.5&limit=500").json()
    high_budget = client.get("/api/alerts?budget_pct=2.0&limit=500").json()
    # 2% budget has a lower risk threshold → at least as many alerts as 0.5%
    assert high_budget["total"] >= low_budget["total"]


def test_alerts_sort_risk_desc(client: TestClient) -> None:
    _score_n_events(client, 10)
    data = client.get("/api/alerts?sort=risk_desc&limit=500").json()
    scores = [a["risk_score"] for a in data["alerts"]]
    assert scores == sorted(scores, reverse=True)


def test_alerts_sort_risk_asc(client: TestClient) -> None:
    _score_n_events(client, 10)
    data = client.get("/api/alerts?sort=risk_asc&limit=500").json()
    scores = [a["risk_score"] for a in data["alerts"]]
    assert scores == sorted(scores)


# --------------------------------------------------------------------------- #
# GET /api/alerts/{event_id}
# --------------------------------------------------------------------------- #


def test_alert_detail_returns_entity_summary(client: TestClient) -> None:
    resp = client.post("/api/score", json=_make_event(event_id="1122334455667788"))
    assert resp.status_code == 200
    detail = client.get("/api/alerts/1122334455667788").json()
    assert "entity_summary" in detail
    assert "similar_alerts" in detail
    es = detail["entity_summary"]
    assert "entity_id" in es
    assert "event_count" in es
    assert "cold_start" in es


def test_alert_detail_404_for_unknown(client: TestClient) -> None:
    resp = client.get("/api/alerts/0000000000000000")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# POST /api/feedback
# --------------------------------------------------------------------------- #


def test_feedback_returns_ok_true(client: TestClient) -> None:
    # Score an event first
    client.post("/api/score", json=_make_event(event_id="feedbeef00000001"))
    resp = client.post(
        "/api/feedback",
        json={"event_id": "feedbeef00000001", "verdict": "false_positive", "note": "test"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True


def test_feedback_stores_verdict(client: TestClient) -> None:
    client.post("/api/score", json=_make_event(event_id="feedbeef00000002"))
    client.post(
        "/api/feedback",
        json={"event_id": "feedbeef00000002", "verdict": "true_positive"},
    )
    assert len(_feedback_store.all()) == 1
    assert _feedback_store.all()[0].verdict == "true_positive"


def test_feedback_false_positive_returns_updated_threshold(client: TestClient) -> None:
    client.post("/api/score", json=_make_event(event_id="feedbeef00000003"))
    resp = client.post(
        "/api/feedback",
        json={"event_id": "feedbeef00000003", "verdict": "false_positive"},
    )
    data = resp.json()
    assert data["ok"] is True
    # false_positive should lower threshold → updated_threshold is a float
    assert data["updated_threshold"] is not None
    assert isinstance(data["updated_threshold"], float)


def test_feedback_non_fp_does_not_change_threshold(client: TestClient) -> None:
    client.post("/api/score", json=_make_event(event_id="feedbeef00000004"))
    resp = client.post(
        "/api/feedback",
        json={"event_id": "feedbeef00000004", "verdict": "true_positive"},
    )
    data = resp.json()
    assert data["ok"] is True
    assert data["updated_threshold"] is None


# --------------------------------------------------------------------------- #
# GET /api/entities/{entity_id}
# --------------------------------------------------------------------------- #


def test_entity_shape_after_three_events(client: TestClient) -> None:
    for i in range(3):
        client.post(
            "/api/score",
            json=_make_event(event_id=f"ent{i:013x}", entity_id="usr_entity99"),
        )

    resp = client.get("/api/entities/usr_entity99")
    assert resp.status_code == 200
    data = resp.json()

    for key in (
        "entity_id", "entity_type", "cohort",
        "first_seen", "last_seen", "event_count", "cold_start",
        "profile_summary", "risk_timeline", "activity_by_hour",
        "top_resources", "peer_comparison", "drift_state",
    ):
        assert key in data, f"Missing field: {key}"

    assert data["entity_id"] == "usr_entity99"
    assert data["event_count"] == 3
    assert len(data["activity_by_hour"]) == 24
    assert isinstance(data["cold_start"], bool)
    assert "drifting" in data["drift_state"]


def test_entity_404_for_unknown(client: TestClient) -> None:
    resp = client.get("/api/entities/usr_nobody")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# GET /api/metrics
# --------------------------------------------------------------------------- #


def test_metrics_returns_200_without_eval_data(client: TestClient) -> None:
    resp = client.get("/api/metrics")
    assert resp.status_code == 200


def test_metrics_has_required_fields(client: TestClient) -> None:
    data = client.get("/api/metrics").json()
    for key in (
        "pr_auc", "roc_auc", "budget_curve", "per_attack_recall",
        "confusion_matrix", "pr_curve", "fp_rate_confounders",
        "fp_rate_insider_drift", "mttd", "cold_start", "post_drift",
        "latency_ms", "ablation",
    ):
        assert key in data, f"Missing field: {key}"

    assert "p50" in data["latency_ms"]
    assert "labels" in data["confusion_matrix"]
    assert "matrix" in data["confusion_matrix"]


# --------------------------------------------------------------------------- #
# POST /api/stream/control
# --------------------------------------------------------------------------- #


def test_stream_control_start(client: TestClient) -> None:
    resp = client.post("/api/stream/control", json={"action": "start"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["state"] in ("running", "paused", "stopped")


def test_stream_control_pause(client: TestClient) -> None:
    client.post("/api/stream/control", json={"action": "start"})
    resp = client.post("/api/stream/control", json={"action": "pause"})
    assert resp.status_code == 200
    assert resp.json()["state"] == "paused"


def test_stream_control_reset_returns_stopped(client: TestClient) -> None:
    client.post("/api/stream/control", json={"action": "start"})
    resp = client.post("/api/stream/control", json={"action": "reset"})
    assert resp.status_code == 200
    assert resp.json()["state"] == "stopped"


def test_stream_control_with_speed(client: TestClient) -> None:
    resp = client.post("/api/stream/control", json={"action": "start", "speed": 100.0})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
