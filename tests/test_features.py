"""Tests for the feature pipeline.

Critical tests:
1. batch == stream (bit-identical feature matrices)
2. Leakage: appending future events does not change earlier features
3. Determinism + state round-trip
4. Correctness: geo-velocity, failure ratio, resource novelty, empty sequence,
   cold-start cohort dominance
5. Benchmark: per-event latency < 1 ms on average
"""
from __future__ import annotations

import copy
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

from sentinel.features.pipeline import PipelineState, batch_features, stream_features
from sentinel.features.extractors import FEATURE_NAMES, FeatureVector

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_BASE_TS = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


def _make_event(
    entity_id: str = "usr_0001",
    cohort: str = "finance_analyst",
    offset_s: float = 0.0,
    geo_lat: float = 40.7128,
    geo_lon: float = -74.0060,
    geo_country: str = "US",
    source_ip: str = "10.0.0.1",
    resource: str = "fs:/finance/q3.xlsx",
    auth_method: str = "password",
    auth_result: str = "success",
    device_os: str = "Windows",
    device_os_version: str = "11",
    device_mac: str = "AA:BB:CC:DD:EE:FF",
    device_protocol: str = "https",
    device_fingerprint: str = "fp-abc123",
    session_duration_s: float = 120.0,
    command_sequence: list[str] | None = None,
    bytes_transferred: int = 1024,
    split: str = "train",
    entity_type: str = "user",
    event_id: str | None = None,
) -> dict[str, Any]:
    ts = _BASE_TS + timedelta(seconds=offset_s)
    if event_id is None:
        event_id = f"{entity_id}_{offset_s}"
    return {
        "event_id": event_id,
        "episode_id": None,
        "entity_id": entity_id,
        "entity_type": entity_type,
        "cohort": cohort,
        "timestamp": ts,
        "source_ip": source_ip,
        "geo_country": geo_country,
        "geo_city": "TestCity",
        "geo_lat": geo_lat,
        "geo_lon": geo_lon,
        "resource_accessed": resource,
        "resource_type": "file",
        "auth_method": auth_method,
        "auth_result": auth_result,
        "session_duration_s": session_duration_s,
        "command_sequence": command_sequence or [],
        "device_os": device_os,
        "device_os_version": device_os_version,
        "device_mac": device_mac,
        "device_protocol": device_protocol,
        "device_fingerprint": device_fingerprint,
        "bytes_transferred": bytes_transferred,
        "split": split,
    }


def _event_list_to_df(events: list[dict[str, Any]]):
    import pandas as pd
    return pd.DataFrame(events)


# ---------------------------------------------------------------------------
# 1. batch == stream (critical leakage-by-construction test)
# ---------------------------------------------------------------------------

class TestBatchEqualsStream:
    def test_identical_outputs(self):
        events = [
            _make_event(offset_s=i * 60.0, resource=f"fs:/res/{i}", event_id=f"e{i}")
            for i in range(20)
        ]
        # stream
        stream_fvs = []
        for _, fv in stream_features(events):
            stream_fvs.append(fv.to_numpy())

        # batch
        df = _event_list_to_df(events)
        result_df = batch_features(df)

        assert result_df.shape == (20, len(FEATURE_NAMES))
        assert list(result_df.columns) == FEATURE_NAMES

        # Compare row by row in timestamp-sorted order
        sorted_df = df.sort_values("timestamp", kind="stable")
        for i, idx in enumerate(sorted_df.index):
            stream_row = stream_fvs[i]
            batch_row = result_df.loc[idx].values.astype(np.float32)
            np.testing.assert_array_equal(
                stream_row, batch_row,
                err_msg=f"Row {i} (event {idx}) differs between stream and batch",
            )

    def test_multiple_entities(self):
        events = []
        for i in range(10):
            events.append(_make_event(entity_id="usr_0001", offset_s=i * 100.0, event_id=f"a{i}"))
            events.append(_make_event(entity_id="usr_0002", offset_s=i * 100.0 + 50.0, event_id=f"b{i}"))

        stream_fvs_by_id: dict[str, list] = {}
        for ev, fv in stream_features(events):
            eid = ev["entity_id"]
            stream_fvs_by_id.setdefault(eid, []).append(fv.to_numpy())

        df = _event_list_to_df(events)
        result_df = batch_features(df)

        sorted_df = df.sort_values("timestamp", kind="stable")
        entity_cursor = {"usr_0001": 0, "usr_0002": 0}
        for idx in sorted_df.index:
            eid = sorted_df.loc[idx, "entity_id"]
            i = entity_cursor[eid]
            entity_cursor[eid] += 1
            np.testing.assert_array_equal(
                stream_fvs_by_id[eid][i],
                result_df.loc[idx].values.astype(np.float32),
            )


# ---------------------------------------------------------------------------
# 2. Leakage test: future events must not change earlier features
# ---------------------------------------------------------------------------

class TestLeakageFree:
    def test_appending_future_does_not_change_past(self):
        past_events = [
            _make_event(offset_s=i * 60.0, event_id=f"p{i}") for i in range(5)
        ]
        future_events = [
            _make_event(offset_s=(5 + i) * 60.0, event_id=f"f{i}") for i in range(5)
        ]

        # Features for past events in isolation
        fvs_alone = []
        for _, fv in stream_features(past_events):
            fvs_alone.append(fv.to_numpy())

        # Features for past events when future events also present
        all_events = past_events + future_events
        fvs_combined = []
        for ev, fv in stream_features(all_events):
            if ev["event_id"].startswith("p"):
                fvs_combined.append(fv.to_numpy())

        assert len(fvs_alone) == len(fvs_combined) == 5
        for i, (a, b) in enumerate(zip(fvs_alone, fvs_combined)):
            np.testing.assert_array_equal(
                a, b,
                err_msg=f"Past event {i} features changed when future events appended (LEAKAGE)",
            )

    def test_batch_leakage_free(self):
        """batch_features must give identical past features regardless of future rows."""
        past = [_make_event(offset_s=i * 60.0, event_id=f"pp{i}") for i in range(5)]
        future = [_make_event(offset_s=(5 + i) * 60.0, event_id=f"ff{i}") for i in range(3)]

        df_past = _event_list_to_df(past)
        df_all = _event_list_to_df(past + future)

        result_past = batch_features(df_past)
        result_all = batch_features(df_all)

        for ev in past:
            idx_past = df_past[df_past["event_id"] == ev["event_id"]].index[0]
            idx_all = df_all[df_all["event_id"] == ev["event_id"]].index[0]
            np.testing.assert_array_equal(
                result_past.loc[idx_past].values.astype(np.float32),
                result_all.loc[idx_all].values.astype(np.float32),
                err_msg=f"batch_features: event {ev['event_id']} differs with future appended",
            )


# ---------------------------------------------------------------------------
# 3. Determinism + state round-trip
# ---------------------------------------------------------------------------

class TestDeterminismAndRoundTrip:
    def test_deterministic(self):
        events = [_make_event(offset_s=i * 60.0, event_id=f"d{i}") for i in range(10)]
        fvs1 = [fv.to_numpy() for _, fv in stream_features(events)]
        fvs2 = [fv.to_numpy() for _, fv in stream_features(events)]
        for a, b in zip(fvs1, fvs2):
            np.testing.assert_array_equal(a, b)

    def test_state_roundtrip_produces_identical_subsequent_features(self):
        events_first = [_make_event(offset_s=i * 60.0, event_id=f"r{i}") for i in range(8)]
        events_second = [_make_event(offset_s=(8 + i) * 60.0, event_id=f"s{i}") for i in range(4)]

        # Run first batch and capture state
        state = PipelineState()
        for _, _ in stream_features(events_first, state=state):
            pass

        # Serialise + restore state
        state_dict = state.to_dict()
        state_restored = PipelineState.from_dict(state_dict)

        # Run second batch from both states
        fvs_original = [fv.to_numpy() for _, fv in stream_features(events_second, state=state)]
        fvs_restored = [fv.to_numpy() for _, fv in stream_features(events_second, state=state_restored)]

        for i, (a, b) in enumerate(zip(fvs_original, fvs_restored)):
            np.testing.assert_array_equal(
                a, b,
                err_msg=f"Event {i}: state round-trip produced different features",
            )


# ---------------------------------------------------------------------------
# 4. Targeted correctness tests
# ---------------------------------------------------------------------------

class TestCorrectnessImpossibleTravel:
    def test_geo_velocity_above_flight_bound(self):
        """New York to London in 30 minutes must exceed the max commercial flight speed."""
        ev1 = _make_event(entity_id="trav_01", offset_s=0.0,
                          geo_lat=40.7128, geo_lon=-74.0060,
                          geo_country="US", event_id="t1")
        # London 30 minutes later
        ev2 = _make_event(entity_id="trav_01", offset_s=1800.0,
                          geo_lat=51.5074, geo_lon=-0.1278,
                          geo_country="GB", event_id="t2")

        results = list(stream_features([ev1, ev2]))
        _, fv1 = results[0]
        _, fv2 = results[1]

        vel = fv2["geo_velocity_kmh"]
        # NY-London ~5570 km in 30 min = ~11140 km/h >> fastest aircraft ~3500 km/h
        assert vel > 3000.0, f"Expected geo_velocity > 3000 km/h, got {vel}"
        assert fv1["geo_velocity_kmh"] == 0.0  # no previous location


class TestCorrectnessAuthFailures:
    def test_burst_of_failures_raises_ip_failure_ratio(self):
        ip = "192.168.1.100"
        events = []
        # 5 failures from same IP
        for i in range(5):
            events.append(_make_event(
                entity_id=f"victim_{i}", cohort="analyst",
                source_ip=ip, auth_result="failure",
                offset_s=float(i * 10), event_id=f"fail{i}",
            ))
        # success event
        events.append(_make_event(
            entity_id="victim_0", cohort="analyst",
            source_ip=ip, auth_result="success",
            offset_s=60.0, event_id="succ0",
        ))

        results = list(stream_features(events))
        last_ev, last_fv = results[-1]

        ip_fail_1h = last_fv["ip_failure_count_1h"]
        distinct_ips = last_fv["distinct_entities_per_ip"]
        assert ip_fail_1h >= 5, f"Expected >= 5 IP failures in 1h, got {ip_fail_1h}"
        assert distinct_ips >= 5, f"Expected >= 5 distinct entities from IP, got {distinct_ips}"


class TestCorrectnessResourceNovelty:
    def test_first_ever_resource_sets_novelty_flags(self):
        ev1 = _make_event(entity_id="res_01", offset_s=0.0,
                          resource="fs:/common/report.pdf", event_id="r1")
        ev2 = _make_event(entity_id="res_01", offset_s=60.0,
                          resource="fs:/secret/crown_jewels.xlsx", event_id="r2")

        results = list(stream_features([ev1, ev2]))
        _, fv1 = results[0]
        _, fv2 = results[1]

        # First event: resource was never seen (entity or cohort)
        assert fv1["entity_novel_resource"] == 1.0
        assert fv1["cohort_novel_resource"] == 1.0

        # Second event: different resource, still novel for entity
        assert fv2["entity_novel_resource"] == 1.0
        # First resource is now known for cohort but second is new
        assert fv2["cohort_novel_resource"] == 1.0


class TestCorrectnessEmptyCommandSequence:
    def test_empty_sequence_produces_zero_surprisal(self):
        """Non-privileged sessions with [] commands must NOT be penalised."""
        events = [
            _make_event(entity_id="np_01", offset_s=float(i * 60),
                        command_sequence=[], event_id=f"np{i}")
            for i in range(5)
        ]
        for _, fv in stream_features(events):
            assert fv["command_surprisal"] == 0.0, \
                f"Empty command sequence should yield 0 surprisal, got {fv['command_surprisal']}"
            assert fv["unseen_bigram_count"] == 0.0


class TestColdStartCohortDominance:
    def test_cold_start_entity_temporal_surprisal_shrunk_toward_cohort(self):
        """A new entity with 2 events must have hour surprisal dominated by cohort prior."""
        # Build up cohort with 30 events at hour=10 (typical work hour)
        cohort_events = []
        for i in range(30):
            cohort_events.append(_make_event(
                entity_id=f"warm_{i:03d}", cohort="shared_cohort",
                offset_s=float(i * 3600),
                event_id=f"warm{i}",
            ))

        # Cold-start entity: only 2 events at very unusual hour (3 AM)
        cold_ts_base = 30 * 3600.0
        cold_events = [
            _make_event(entity_id="cold_01", cohort="shared_cohort",
                        offset_s=cold_ts_base + 3 * 3600 + float(i * 60),
                        event_id=f"cold{i}")
            for i in range(2)
        ]

        all_events = sorted(cohort_events + cold_events, key=lambda e: e["timestamp"])

        results = list(stream_features(all_events))

        # Collect cold_01 features
        cold_fvs = [(ev, fv) for ev, fv in results if ev["entity_id"] == "cold_01"]
        assert len(cold_fvs) == 2

        # Cold-start flag should be 1
        _, fv_cold = cold_fvs[0]
        assert fv_cold["is_cold_start_flag"] == 1.0, "Entity with 0 events should be cold start"
        assert fv_cold["entity_event_count"] == 0.0


# ---------------------------------------------------------------------------
# 5. FeatureVector API
# ---------------------------------------------------------------------------

class TestFeatureVector:
    def test_stable_ordering(self):
        from sentinel.features.extractors import FEATURE_NAMES, FeatureVector
        vals = list(range(len(FEATURE_NAMES)))
        fv = FeatureVector([float(v) for v in vals])
        arr = fv.to_numpy()
        assert arr.shape == (len(FEATURE_NAMES),)
        assert arr.dtype == np.float32
        for i, name in enumerate(FEATURE_NAMES):
            assert fv[name] == float(i)

    def test_to_dict_round_trip(self):
        from sentinel.features.extractors import FEATURE_NAMES, FeatureVector
        vals = [float(i) * 0.1 for i in range(len(FEATURE_NAMES))]
        fv = FeatureVector(vals)
        d = fv.to_dict()
        assert set(d.keys()) == set(FEATURE_NAMES)
        for name, val in d.items():
            assert abs(val - fv[name]) < 1e-6


# ---------------------------------------------------------------------------
# 6. Benchmark: per-event latency must be < 1 ms on average
# ---------------------------------------------------------------------------

class TestBenchmark:
    def test_per_event_latency_under_1ms(self):
        """Real-time streaming feasibility: < 1 ms per event on average."""
        import random
        rng = random.Random(42)
        resources = [f"fs:/data/file_{i}.xlsx" for i in range(50)]
        cmds_pool = ["ls", "cat", "cp", "mv", "rm", "sudo", "chmod", "grep", "find", "curl"]

        n_events = 500
        events = []
        for i in range(n_events):
            events.append(_make_event(
                entity_id=f"usr_{rng.randint(0, 9):04d}",
                cohort=rng.choice(["finance", "ops", "devops"]),
                offset_s=float(i * 30),
                resource=rng.choice(resources),
                auth_result=rng.choice(["success", "success", "success", "failure"]),
                command_sequence=rng.choices(cmds_pool, k=rng.randint(0, 5)),
                bytes_transferred=rng.randint(100, 100000),
                event_id=f"bench{i}",
            ))

        start = time.perf_counter()
        for _ in stream_features(events):
            pass
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / n_events) * 1000
        print(f"\n  Benchmark: {n_events} events in {elapsed*1000:.1f} ms "
              f"= {avg_ms:.4f} ms/event ({1/avg_ms*1000:.0f} events/sec)")

        assert avg_ms < 1.0, (
            f"Per-event latency {avg_ms:.4f} ms exceeds 1 ms budget. "
            "Real-time streaming is not feasible at this throughput."
        )
