"""Baseline (benign) event sampler - vectorised for speed.

Generates realistic-looking normal access events from an EntityProfile.
The bulk path generates all N events for one entity at once using numpy
arrays, then assembles a list of dicts only at the end.

Noise that prevents trivial detection:
- occasional legitimate resource novelty (rare_resource_fraction)
- small baseline auth failure rate per entity
- off-hours activity fraction
- bytes sampled from a personal lognormal
"""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from sentinel.datagen.catalog import (
    COHORTS,
    GEO_SITES,
    device_fingerprint,
    ip_for_site,
)
from sentinel.datagen.profiles import EntityProfile
from sentinel.schema import LABEL_NORMAL

__all__ = ["sample_normal_events"]


def _event_id(seed_bytes: bytes) -> str:
    return hashlib.sha256(seed_bytes).hexdigest()[:16]


def sample_normal_events(
    profile: EntityProfile,
    day: datetime,
    n_events: int,
    split: str,
    rng: np.random.Generator,
    noise_off_hours: float = 0.05,
    noise_rare_resource: float = 0.02,
    noise_auth_failure: float = 0.03,
    episode_id: str | None = None,
    apply_drift: bool = False,
    drift_magnitude: float = 0.0,
    global_event_counter: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Generate n_events benign events for one entity on one day (vectorised)."""
    if global_event_counter is None:
        global_event_counter = [0]

    cohort = COHORTS[profile.cohort]
    n_resources = len(cohort.resources)
    n_auth = len(cohort.auth_methods)
    n_fps = len(profile.device_fingerprints)

    # ---- resource weights ----
    if apply_drift and drift_magnitude > 0:
        res_weights = np.array(profile.resource_weights, dtype=np.float64)
        for idx in profile.rare_resource_idxs:
            res_weights[idx] += drift_magnitude * 0.3
        res_weights /= res_weights.sum()
    else:
        res_weights = np.array(profile.resource_weights, dtype=np.float64)
        if res_weights.sum() <= 0:
            res_weights = np.ones(n_resources) / n_resources
        else:
            res_weights /= res_weights.sum()

    # ---- timestamps ----
    off_mask = rng.random(n_events) < noise_off_hours
    hours = np.empty(n_events, dtype=np.float64)
    # Normal hours: sample from circadian PMF
    circ = np.array(profile.circadian_weights, dtype=np.float64)
    circ /= circ.sum()
    n_normal_h = int((~off_mask).sum())
    n_off_h = n_events - n_normal_h
    if n_normal_h > 0:
        base_hours = rng.choice(24, size=n_normal_h, p=circ).astype(np.float64)
        base_hours += rng.uniform(0, 1, size=n_normal_h)
        hours[~off_mask] = base_hours
    if n_off_h > 0:
        hours[off_mask] = (rng.uniform(22, 30, size=n_off_h) % 24)

    seconds_offset = (hours * 3600).astype(np.int64)

    # ---- geo / site selection ----
    # 5% chance secondary site
    use_secondary = (rng.random(n_events) < 0.05) & (len(profile.secondary_site_idxs) > 0)
    n_sec = int(use_secondary.sum())
    site_idxs = np.full(n_events, profile.home_site_idx, dtype=np.int32)
    if n_sec > 0:
        sec_arr = np.array(profile.secondary_site_idxs, dtype=np.int32)
        site_idxs[use_secondary] = rng.choice(sec_arr, size=n_sec)

    # ---- resources ----
    use_rare = rng.random(n_events) < noise_rare_resource
    n_rare = int(use_rare.sum())
    n_norm_r = n_events - n_rare
    res_idxs = np.empty(n_events, dtype=np.int32)
    if n_norm_r > 0:
        res_idxs[~use_rare] = rng.choice(n_resources, size=n_norm_r, p=res_weights)
    if n_rare > 0 and profile.rare_resource_idxs:
        rare_arr = np.array(profile.rare_resource_idxs, dtype=np.int32)
        res_idxs[use_rare] = rng.choice(rare_arr, size=n_rare)
    elif n_rare > 0:
        res_idxs[use_rare] = rng.integers(0, n_resources, size=n_rare)

    # ---- auth ----
    auth_weights = np.array(profile.auth_method_weights, dtype=np.float64)
    auth_weights /= auth_weights.sum()
    auth_method_idxs = rng.choice(n_auth, size=n_events, p=auth_weights)

    eff_fail_rate = min(0.99, profile.auth_failure_rate + noise_auth_failure)
    auth_failed = rng.random(n_events) < eff_fail_rate

    # ---- duration ----
    durations = np.exp(rng.normal(
        profile.session_duration_mu + (0.5 * drift_magnitude if apply_drift else 0),
        profile.session_duration_sigma,
        size=n_events,
    ))
    durations = np.maximum(1.0, durations)

    # ---- bytes ----
    bytes_mu_eff = profile.bytes_mu + (0.5 * drift_magnitude if apply_drift else 0)
    raw_bytes = np.exp(rng.normal(bytes_mu_eff, profile.bytes_sigma, size=n_events)).astype(np.int64)
    raw_bytes = np.maximum(0, raw_bytes)
    raw_bytes[auth_failed] = rng.integers(0, 512, size=int(auth_failed.sum()))

    # ---- device ----
    fp_idxs = rng.integers(0, n_fps, size=n_events)

    # ---- assemble rows ----
    rows: list[dict[str, Any]] = []
    day_ts_sec = int(day.replace(tzinfo=UTC).timestamp())

    for i in range(n_events):
        global_event_counter[0] += 1
        seed_bytes = f"{profile.entity_id}:{global_event_counter[0]}".encode()
        eid = _event_id(seed_bytes)

        ts_epoch = day_ts_sec + int(seconds_offset[i])
        ts = datetime.fromtimestamp(ts_epoch, tz=UTC)

        site = GEO_SITES[site_idxs[i]]
        # IP: generate a simple stable-looking IP from the site subnet
        site_net = site.subnet  # e.g. "10.10.0.0/16"
        base = site_net.split(".")[0:2]
        ip_oct3 = int(rng.integers(1, 255))
        ip_oct4 = int(rng.integers(1, 255))
        source_ip = f"{base[0]}.{base[1]}.{ip_oct3}.{ip_oct4}"

        res = cohort.resources[res_idxs[i]]
        dev = profile.device_fingerprints[fp_idxs[i]]
        auth_method = cohort.auth_methods[auth_method_idxs[i]]

        rows.append({
            "event_id": eid,
            "episode_id": episode_id,
            "entity_id": profile.entity_id,
            "entity_type": profile.entity_type,
            "cohort": profile.cohort,
            "timestamp": ts,
            "source_ip": source_ip,
            "geo_country": site.country,
            "geo_city": site.city,
            "geo_lat": site.lat,
            "geo_lon": site.lon,
            "resource_accessed": res[0],
            "resource_type": res[1],
            "auth_method": auth_method,
            "auth_result": "failure" if auth_failed[i] else "success",
            "session_duration_s": float(durations[i]),
            "command_sequence": [],
            "device_os": dev["os"],
            "device_os_version": dev["version"],
            "device_mac": dev["mac"],
            "device_protocol": dev["protocol"],
            "device_fingerprint": dev["fp"],
            "bytes_transferred": int(raw_bytes[i]),
            "split": split,
            "_label": LABEL_NORMAL,
            "_is_anomaly": False,
            "_confounder_type": None,
            "_attack_stage": None,
        })

    return rows
