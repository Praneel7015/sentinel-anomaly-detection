"""Benign confounder injectors.

Each one generates events that look anomalous to a naive rule-based detector
but are entirely legitimate business activities. All are labelled
``benign_confounder`` with ``is_anomaly = False``.

Also includes the ``insider_drift`` edge case: a legitimate entity slowly
expanding its resource footprint, labelled separately so FP-tuning can be
measured distinctly.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from sentinel.datagen.catalog import (
    COHORTS,
    GEO_SITES,
    OS_POOL,
    device_fingerprint,
    ip_for_site,
    random_mac,
)
from sentinel.datagen.attacks import _haversine_km
from sentinel.datagen.profiles import EntityProfile
from sentinel.schema import LABEL_BENIGN_CONFOUNDER

__all__ = [
    "inject_legit_travel",
    "inject_new_device_enrollment",
    "inject_password_rotation",
    "inject_vacation_return",
    "inject_maintenance_burst",
    "inject_insider_drift",
]


def _ep_id() -> str:
    return uuid.uuid4().hex[:16]


def _make_row(
    profile: EntityProfile,
    ts: datetime,
    site: Any,
    source_ip: str,
    resource: tuple[str, str],
    auth_method: str,
    auth_result: str,
    duration: float,
    commands: list[str],
    device: dict[str, str],
    raw_bytes: int,
    split: str,
    episode_id: str,
    label: str,
    is_anomaly: bool,
    confounder_type: str | None,
    attack_stage: str | None,
    counter: int,
) -> dict[str, Any]:
    seed_bytes = f"{profile.entity_id}:conf:{episode_id}:{counter}".encode()
    eid = hashlib.sha256(seed_bytes).hexdigest()[:16]
    return {
        "event_id": eid,
        "episode_id": episode_id,
        "entity_id": profile.entity_id,
        "entity_type": profile.entity_type,
        "cohort": profile.cohort,
        "timestamp": ts.replace(tzinfo=UTC),
        "source_ip": source_ip,
        "geo_country": site.country,
        "geo_city": site.city,
        "geo_lat": site.lat,
        "geo_lon": site.lon,
        "resource_accessed": resource[0],
        "resource_type": resource[1],
        "auth_method": auth_method,
        "auth_result": auth_result,
        "session_duration_s": float(max(0.1, duration)),
        "command_sequence": commands,
        "device_os": device["os"],
        "device_os_version": device["version"],
        "device_mac": device["mac"],
        "device_protocol": device["protocol"],
        "device_fingerprint": device["fp"],
        "bytes_transferred": max(0, int(raw_bytes)),
        "split": split,
        "_label": label,
        "_is_anomaly": is_anomaly,
        "_confounder_type": confounder_type,
        "_attack_stage": attack_stage,
    }


# ---------------------------------------------------------------------------
# 1. Legitimate Travel  (trips naive impossible-travel rule)
# ---------------------------------------------------------------------------

def inject_legit_travel(
    profile: EntityProfile,
    start_time: datetime,
    split: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Sales/exec travel at plausible aircraft velocity.

    Gap is >= 6h (typical flight), distance is plausible.
    """
    ep = _ep_id()
    cohort = COHORTS[profile.cohort]
    home = GEO_SITES[profile.home_site_idx]

    # Pick a destination that a flight could reach in 6-18h (roughly 3000-15000 km)
    travel_time_h = float(rng.uniform(6.0, 18.0))
    candidate_sites = [
        s for s in GEO_SITES
        if 2500 < _haversine_km(home.lat, home.lon, s.lat, s.lon) < 16000
    ]
    if not candidate_sites:
        candidate_sites = GEO_SITES

    dest = candidate_sites[int(rng.integers(0, len(candidate_sites)))]

    # Login from home before departure
    t1 = start_time
    ip1 = ip_for_site(home, rng)
    resource = profile.sample_resource(rng)
    dev = profile.device_fingerprints[0]
    auth_method = str(rng.choice(cohort.auth_methods))

    row1 = _make_row(
        profile=profile, ts=t1, site=home, source_ip=ip1,
        resource=resource, auth_method=auth_method, auth_result="success",
        duration=float(rng.lognormal(profile.session_duration_mu, 0.5)),
        commands=[], device=dev,
        raw_bytes=int(rng.lognormal(profile.bytes_mu, profile.bytes_sigma)),
        split=split, episode_id=ep, label=LABEL_BENIGN_CONFOUNDER,
        is_anomaly=False, confounder_type="legit_travel", attack_stage=None,
        counter=0,
    )

    # Login from destination after flight
    t2 = t1 + timedelta(hours=travel_time_h + float(rng.uniform(0.5, 2.0)))
    ip2 = ip_for_site(dest, rng)
    row2 = _make_row(
        profile=profile, ts=t2, site=dest, source_ip=ip2,
        resource=profile.sample_resource(rng), auth_method=auth_method, auth_result="success",
        duration=float(rng.lognormal(profile.session_duration_mu, 0.5)),
        commands=[], device=dev,
        raw_bytes=int(rng.lognormal(profile.bytes_mu, profile.bytes_sigma)),
        split=split, episode_id=ep, label=LABEL_BENIGN_CONFOUNDER,
        is_anomaly=False, confounder_type="legit_travel", attack_stage=None,
        counter=1,
    )

    return [row1, row2]


# ---------------------------------------------------------------------------
# 2. New Device Enrollment  (trips naive fingerprint-change rule)
# ---------------------------------------------------------------------------

def inject_new_device_enrollment(
    profile: EntityProfile,
    start_time: datetime,
    split: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """New device registered during work hours from the home geo - fully legitimate."""
    ep = _ep_id()
    cohort = COHORTS[profile.cohort]
    home = GEO_SITES[profile.home_site_idx]

    # Work hours: 9-17
    hour = int(rng.integers(9, 17))
    t = start_time.replace(hour=hour, minute=int(rng.integers(0, 60)))

    # Brand-new device fingerprint
    new_os, new_ver = OS_POOL[int(rng.integers(0, len(OS_POOL)))]
    new_mac = random_mac(rng)
    new_proto = str(rng.choice(cohort.protocols))
    new_fp = device_fingerprint(new_os, new_ver, new_mac, new_proto)
    new_dev = {"os": new_os, "version": new_ver, "mac": new_mac,
               "protocol": new_proto, "fp": new_fp}

    ip = ip_for_site(home, rng)
    resource = profile.sample_resource(rng)
    auth_method = str(rng.choice(cohort.auth_methods))
    rows = []

    # 2-4 events with the new device - normal resources, normal bytes, work hours
    n = int(rng.integers(2, 5))
    for i in range(n):
        t2 = t + timedelta(minutes=float(rng.uniform(5, 45)))
        rows.append(_make_row(
            profile=profile, ts=t2, site=home, source_ip=ip,
            resource=profile.sample_resource(rng), auth_method=auth_method, auth_result="success",
            duration=float(rng.lognormal(profile.session_duration_mu, 0.4)),
            commands=[], device=new_dev,
            raw_bytes=int(rng.lognormal(profile.bytes_mu, profile.bytes_sigma)),
            split=split, episode_id=ep, label=LABEL_BENIGN_CONFOUNDER,
            is_anomaly=False, confounder_type="new_device_enrollment", attack_stage=None,
            counter=i,
        ))
    return rows


# ---------------------------------------------------------------------------
# 3. Password Rotation  (trips naive brute-force rule)
# ---------------------------------------------------------------------------

def inject_password_rotation(
    profile: EntityProfile,
    start_time: datetime,
    split: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """A cluster of auth failures (fat-fingering the new password) then success."""
    ep = _ep_id()
    cohort = COHORTS[profile.cohort]
    home = GEO_SITES[profile.home_site_idx]
    ip = ip_for_site(home, rng)
    dev = profile.device_fingerprints[0]
    resource = profile.sample_resource(rng)
    auth_method = "password"   # password rotation always uses password

    # 3-8 failures (fat-fingers), spaced realistically (5-60s apart, human speed)
    n_failures = int(rng.integers(3, 9))
    rows = []
    t = start_time

    for i in range(n_failures):
        t = t + timedelta(seconds=float(rng.uniform(5, 60)))
        rows.append(_make_row(
            profile=profile, ts=t, site=home, source_ip=ip,
            resource=resource, auth_method=auth_method, auth_result="failure",
            duration=float(rng.uniform(0.5, 3.0)),
            commands=[], device=dev, raw_bytes=0,
            split=split, episode_id=ep, label=LABEL_BENIGN_CONFOUNDER,
            is_anomaly=False, confounder_type="password_rotation", attack_stage=None,
            counter=i,
        ))

    # Final success
    t = t + timedelta(seconds=float(rng.uniform(5, 30)))
    rows.append(_make_row(
        profile=profile, ts=t, site=home, source_ip=ip,
        resource=resource, auth_method=auth_method, auth_result="success",
        duration=float(rng.lognormal(profile.session_duration_mu, 0.5)),
        commands=[], device=dev,
        raw_bytes=int(rng.lognormal(profile.bytes_mu, profile.bytes_sigma)),
        split=split, episode_id=ep, label=LABEL_BENIGN_CONFOUNDER,
        is_anomaly=False, confounder_type="password_rotation", attack_stage=None,
        counter=n_failures,
    ))

    return rows


# ---------------------------------------------------------------------------
# 4. Vacation Return  (trips naive inactivity-burst rule)
# ---------------------------------------------------------------------------

def inject_vacation_return(
    profile: EntityProfile,
    return_day: datetime,
    split: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """After a long gap, a burst of catch-up activity on return day."""
    ep = _ep_id()
    cohort = COHORTS[profile.cohort]
    home = GEO_SITES[profile.home_site_idx]
    ip = ip_for_site(home, rng)
    dev = profile.device_fingerprints[0]
    auth_method = str(rng.choice(cohort.auth_methods))

    # Heavy day: 2-4x normal events
    n_events = int(rng.integers(20, 50))
    rows = []

    # Spread throughout the work day
    for i in range(n_events):
        hour = float(rng.uniform(8.0, 18.0))
        t = return_day.replace(tzinfo=UTC) + timedelta(hours=hour, minutes=float(rng.uniform(0, 60)))
        resource = profile.sample_resource(rng)
        rows.append(_make_row(
            profile=profile, ts=t, site=home, source_ip=ip,
            resource=resource, auth_method=auth_method, auth_result="success",
            duration=float(rng.lognormal(profile.session_duration_mu, 0.6)),
            commands=[], device=dev,
            raw_bytes=int(rng.lognormal(profile.bytes_mu, profile.bytes_sigma)),
            split=split, episode_id=ep, label=LABEL_BENIGN_CONFOUNDER,
            is_anomaly=False, confounder_type="vacation_return", attack_stage=None,
            counter=i,
        ))
    return rows


# ---------------------------------------------------------------------------
# 5. Maintenance Burst  (trips naive volume-anomaly rule for service accounts)
# ---------------------------------------------------------------------------

def inject_maintenance_burst(
    profile: EntityProfile,
    start_time: datetime,
    split: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Service account hammers many resources in a maintenance window."""
    ep = _ep_id()
    cohort = COHORTS[profile.cohort]
    home = GEO_SITES[profile.home_site_idx]
    ip = ip_for_site(home, rng)
    dev = profile.device_fingerprints[0]
    auth_method = str(rng.choice(cohort.auth_methods))

    # Late night / weekend maintenance window
    hour = int(rng.choice([1, 2, 3, 4, 22, 23]))
    t_base = start_time.replace(hour=hour, minute=0)

    # Many resources in a short window (30-120 min)
    n_events = int(rng.integers(30, 100))
    window_s = float(rng.uniform(1800, 7200))
    rows = []

    all_resources = cohort.resources
    for i in range(n_events):
        t = t_base.replace(tzinfo=UTC) + timedelta(seconds=float(rng.uniform(0, window_s)))
        resource = all_resources[int(rng.integers(0, len(all_resources)))]
        rows.append(_make_row(
            profile=profile, ts=t, site=home, source_ip=ip,
            resource=resource, auth_method=auth_method, auth_result="success",
            duration=float(rng.lognormal(5.0, 0.5)),
            commands=profile.sample_commands(rng, 3), device=dev,
            raw_bytes=int(rng.lognormal(profile.bytes_mu + 1.0, profile.bytes_sigma)),
            split=split, episode_id=ep, label=LABEL_BENIGN_CONFOUNDER,
            is_anomaly=False, confounder_type="maintenance_burst", attack_stage=None,
            counter=i,
        ))
    return rows


# ---------------------------------------------------------------------------
# 6. Insider Drift  (edge case, NOT an anomaly)
# ---------------------------------------------------------------------------

def inject_insider_drift(
    profile: EntityProfile,
    start_day: datetime,
    ramp_days: int,
    new_resource_fraction: float,
    split: str,
    rng: np.random.Generator,
    all_resource_pool: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Legitimate entity slowly expanding resource footprint over ramp_days.

    Simulates an employee whose role has genuinely changed (new project,
    promotion). is_anomaly = False.
    """
    ep = _ep_id()
    cohort = COHORTS[profile.cohort]
    home = GEO_SITES[profile.home_site_idx]
    ip = ip_for_site(home, rng)
    dev = profile.device_fingerprints[0]
    auth_method = str(rng.choice(cohort.auth_methods))

    rows = []
    for day_offset in range(ramp_days):
        day = start_day + timedelta(days=day_offset)
        progress = day_offset / max(1, ramp_days - 1)   # 0 -> 1

        # Gradually mix in new resources
        n_events = int(rng.integers(5, 15))
        for i in range(n_events):
            hour = float(rng.uniform(8.0, 18.0))
            t = day.replace(tzinfo=UTC) + timedelta(hours=hour)

            if rng.random() < new_resource_fraction * progress and all_resource_pool:
                resource = all_resource_pool[int(rng.integers(0, len(all_resource_pool)))]
            else:
                resource = profile.sample_resource(rng)

            rows.append(_make_row(
                profile=profile, ts=t, site=home, source_ip=ip,
                resource=resource, auth_method=auth_method, auth_result="success",
                duration=float(rng.lognormal(profile.session_duration_mu, 0.5)),
                commands=[], device=dev,
                raw_bytes=int(rng.lognormal(profile.bytes_mu, profile.bytes_sigma)),
                split=split, episode_id=ep, label="insider_drift",
                is_anomaly=False, confounder_type=None, attack_stage=None,
                counter=day_offset * 20 + i,
            ))

    return rows
