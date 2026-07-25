"""Six attack injectors.  Each returns a list of row dicts (one per event)
sharing an ``episode_id`` and tagged with ``_label``, ``_is_anomaly``,
``_attack_stage``, and ``_confounder_type = None``.

Internal label columns use the ``_`` prefix so they can be split cleanly
into a separate labels frame at write time without touching events.
"""
from __future__ import annotations

import hashlib
import math
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from sentinel.datagen.catalog import (
    ATTACKER_COMMANDS,
    ATTACKER_CHAIN_STAGES,
    COHORTS,
    GEO_SITES,
    OS_POOL,
    device_fingerprint,
    ip_for_site,
    random_mac,
)
from sentinel.datagen.profiles import EntityProfile
from sentinel.schema import ATTACK_TYPES

__all__ = [
    "inject_brute_force",
    "inject_impossible_travel",
    "inject_credential_stuffing",
    "inject_lateral_movement",
    "inject_device_spoofing",
    "inject_low_and_slow_exfil",
]

_GEO_RADIUS_KM = 6371.0   # Earth mean radius


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = _GEO_RADIUS_KM
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _ep_id() -> str:
    return uuid.uuid4().hex[:16]


def _eid(seed_bytes: bytes) -> str:
    return hashlib.sha256(seed_bytes).hexdigest()[:16]


def _base_row(
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
    attack_stage: str | None,
    counter: int,
) -> dict[str, Any]:
    seed_bytes = f"{profile.entity_id}:atk:{episode_id}:{counter}".encode()
    return {
        "event_id": _eid(seed_bytes),
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
        "session_duration_s": float(duration),
        "command_sequence": commands,
        "device_os": device["os"],
        "device_os_version": device["version"],
        "device_mac": device["mac"],
        "device_protocol": device["protocol"],
        "device_fingerprint": device["fp"],
        "bytes_transferred": max(0, int(raw_bytes)),
        "split": split,
        "_label": label,
        "_is_anomaly": True,
        "_confounder_type": None,
        "_attack_stage": attack_stage,
    }


# ---------------------------------------------------------------------------
# 1. Brute Force
# ---------------------------------------------------------------------------

def inject_brute_force(
    profile: EntityProfile,
    start_time: datetime,
    split: str,
    rng: np.random.Generator,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Burst of auth failures from one IP against one entity."""
    attempts = int(rng.integers(params["attempts_min"], params["attempts_max"] + 1))
    interval_s_min = params["interval_s_min"]
    interval_s_max = params["interval_s_max"]
    success_prob = params["success_probability"]

    # Throttled vs loud: ~30% are throttled (slower)
    is_throttled = rng.random() < 0.30
    if is_throttled:
        interval_s_min = max(interval_s_min, 20)
        interval_s_max = max(interval_s_max, 90)
        attempts = min(attempts, 25)

    ep = _ep_id()
    rows = []
    cohort = COHORTS[profile.cohort]

    # Attacker uses a single external IP (not in any known subnet)
    attacker_ip = f"185.{rng.integers(1,255)}.{rng.integers(1,255)}.{rng.integers(1,255)}"
    # External site - pick a site far from home
    home = GEO_SITES[profile.home_site_idx]
    far_sites = [s for s in GEO_SITES if _haversine_km(home.lat, home.lon, s.lat, s.lon) > 3000]
    if not far_sites:
        far_sites = GEO_SITES
    ext_site = far_sites[int(rng.integers(0, len(far_sites)))]

    auth_method = str(rng.choice(cohort.auth_methods))
    resource = cohort.resources[int(rng.integers(0, len(cohort.resources)))]
    dev = profile.device_fingerprints[0]

    t = start_time
    for i in range(attempts):
        interval = float(rng.uniform(interval_s_min, interval_s_max))
        t = t + timedelta(seconds=interval)

        is_last = (i == attempts - 1)
        if is_last and rng.random() < success_prob:
            auth_result = "success"
            stage = "initial_access"
            raw_bytes = int(rng.lognormal(profile.bytes_mu, 0.5))
        else:
            auth_result = "failure"
            stage = "recon"
            raw_bytes = int(rng.integers(0, 256))

        rows.append(_base_row(
            profile=profile, ts=t, site=ext_site, source_ip=attacker_ip,
            resource=resource, auth_method=auth_method, auth_result=auth_result,
            duration=float(rng.uniform(0.1, 2.0)),
            commands=[], device=dev, raw_bytes=raw_bytes,
            split=split, episode_id=ep, label="brute_force", attack_stage=stage,
            counter=i,
        ))
    return rows


# ---------------------------------------------------------------------------
# 2. Impossible Travel
# ---------------------------------------------------------------------------

def inject_impossible_travel(
    profile: EntityProfile,
    start_time: datetime,
    split: str,
    rng: np.random.Generator,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Two logins from geographically impossible locations."""
    min_kmph = params["min_km_per_hour"]
    gap_h_min = params["gap_hours_min"]
    gap_h_max = params["gap_hours_max"]

    ep = _ep_id()
    cohort = COHORTS[profile.cohort]
    home = GEO_SITES[profile.home_site_idx]
    resource = profile.sample_resource(rng)
    auth_method = str(rng.choice(cohort.auth_methods))

    # First login: from home
    t1 = start_time
    ip1 = ip_for_site(home, rng)
    dev = profile.device_fingerprints[0]
    dur1 = float(max(1.0, np.exp(rng.normal(profile.session_duration_mu, profile.session_duration_sigma))))

    row1 = _base_row(
        profile=profile, ts=t1, site=home, source_ip=ip1,
        resource=resource, auth_method=auth_method, auth_result="success",
        duration=dur1, commands=[], device=dev,
        raw_bytes=int(rng.lognormal(profile.bytes_mu, profile.bytes_sigma)),
        split=split, episode_id=ep, label="impossible_travel",
        attack_stage="initial_access", counter=0,
    )

    # Gap: must be short enough that implied velocity exceeds min_kmph
    gap_h = float(rng.uniform(gap_h_min, gap_h_max))

    # Choose a far site such that distance / gap_h > min_kmph
    far_sites = [
        s for s in GEO_SITES
        if _haversine_km(home.lat, home.lon, s.lat, s.lon) > min_kmph * gap_h
    ]
    if not far_sites:
        far_sites = sorted(
            GEO_SITES,
            key=lambda s: _haversine_km(home.lat, home.lon, s.lat, s.lon),
            reverse=True,
        )[:3]

    far_site = far_sites[int(rng.integers(0, len(far_sites)))]
    t2 = t1 + timedelta(hours=gap_h)
    ip2 = ip_for_site(far_site, rng)

    # Attacker device
    att_os, att_ver = OS_POOL[int(rng.integers(0, len(OS_POOL)))]
    att_mac = random_mac(rng)
    att_proto = str(rng.choice(cohort.protocols))
    att_fp = device_fingerprint(att_os, att_ver, att_mac, att_proto)
    att_dev = {"os": att_os, "version": att_ver, "mac": att_mac,
               "protocol": att_proto, "fp": att_fp}

    row2 = _base_row(
        profile=profile, ts=t2, site=far_site, source_ip=ip2,
        resource=profile.sample_resource(rng), auth_method=auth_method, auth_result="success",
        duration=float(max(1.0, np.exp(rng.normal(profile.session_duration_mu, 0.5)))),
        commands=[], device=att_dev,
        raw_bytes=int(rng.lognormal(profile.bytes_mu + 1.0, profile.bytes_sigma)),
        split=split, episode_id=ep, label="impossible_travel",
        attack_stage="action_on_objective", counter=1,
    )

    return [row1, row2]


# ---------------------------------------------------------------------------
# 3. Credential Stuffing
# ---------------------------------------------------------------------------

def inject_credential_stuffing(
    all_profiles: list[EntityProfile],
    start_time: datetime,
    split: str,
    rng: np.random.Generator,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Many accounts, few IPs, high failure rate, spread over a window."""
    accts_min = params["accounts_per_source_ip_min"]
    accts_max = params["accounts_per_source_ip_max"]
    per_acct_max = params["per_account_attempts_max"]

    ep = _ep_id()
    rows = []

    # Pick a small pool of attacker IPs (1-3)
    n_ips = int(rng.integers(1, 4))
    attacker_ips = [
        f"194.{rng.integers(1,255)}.{rng.integers(1,255)}.{rng.integers(1,255)}"
        for _ in range(n_ips)
    ]

    n_targets = int(rng.integers(accts_min, accts_max + 1))
    targets = list(rng.choice(len(all_profiles), size=min(n_targets, len(all_profiles)), replace=False))

    # Window: spread attempts over 0-6 hours
    window_s = float(rng.uniform(1800, 21600))
    success_threshold = 0.08   # ~8% of stuffed accounts yield a hit

    for i, tidx in enumerate(targets):
        prof = all_profiles[tidx]
        cohort = COHORTS[prof.cohort]
        n_attempts = int(rng.integers(1, per_acct_max + 1))
        t_base = start_time + timedelta(seconds=float(rng.uniform(0, window_s)))
        ip = attacker_ips[int(rng.integers(0, len(attacker_ips)))]
        auth_method = str(rng.choice(cohort.auth_methods))
        resource = cohort.resources[int(rng.integers(0, len(cohort.resources)))]
        dev = prof.device_fingerprints[0]

        for j in range(n_attempts):
            t = t_base + timedelta(seconds=float(rng.uniform(0, 120)))
            is_success = (j == n_attempts - 1) and (rng.random() < success_threshold)
            rows.append(_base_row(
                profile=prof, ts=t, site=GEO_SITES[0],
                source_ip=ip, resource=resource,
                auth_method=auth_method,
                auth_result="success" if is_success else "failure",
                duration=float(rng.uniform(0.1, 1.5)),
                commands=[], device=dev,
                raw_bytes=0 if not is_success else int(rng.lognormal(8.0, 1.0)),
                split=split, episode_id=ep, label="credential_stuffing",
                attack_stage="initial_access" if is_success else "recon",
                counter=i * per_acct_max + j,
            ))
    return rows


# ---------------------------------------------------------------------------
# 4. Lateral Movement
# ---------------------------------------------------------------------------

def inject_lateral_movement(
    profile: EntityProfile,
    all_profiles: list[EntityProfile],
    start_time: datetime,
    split: str,
    rng: np.random.Generator,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compromised entity touches unusual resources + moves laterally."""
    hops_min = params["hops_min"]
    hops_max = params["hops_max"]
    protocols = params["protocols"]

    ep = _ep_id()
    rows = []
    n_hops = int(rng.integers(hops_min, hops_max + 1))

    # Build attacker command sequence from the attacker chain
    attacker_cmds = list(ATTACKER_COMMANDS)
    # Walk the chain
    seq_len = min(n_hops + 2, len(attacker_cmds))
    cmd_sequence = attacker_cmds[:seq_len]

    # Stage map
    stage_seq = [ATTACKER_CHAIN_STAGES.get(c, "recon") for c in cmd_sequence]

    # Use off-hours most of the time (but not always - don't make time a giveaway)
    hour_off = rng.random() < 0.65

    # Victim device
    att_os, att_ver = OS_POOL[int(rng.integers(0, len(OS_POOL)))]
    att_mac = random_mac(rng)
    att_proto = str(rng.choice(protocols))
    att_fp = device_fingerprint(att_os, att_ver, att_mac, att_proto)
    att_dev = {"os": att_os, "version": att_ver, "mac": att_mac,
               "protocol": att_proto, "fp": att_fp}

    home = GEO_SITES[profile.home_site_idx]
    source_ip = ip_for_site(home, rng)

    # Collect resources from OTHER cohorts - things this entity never touches
    foreign_resources: list[tuple[str, str]] = []
    for other_prof in all_profiles:
        if other_prof.cohort != profile.cohort:
            other_cohort = COHORTS[other_prof.cohort]
            foreign_resources.extend(other_cohort.resources[:3])
    if not foreign_resources:
        foreign_resources = [("port:445", "port"), ("port:3389", "port"), ("api:/v1/admin", "endpoint")]
    rng.shuffle(foreign_resources)

    t = start_time
    if hour_off:
        t = t.replace(hour=int(rng.integers(23, 24)), minute=int(rng.integers(0, 60)))

    for i in range(n_hops):
        t = t + timedelta(seconds=float(rng.uniform(30, 600)))
        resource = foreign_resources[i % len(foreign_resources)]
        stage_idx = min(i, len(stage_seq) - 1)
        stage = stage_seq[stage_idx]
        cmds = cmd_sequence[stage_idx:stage_idx + 2]
        bytes_val = int(rng.lognormal(profile.bytes_mu + 1.5, profile.bytes_sigma)) if "exfil" in cmds else int(rng.lognormal(8, 1))

        rows.append(_base_row(
            profile=profile, ts=t, site=home, source_ip=source_ip,
            resource=resource, auth_method="password", auth_result="success",
            duration=float(rng.uniform(10, 300)),
            commands=cmds, device=att_dev,
            raw_bytes=bytes_val,
            split=split, episode_id=ep, label="lateral_movement",
            attack_stage=stage, counter=i,
        ))
    return rows


# ---------------------------------------------------------------------------
# 5. Device Spoofing
# ---------------------------------------------------------------------------

def inject_device_spoofing(
    profile: EntityProfile,
    start_time: datetime,
    split: str,
    rng: np.random.Generator,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Device reappears with mismatched fingerprint."""
    reuse_mac = params.get("reuse_mac", True)
    ep = _ep_id()
    rows = []

    cohort = COHORTS[profile.cohort]
    home = GEO_SITES[profile.home_site_idx]
    legitimate_dev = profile.device_fingerprints[0]

    # 1. Legitimate activity first
    t1 = start_time
    ip1 = ip_for_site(home, rng)
    resource1 = profile.sample_resource(rng)
    row1 = _base_row(
        profile=profile, ts=t1, site=home, source_ip=ip1,
        resource=resource1, auth_method=str(rng.choice(cohort.auth_methods)),
        auth_result="success", duration=float(rng.lognormal(5, 0.5)),
        commands=[], device=legitimate_dev,
        raw_bytes=int(rng.lognormal(profile.bytes_mu, profile.bytes_sigma)),
        split=split, episode_id=ep, label="device_spoofing",
        attack_stage="recon", counter=0,
    )
    rows.append(row1)

    # 2. Spoofed device: same MAC (if reuse_mac) but different OS/version
    orig_mac = legitimate_dev["mac"]
    spoof_mac = orig_mac if reuse_mac else random_mac(rng)

    # Pick different OS
    available_os = [(o, v) for o, v in OS_POOL if o != legitimate_dev["os"]]
    if not available_os:
        available_os = OS_POOL
    spoof_os, spoof_ver = available_os[int(rng.integers(0, len(available_os)))]
    spoof_proto = str(rng.choice(cohort.protocols))
    spoof_fp = device_fingerprint(spoof_os, spoof_ver, spoof_mac, spoof_proto)
    spoof_dev = {"os": spoof_os, "version": spoof_ver, "mac": spoof_mac,
                 "protocol": spoof_proto, "fp": spoof_fp}

    # Interleave 3-5 events with spoofed device
    n_spoof = int(rng.integers(3, 6))
    t = t1 + timedelta(minutes=float(rng.uniform(5, 60)))
    for i in range(n_spoof):
        t = t + timedelta(minutes=float(rng.uniform(1, 30)))
        stage = "action_on_objective" if i >= n_spoof - 2 else "escalation"
        resource = profile.sample_resource(rng)
        rows.append(_base_row(
            profile=profile, ts=t, site=home, source_ip=ip_for_site(home, rng),
            resource=resource, auth_method=str(rng.choice(cohort.auth_methods)),
            auth_result="success", duration=float(rng.lognormal(5, 0.8)),
            commands=[], device=spoof_dev,
            raw_bytes=int(rng.lognormal(profile.bytes_mu + 0.5, profile.bytes_sigma)),
            split=split, episode_id=ep, label="device_spoofing",
            attack_stage=stage, counter=i + 1,
        ))

    return rows


# ---------------------------------------------------------------------------
# 6. Low and Slow Exfil
# ---------------------------------------------------------------------------

def inject_low_and_slow_exfil(
    profile: EntityProfile,
    start_day: datetime,
    split: str,
    rng: np.random.Generator,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Small off-hours transfers accumulating over days/weeks.

    Each individual event looks near-normal; only the cumulative picture betrays it.
    The bytes_per_session_multiplier is deliberately below a naive 10x rule.
    """
    days_min = params["days_min"]
    days_max = params["days_max"]
    multiplier = params["bytes_per_session_multiplier"]
    prefer_off_hours = params["prefer_off_hours"]

    n_days = int(rng.integers(days_min, days_max + 1))
    ep = _ep_id()
    rows = []
    cohort = COHORTS[profile.cohort]
    home = GEO_SITES[profile.home_site_idx]

    # Pick a sensitive resource to exfil from
    resource = profile.sample_resource(rng, allow_rare=False)
    dev = profile.device_fingerprints[0]
    auth_method = str(rng.choice(cohort.auth_methods))

    for day_offset in range(n_days):
        day = start_day + timedelta(days=day_offset)

        # Not every day - about 70% of days have activity
        if rng.random() > 0.70:
            continue

        # 1-3 events per day to avoid standing out
        n_events = int(rng.integers(1, 4))

        for i in range(n_events):
            if prefer_off_hours:
                # Prefer 23:00-05:00
                hour = float(rng.choice([23, 0, 1, 2, 3, 4, 5]))
                ts = day + timedelta(hours=hour, minutes=float(rng.uniform(0, 60)))
            else:
                ts = day + timedelta(hours=float(rng.uniform(0, 24)))

            # Bytes: slightly above normal but below a naive alert threshold
            normal_bytes = np.exp(rng.normal(profile.bytes_mu, profile.bytes_sigma))
            exfil_bytes = int(normal_bytes * multiplier * float(rng.uniform(0.8, 1.2)))

            stage = "action_on_objective" if day_offset >= n_days - 3 else "recon"

            rows.append(_base_row(
                profile=profile, ts=ts.replace(tzinfo=UTC), site=home,
                source_ip=ip_for_site(home, rng),
                resource=resource, auth_method=auth_method, auth_result="success",
                duration=float(rng.lognormal(profile.session_duration_mu, 0.5)),
                commands=[], device=dev,
                raw_bytes=exfil_bytes,
                split=split, episode_id=ep, label="low_and_slow_exfil",
                attack_stage=stage, counter=day_offset * 10 + i,
            ))

    return rows
