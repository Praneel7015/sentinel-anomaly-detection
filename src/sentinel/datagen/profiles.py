"""Per-entity behavioural profiles sampled from cohort priors.

Each entity gets a Profile drawn once from its cohort, with genuine
within-cohort variance so individual entities have different habits even
inside the same role.  Profiles are serialisable to JSON for dashboard
inspection and deterministic under a fixed seed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sentinel.datagen.catalog import (
    COHORTS,
    COHORTS_BY_TYPE,
    GEO_SITES,
    OS_POOL,
    CohortDef,
    device_fingerprint,
    ip_for_site,
    random_mac,
)

__all__ = ["EntityProfile", "build_profiles"]

# ---------------------------------------------------------------------------
# Profile dataclass
# ---------------------------------------------------------------------------

@dataclass
class EntityProfile:
    entity_id: str
    entity_type: str          # user | service_account | edge_device
    cohort: str
    cold_start: bool          # first appears only in the test split

    # Geo
    home_site_idx: int        # index into GEO_SITES
    secondary_site_idxs: list[int]   # occasional travel sites
    home_ip_prefix: str       # for consistent-looking IPs

    # Circadian: Gaussian mixture weights over 24h (24-element array stored as list)
    circadian_weights: list[float]
    weekday_weights: list[float]      # 7 elements, Mon=0

    # Resources: Dirichlet weights over cohort's resource pool (same length)
    resource_weights: list[float]
    # Rare resources: indices into the cohort pool that are "novel" for this entity
    rare_resource_idxs: list[int]

    # Auth
    auth_method_weights: list[float]  # over cohort.auth_methods

    # Session
    bytes_mu: float           # lognormal params (log-scale)
    bytes_sigma: float
    session_duration_mu: float
    session_duration_sigma: float

    # Device fingerprints (1-3 per entity)
    device_fingerprints: list[dict[str, str]]  # list of {os, version, mac, protocol, fp}

    # Markov command chain: stored as flat transition dict {from_cmd: {to_cmd: prob}}
    command_chain: dict[str, dict[str, float]]

    # Auth failure baseline
    auth_failure_rate: float  # per event

    # Drift support
    drifting: bool
    drift_profile: dict[str, Any] | None  # filled at generation time if drifting

    def home_site(self):
        return GEO_SITES[self.home_site_idx]

    def sample_site(self, rng: np.random.Generator) -> Any:
        """Return a GeoSite: home 90% of the time, secondary site otherwise."""
        if (not self.secondary_site_idxs) or rng.random() < 0.90:
            return GEO_SITES[self.home_site_idx]
        idx = rng.choice(self.secondary_site_idxs)
        return GEO_SITES[idx]

    def sample_hour(self, rng: np.random.Generator) -> float:
        """Sample hour-of-day from the circadian mixture."""
        weights = np.array(self.circadian_weights)
        hour = rng.choice(24, p=weights / weights.sum())
        # add sub-hour jitter
        return float(hour) + float(rng.uniform(0, 1))

    def sample_resource(self, rng: np.random.Generator, allow_rare: bool = True) -> tuple[str, str]:
        """Sample a resource from the Dirichlet-weighted pool."""
        cohort = COHORTS[self.cohort]
        weights = np.array(self.resource_weights)
        if not allow_rare:
            for idx in self.rare_resource_idxs:
                weights[idx] = 0.0
        if weights.sum() <= 0:
            weights = np.ones(len(cohort.resources))
        weights = weights / weights.sum()
        idx = rng.choice(len(cohort.resources), p=weights)
        return cohort.resources[idx]

    def sample_commands(self, rng: np.random.Generator, length: int) -> list[str]:
        """Walk the entity's Markov chain for `length` steps."""
        if not self.command_chain:
            return []
        cmds = list(self.command_chain.keys())
        if not cmds:
            return []
        current = str(rng.choice(cmds))
        result = [current]
        for _ in range(length - 1):
            transitions = self.command_chain.get(current, {})
            if not transitions:
                current = str(rng.choice(cmds))
            else:
                next_cmds = list(transitions.keys())
                probs = np.array([transitions[c] for c in next_cmds], dtype=float)
                probs /= probs.sum()
                current = str(rng.choice(next_cmds, p=probs))
            result.append(current)
        return result

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["home_site_city"] = GEO_SITES[self.home_site_idx].city
        d["home_site_country"] = GEO_SITES[self.home_site_idx].country
        return d


# ---------------------------------------------------------------------------
# Profile builder
# ---------------------------------------------------------------------------

_SITE_IDX_BY_CITY: dict[str, int] = {s.city: i for i, s in enumerate(GEO_SITES)}


def _build_circadian_weights(cohort: CohortDef, rng: np.random.Generator) -> list[float]:
    """Build a 24-element discrete PMF from the cohort's GMM spec."""
    hours = np.arange(24, dtype=float)
    density = np.zeros(24, dtype=float)
    for mean_h, weight in cohort.circadian_peaks:
        # perturb mean slightly per entity
        mu = mean_h + float(rng.normal(0, 0.5))
        sigma = cohort.circadian_sigma * float(rng.uniform(0.8, 1.2))
        # Gaussian evaluated at each hour (wrap-around via modular distance)
        diff = np.minimum(np.abs(hours - mu), 24 - np.abs(hours - mu))
        density += weight * np.exp(-0.5 * (diff / sigma) ** 2)
    # Add tiny floor so all hours have non-zero probability
    density += 1e-3
    density /= density.sum()
    return density.tolist()


def _build_resource_weights(cohort: CohortDef, rng: np.random.Generator) -> tuple[list[float], list[int]]:
    """Dirichlet-sampled weights over the cohort's resource pool."""
    n = len(cohort.resources)
    # concentration ~ 0.5 => each entity habitually touches a subset
    alpha = np.full(n, 0.5)
    weights = rng.dirichlet(alpha)
    # Mark the bottom 20% as "rare" for this entity (novelty detection fodder)
    threshold = np.percentile(weights, 20)
    rare_idxs = [i for i, w in enumerate(weights) if w <= threshold]
    return weights.tolist(), rare_idxs


def _build_command_chain(
    cohort: CohortDef, rng: np.random.Generator
) -> dict[str, dict[str, float]]:
    """Build a per-entity Markov chain perturbed from the cohort base sequence."""
    cmds = cohort.base_commands
    if not cmds:
        return {}
    n = len(cmds)
    chain: dict[str, dict[str, float]] = {}
    for i, cmd in enumerate(cmds):
        # Base: forward transition with probability ~0.6, with noise
        row = np.zeros(n)
        next_idx = (i + 1) % n
        row[next_idx] = 0.6
        # Small perturbation: distribute remaining weight
        noise = rng.dirichlet(np.ones(n) * 0.3)
        row += 0.4 * noise
        row /= row.sum()
        chain[cmd] = {cmds[j]: float(row[j]) for j in range(n) if row[j] > 1e-4}
    return chain


def _build_device_fps(
    cohort: CohortDef, rng: np.random.Generator, n_devices: int
) -> list[dict[str, str]]:
    """Build n_devices distinct device fingerprints for this entity."""
    fps = []
    proto = str(rng.choice(cohort.protocols))
    for _ in range(n_devices):
        os_name, os_ver = OS_POOL[int(rng.integers(0, len(OS_POOL)))]
        mac = random_mac(rng)
        fp = device_fingerprint(os_name, os_ver, mac, proto)
        fps.append({"os": os_name, "version": os_ver, "mac": mac, "protocol": proto, "fp": fp})
    return fps


def build_profiles(
    entity_counts: dict[str, int],   # {"user": N, "service_account": M, "edge_device": K}
    cold_start_entities: int,
    seed: int,
    drifting_entity_count: int = 0,
) -> dict[str, EntityProfile]:
    """Build one Profile per entity.  Deterministic under ``seed``.

    Returns
    -------
    dict mapping entity_id -> EntityProfile
    """
    rng = np.random.default_rng(seed)
    profiles: dict[str, EntityProfile] = {}

    # How many cold-start entities per type (proportional)
    total = sum(entity_counts.values())
    cold_remaining = cold_start_entities

    for etype, cohort_names in COHORTS_BY_TYPE.items():
        n_entities = entity_counts.get(etype, 0)
        if n_entities == 0:
            continue

        # Cold-start count for this type
        n_cold = round(cold_start_entities * n_entities / total)
        n_cold = min(n_cold, cold_remaining, n_entities)
        cold_remaining = max(0, cold_remaining - n_cold)

        # Assign cohorts with roughly equal weight
        cohort_weights = np.ones(len(cohort_names))
        cohort_assignments = rng.choice(
            cohort_names, size=n_entities, p=cohort_weights / cohort_weights.sum()
        )

        # Entity ID prefix
        prefix = {"user": "usr", "service_account": "svc", "edge_device": "dev"}[etype]
        # Start index per type (cumulative)
        offset_map = {"user": 0, "service_account": entity_counts.get("user", 0),
                      "edge_device": entity_counts.get("user", 0) + entity_counts.get("service_account", 0)}
        offset = offset_map[etype]

        for i in range(n_entities):
            entity_id = f"{prefix}_{(offset + i + 1):04d}"
            cohort_name = str(cohort_assignments[i])
            cohort = COHORTS[cohort_name]
            cold_start = i < n_cold

            # Home site
            home_cities = cohort.home_sites
            home_city = str(rng.choice(home_cities))
            home_site_idx = _SITE_IDX_BY_CITY[home_city]

            # Secondary sites (0-3 from the cohort list, not equal to home)
            sec_cities = [c for c in cohort.secondary_sites if c != home_city]
            if sec_cities:
                n_sec = int(rng.integers(0, min(3, len(sec_cities)) + 1))
                sec_idxs_arr = rng.choice(len(sec_cities), size=n_sec, replace=False)
                sec_site_idxs = [_SITE_IDX_BY_CITY[sec_cities[j]] for j in sec_idxs_arr]
            else:
                sec_site_idxs = []

            # Circadian
            circ_weights = _build_circadian_weights(cohort, rng)
            # Weekday weights with per-entity noise
            wd_noise = rng.uniform(0.9, 1.1, size=7)
            weekday_w = (np.array(cohort.weekday_weights) * wd_noise).tolist()

            # Resources
            res_weights, rare_idxs = _build_resource_weights(cohort, rng)

            # Auth method preference
            n_auth = len(cohort.auth_methods)
            auth_alpha = np.full(n_auth, 2.0)
            auth_weights = rng.dirichlet(auth_alpha).tolist()

            # Session bytes / duration
            bytes_mu = cohort.bytes_mu + float(rng.normal(0, 0.3))
            bytes_sigma = cohort.bytes_sigma * float(rng.uniform(0.8, 1.2))
            dur_mu = cohort.session_duration_mu + float(rng.normal(0, 0.2))
            dur_sigma = cohort.session_duration_sigma * float(rng.uniform(0.8, 1.2))

            # Device fingerprints (1-3)
            n_devices = int(rng.integers(1, 4))
            fps = _build_device_fps(cohort, rng, n_devices)

            # Markov chain
            cmd_chain = _build_command_chain(cohort, rng)

            # Auth failure rate: lognormal around 0.03 baseline
            afr = float(np.clip(rng.lognormal(np.log(0.03), 0.5), 0.005, 0.15))

            profiles[entity_id] = EntityProfile(
                entity_id=entity_id,
                entity_type=etype,
                cohort=cohort_name,
                cold_start=cold_start,
                home_site_idx=home_site_idx,
                secondary_site_idxs=sec_site_idxs,
                home_ip_prefix=home_city,
                circadian_weights=circ_weights,
                weekday_weights=weekday_w,
                resource_weights=res_weights,
                rare_resource_idxs=rare_idxs,
                auth_method_weights=auth_weights,
                bytes_mu=bytes_mu,
                bytes_sigma=bytes_sigma,
                session_duration_mu=dur_mu,
                session_duration_sigma=dur_sigma,
                device_fingerprints=fps,
                command_chain=cmd_chain,
                auth_failure_rate=afr,
                drifting=False,
                drift_profile=None,
            )

    # Assign drifting flag to a random subset of non-cold-start entities
    if drifting_entity_count > 0:
        candidates = [eid for eid, p in profiles.items() if not p.cold_start]
        n_drift = min(drifting_entity_count, len(candidates))
        chosen = rng.choice(candidates, size=n_drift, replace=False)
        for eid in chosen:
            profiles[eid].drifting = True

    return profiles


def save_profiles(profiles: dict[str, EntityProfile], path: str | Path) -> None:
    """Serialise all profiles to JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = {eid: p.to_json_dict() for eid, p in profiles.items()}
    with target.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
