"""Main generation orchestrator - fully vectorised normal event generation.

Strategy: for each day, build large matrices across all active entities in
one numpy batch, then construct the DataFrame directly from arrays.
Attacks and confounders are small (O(episodes)) so they use the original
event-by-event helpers.
"""
from __future__ import annotations

import hashlib
import json
import logging
import struct
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentinel.config import DataConfig, load_data_config, project_root
from sentinel.datagen.attacks import (
    inject_brute_force,
    inject_credential_stuffing,
    inject_device_spoofing,
    inject_impossible_travel,
    inject_lateral_movement,
    inject_low_and_slow_exfil,
)
from sentinel.datagen.catalog import COHORTS, GEO_SITES
from sentinel.datagen.confounders import (
    inject_insider_drift,
    inject_legit_travel,
    inject_maintenance_burst,
    inject_new_device_enrollment,
    inject_password_rotation,
    inject_vacation_return,
)
from sentinel.datagen.profiles import EntityProfile, build_profiles, save_profiles
from sentinel.io import write_events, write_labels
from sentinel.schema import EVENT_FIELDS, LABEL_FIELDS

log = logging.getLogger(__name__)

__all__ = ["generate"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_for_day(day_index: int, cfg: DataConfig) -> str:
    if day_index < cfg.time.train_days:
        return "train"
    if day_index < cfg.time.train_days + cfg.time.val_days:
        return "val"
    return "test"


def _day_dt(cfg: DataConfig, day_index: int) -> datetime:
    d = cfg.time.start_date + timedelta(days=day_index)
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def _config_hash(cfg: DataConfig) -> str:
    blob = json.dumps(
        {
            "seed": cfg.seed,
            "total_days": cfg.time.total_days,
            "n_users": cfg.entities.users,
            "n_svc": cfg.entities.service_accounts,
            "n_dev": cfg.entities.edge_devices,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _rows_to_frames(all_rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split mixed rows into (events_df, labels_df)."""
    events_records = []
    label_records = []
    for row in all_rows:
        evt = {k: row[k] for k in EVENT_FIELDS if k in row}
        events_records.append(evt)
        label_records.append({
            "event_id": row["event_id"],
            "episode_id": row.get("episode_id"),
            "label": row["_label"],
            "is_anomaly": row["_is_anomaly"],
            "confounder_type": row.get("_confounder_type"),
            "attack_stage": row.get("_attack_stage"),
        })
    events_df = pd.DataFrame(events_records, columns=EVENT_FIELDS)
    labels_df = pd.DataFrame(label_records, columns=LABEL_FIELDS)
    return events_df, labels_df


def _df_to_frames(combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a combined DataFrame (events + label columns) into events_df and labels_df."""
    events_df = combined[[c for c in EVENT_FIELDS if c in combined.columns]].copy()
    # Ensure all EVENT_FIELDS columns are present (fill missing with None)
    for col in EVENT_FIELDS:
        if col not in events_df.columns:
            events_df[col] = None

    labels_df = pd.DataFrame({
        "event_id":       combined["event_id"],
        "episode_id":     combined["episode_id"] if "episode_id" in combined.columns else None,
        "label":          combined["_label"],
        "is_anomaly":     combined["_is_anomaly"],
        "confounder_type": combined["_confounder_type"] if "_confounder_type" in combined.columns else None,
        "attack_stage":   combined["_attack_stage"] if "_attack_stage" in combined.columns else None,
    }, columns=LABEL_FIELDS)

    return events_df[EVENT_FIELDS], labels_df


# ---------------------------------------------------------------------------
# Vectorised normal event generation
# ---------------------------------------------------------------------------

# Module-level cache: precompute site subnet prefixes once
_SITE_NET_PREFIXES: list[str] = []

def _ensure_site_prefixes() -> list[str]:
    global _SITE_NET_PREFIXES
    if not _SITE_NET_PREFIXES:
        for site in GEO_SITES:
            parts = site.subnet.split(".")
            _SITE_NET_PREFIXES.append(f"{parts[0]}.{parts[1]}")
    return _SITE_NET_PREFIXES


def _generate_normal_batch(
    profiles_today: list[EntityProfile],
    n_per_entity: np.ndarray,   # shape (n_entities,) int
    day_epoch_sec: int,
    split: str,
    rng: np.random.Generator,
    noise_off_hours: float,
    noise_rare_resource: float,
    noise_auth_failure: float,
    drift_magnitudes: np.ndarray,  # shape (n_entities,) float
    global_counter: list[int],
) -> pd.DataFrame:
    """Generate all normal events for a batch of entities on one day.

    Returns a DataFrame (not list[dict]) to avoid per-row Python overhead.
    All sampling is done in NumPy arrays; the DataFrame is assembled from
    column arrays in a single pd.DataFrame(...) call.
    """
    total_events = int(n_per_entity.sum())
    if total_events == 0:
        return pd.DataFrame(columns=list(EVENT_FIELDS) + ["_label", "_is_anomaly", "_confounder_type", "_attack_stage", "episode_id"])

    # Build index arrays: which entity does each event belong to?
    entity_indices = np.repeat(np.arange(len(profiles_today)), n_per_entity)

    # Pre-extract cohort data for each profile
    cohorts_today = [COHORTS[p.cohort] for p in profiles_today]

    # Precompute slice boundaries for per-entity loops (avoids O(n_entities * n_events) mask ops)
    # Since entity_indices = [0,0,...,1,1,...,k,k,...] via np.repeat, slices are contiguous.
    cum_counts = np.concatenate([[0], np.cumsum(n_per_entity)]).astype(np.int64)

    # ---- Timestamps ----
    off_mask = rng.random(total_events) < noise_off_hours
    hours = np.empty(total_events, dtype=np.float64)

    for eidx, prof in enumerate(profiles_today):
        s, e = int(cum_counts[eidx]), int(cum_counts[eidx + 1])
        if s == e:
            continue
        off_e = off_mask[s:e]
        circ = np.array(prof.circadian_weights, dtype=np.float64)
        circ /= circ.sum()
        h = np.empty(e - s, dtype=np.float64)
        n_on = int((~off_e).sum())
        n_off_h = (e - s) - n_on
        if n_on > 0:
            h[~off_e] = rng.choice(24, size=n_on, p=circ).astype(np.float64) + rng.uniform(0, 1, size=n_on)
        if n_off_h > 0:
            h[off_e] = rng.uniform(22, 30, size=n_off_h) % 24
        hours[s:e] = h

    seconds_offset = (hours * 3600).astype(np.int64)

    # ---- Auth failures ----
    fail_rates = np.array([
        min(0.99, p.auth_failure_rate + noise_auth_failure) for p in profiles_today
    ])
    per_event_fail_rate = fail_rates[entity_indices]
    auth_failed = rng.random(total_events) < per_event_fail_rate

    # ---- Bytes ----
    bytes_mu_arr = np.array([p.bytes_mu for p in profiles_today])
    bytes_sig_arr = np.array([p.bytes_sigma for p in profiles_today])
    per_event_bytes_mu = bytes_mu_arr[entity_indices] + 0.5 * drift_magnitudes[entity_indices]
    per_event_bytes_sig = bytes_sig_arr[entity_indices]
    raw_bytes = np.exp(rng.normal(per_event_bytes_mu, per_event_bytes_sig)).astype(np.int64)
    raw_bytes = np.maximum(0, raw_bytes)
    fail_idxs = np.where(auth_failed)[0]
    if len(fail_idxs) > 0:
        raw_bytes[fail_idxs] = rng.integers(0, 512, size=len(fail_idxs))

    # ---- Duration ----
    dur_mu_arr = np.array([p.session_duration_mu for p in profiles_today])
    dur_sig_arr = np.array([p.session_duration_sigma for p in profiles_today])
    durations = np.maximum(1.0, np.exp(
        rng.normal(dur_mu_arr[entity_indices], dur_sig_arr[entity_indices])
    ))

    # ---- Device selection ----
    fp_count_arr = np.array([len(p.device_fingerprints) for p in profiles_today])
    fp_idxs = (rng.random(total_events) * fp_count_arr[entity_indices]).astype(np.int32)

    # ---- Site selection ----
    use_secondary = rng.random(total_events) < 0.05
    site_idxs = np.array([p.home_site_idx for p in profiles_today])[entity_indices]
    for eidx, prof in enumerate(profiles_today):
        if not prof.secondary_site_idxs:
            continue
        s, e = int(cum_counts[eidx]), int(cum_counts[eidx + 1])
        if s == e:
            continue
        sec_e = use_secondary[s:e]
        n_sec = int(sec_e.sum())
        if n_sec > 0:
            sec_arr = np.array(prof.secondary_site_idxs, dtype=np.int32)
            site_idxs[s:e][sec_e] = rng.choice(sec_arr, size=n_sec)

    # ---- Resource selection ----
    use_rare = rng.random(total_events) < noise_rare_resource
    res_idxs = np.zeros(total_events, dtype=np.int32)
    for eidx, prof in enumerate(profiles_today):
        s, e = int(cum_counts[eidx]), int(cum_counts[eidx + 1])
        n_e = e - s
        if n_e == 0:
            continue
        coh = cohorts_today[eidx]
        n_res = len(coh.resources)
        dm = float(drift_magnitudes[eidx])
        rw = np.array(prof.resource_weights, dtype=np.float64)
        if dm > 0:
            for idx in prof.rare_resource_idxs:
                rw[idx] += dm * 0.3
        if rw.sum() <= 0:
            rw = np.ones(n_res)
        rw /= rw.sum()

        rare_e = use_rare[s:e]
        n_norm = int((~rare_e).sum())
        n_rare = n_e - n_norm
        if n_norm > 0:
            res_idxs[s:e][~rare_e] = rng.choice(n_res, size=n_norm, p=rw)
        if n_rare > 0:
            if prof.rare_resource_idxs:
                rare_arr = np.array(prof.rare_resource_idxs, dtype=np.int32)
                res_idxs[s:e][rare_e] = rng.choice(rare_arr, size=n_rare)
            else:
                res_idxs[s:e][rare_e] = rng.integers(0, n_res, size=n_rare)

    # ---- Auth method ----
    auth_method_idxs = np.zeros(total_events, dtype=np.int32)
    for eidx, prof in enumerate(profiles_today):
        s, e = int(cum_counts[eidx]), int(cum_counts[eidx + 1])
        n_e = e - s
        if n_e == 0:
            continue
        coh = cohorts_today[eidx]
        n_auth = len(coh.auth_methods)
        aw = np.array(prof.auth_method_weights, dtype=np.float64)
        aw /= aw.sum()
        auth_method_idxs[s:e] = rng.choice(n_auth, size=n_e, p=aw)

    # ---- Vectorised IP generation ----
    site_prefixes = _ensure_site_prefixes()
    ip_oct3 = rng.integers(1, 255, size=total_events, dtype=np.int32)
    ip_oct4 = rng.integers(1, 255, size=total_events, dtype=np.int32)

    # ---- Vectorised event IDs ----
    # Pre-hash each entity_id once, then XOR with a per-event counter
    # (avoids calling hashlib.sha256 once per event in a Python loop)
    counter_start = global_counter[0] + 1
    global_counter[0] = counter_start + total_events - 1
    counters = np.arange(counter_start, counter_start + total_events, dtype=np.uint64)

    entity_id_hashes = np.empty(len(profiles_today), dtype=np.uint64)
    for eidx2, prof in enumerate(profiles_today):
        h = hashlib.sha256(prof.entity_id.encode()).digest()
        entity_id_hashes[eidx2] = struct.unpack_from("<Q", h, 0)[0]

    mixed = entity_id_hashes[entity_indices] ^ counters
    event_ids = [f"{v:016x}" for v in mixed]

    # ---- Vectorised timestamps (UTC, no per-event datetime call) ----
    ts_epochs_ns = (day_epoch_sec + seconds_offset).astype("int64") * 1_000_000_000
    timestamps = pd.to_datetime(ts_epochs_ns, utc=True)

    # ---- Vectorised string columns (zero Python per-event loop) ----
    # Use np.repeat on per-entity arrays so string lookups are done O(n_entities) not O(n_events)

    # Entity-level string arrays (replicate n_per_entity times each)
    entity_id_arr   = np.array([p.entity_id   for p in profiles_today], dtype=object)
    entity_type_arr = np.array([p.entity_type for p in profiles_today], dtype=object)
    cohort_arr      = np.array([p.cohort       for p in profiles_today], dtype=object)

    entity_id_col   = entity_id_arr[entity_indices]
    entity_type_col = entity_type_arr[entity_indices]
    cohort_col      = cohort_arr[entity_indices]

    # Site-level string arrays indexed by site_idxs (all GEO_SITES, pre-expanded)
    _site_country = np.array([s.country for s in GEO_SITES], dtype=object)
    _site_city    = np.array([s.city    for s in GEO_SITES], dtype=object)
    _site_lat     = np.array([s.lat     for s in GEO_SITES], dtype=np.float64)
    _site_lon     = np.array([s.lon     for s in GEO_SITES], dtype=np.float64)
    _site_prefix  = np.array(site_prefixes, dtype=object)

    geo_country_col = _site_country[site_idxs]
    geo_city_col    = _site_city[site_idxs]
    geo_lat_col     = _site_lat[site_idxs]
    geo_lon_col     = _site_lon[site_idxs]
    pfx_col         = _site_prefix[site_idxs]

    # IPs: fully vectorised via numpy char operations (avoids Python-level loop)
    pfx_col = _site_prefix[site_idxs]
    sep = np.full(total_events, ".", dtype=object)
    source_ip_col = np.char.add(
        np.char.add(np.char.add(np.char.add(pfx_col.astype(str), sep), ip_oct3.astype(str)), sep),
        ip_oct4.astype(str),
    )

    # For resource, auth_method, and device columns: use cum_counts for direct slicing
    # entity_indices is sorted (0,0,...,1,1,...,k,k,...) because of np.repeat
    resource_acc_flat = np.empty(total_events, dtype=object)
    resource_typ_flat = np.empty(total_events, dtype=object)
    auth_method_flat  = np.empty(total_events, dtype=object)
    dev_os_flat       = np.empty(total_events, dtype=object)
    dev_ver_flat      = np.empty(total_events, dtype=object)
    dev_mac_flat      = np.empty(total_events, dtype=object)
    dev_proto_flat    = np.empty(total_events, dtype=object)
    dev_fp_flat       = np.empty(total_events, dtype=object)

    for eidx in range(len(profiles_today)):
        s, e = int(cum_counts[eidx]), int(cum_counts[eidx + 1])
        n_e = e - s
        if n_e == 0:
            continue
        prof_e = profiles_today[eidx]
        coh_e  = cohorts_today[eidx]

        # Resources — direct slice, no np.split
        res_list = coh_e.resources
        ri = res_idxs[s:e]
        resource_acc_flat[s:e] = [res_list[r][0] for r in ri]
        resource_typ_flat[s:e] = [res_list[r][1] for r in ri]

        # Auth method
        am_list = coh_e.auth_methods
        ai = auth_method_idxs[s:e]
        auth_method_flat[s:e] = [am_list[a] for a in ai]

        # Device fingerprints
        fp_list = prof_e.device_fingerprints
        fi = fp_idxs[s:e]
        dev_os_flat[s:e]    = [fp_list[f]["os"]       for f in fi]
        dev_ver_flat[s:e]   = [fp_list[f]["version"]  for f in fi]
        dev_mac_flat[s:e]   = [fp_list[f]["mac"]      for f in fi]
        dev_proto_flat[s:e] = [fp_list[f]["protocol"] for f in fi]
        dev_fp_flat[s:e]    = [fp_list[f]["fp"]       for f in fi]

    # ---- Build DataFrame from columns (single allocation, no per-row dict) ----
    # command_sequence: store None placeholder here; the caller replaces with []
    # after all per-day DataFrames have been concatenated.  Storing Python list
    # objects ([]) inside a DataFrame column forces Pandas to copy each list
    # individually during pd.concat — with 70 day-batches and 2-3M rows total
    # that loop dominates runtime.  A None column concatenates as a fast object
    # array of scalar NoneType values; one vectorised replace at the end is O(N).
    df = pd.DataFrame({
        "event_id":            event_ids,
        "episode_id":          None,
        "entity_id":           entity_id_col,
        "entity_type":         entity_type_col,
        "cohort":              cohort_col,
        "timestamp":           timestamps,
        "source_ip":           source_ip_col,
        "geo_country":         geo_country_col,
        "geo_city":            geo_city_col,
        "geo_lat":             geo_lat_col,
        "geo_lon":             geo_lon_col,
        "resource_accessed":   resource_acc_flat,
        "resource_type":       resource_typ_flat,
        "auth_method":         auth_method_flat,
        "auth_result":         np.where(auth_failed, "failure", "success"),
        "session_duration_s":  durations.astype(np.float64),
        "command_sequence":    None,  # placeholder — filled after concat
        "device_os":           dev_os_flat,
        "device_os_version":   dev_ver_flat,
        "device_mac":          dev_mac_flat,
        "device_protocol":     dev_proto_flat,
        "device_fingerprint":  dev_fp_flat,
        "bytes_transferred":   raw_bytes.astype(np.int64),
        "split":               split,
        "_label":              "normal",
        "_is_anomaly":         False,
        "_confounder_type":    None,
        "_attack_stage":       None,
    })
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate(
    config_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    seed_override: int | None = None,
    artifacts_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the full corpus.  Returns (events_df, labels_df)."""
    cfg = load_data_config(config_path)
    seed = seed_override if seed_override is not None else cfg.seed
    rng = np.random.default_rng(seed)

    out_path = Path(out_dir) if out_dir else project_root() / cfg.output_dir
    arts_path = Path(artifacts_dir) if artifacts_dir else project_root() / "artifacts"
    out_path.mkdir(parents=True, exist_ok=True)
    arts_path.mkdir(parents=True, exist_ok=True)

    log.info("Building entity profiles (seed=%d)...", seed)
    entity_counts = cfg.entities.as_dict()
    profiles = build_profiles(
        entity_counts=entity_counts,
        cold_start_entities=cfg.cold_start_entities,
        seed=seed,
        drifting_entity_count=cfg.drift.drifting_entities,
    )
    save_profiles(profiles, arts_path / "entity_profiles.json")

    all_entity_ids = list(profiles.keys())
    n_total_entities = len(all_entity_ids)
    log.info("Profiles built: %d entities", n_total_entities)

    total_days = cfg.time.total_days
    train_end = cfg.time.train_days
    val_end = train_end + cfg.time.val_days

    cold_entities = {eid for eid, p in profiles.items() if p.cold_start}
    normal_entity_ids = [eid for eid in all_entity_ids if eid not in cold_entities]
    vol_cfg = cfg.events_per_entity_per_day

    # ---- Generate normal events (vectorised per day) ----
    log.info("Generating normal events (vectorised)...")
    normal_dfs: list[pd.DataFrame] = []
    global_counter: list[int] = [0]

    for day_idx in range(total_days):
        if day_idx % 10 == 0:
            log.info("  Normal events: day %d / %d", day_idx, total_days)
        split = _split_for_day(day_idx, cfg)
        day_dt = _day_dt(cfg, day_idx)
        day_epoch = int(day_dt.timestamp())
        weekday = day_dt.weekday()

        active_ids = normal_entity_ids if split != "test" else all_entity_ids
        active_profiles = [profiles[eid] for eid in active_ids]

        # Weekday filter
        wd_weights = np.array([p.weekday_weights[weekday] for p in active_profiles])
        active_mask = rng.random(len(active_profiles)) < wd_weights
        day_profiles = [p for p, m in zip(active_profiles, active_mask) if m]
        if not day_profiles:
            continue

        # Event counts per entity
        n_per_entity = np.array([
            int(rng.integers(vol_cfg[p.entity_type].min, vol_cfg[p.entity_type].max + 1))
            for p in day_profiles
        ], dtype=np.int64)

        # Drift magnitudes
        drift_mags = np.zeros(len(day_profiles), dtype=np.float64)
        for i, p in enumerate(day_profiles):
            if p.drifting and day_idx >= cfg.drift.drift_start_day:
                progress = min(1.0, (day_idx - cfg.drift.drift_start_day) / max(1, cfg.drift.ramp_days))
                drift_mags[i] = cfg.drift.magnitude * progress

        day_df = _generate_normal_batch(
            profiles_today=day_profiles,
            n_per_entity=n_per_entity,
            day_epoch_sec=day_epoch,
            split=split,
            rng=rng,
            noise_off_hours=cfg.noise.off_hours_fraction,
            noise_rare_resource=cfg.noise.rare_resource_fraction,
            noise_auth_failure=cfg.noise.baseline_auth_failure_rate,
            drift_magnitudes=drift_mags,
            global_counter=global_counter,
        )
        if len(day_df) > 0:
            normal_dfs.append(day_df)

    # Concatenate all normal-event DataFrames at once
    if normal_dfs:
        normal_df = pd.concat(normal_dfs, ignore_index=True)
        # Fill command_sequence placeholder (None → []) in one pass after concat.
        # This avoids storing Python list objects in per-day DataFrames, which
        # makes pd.concat copy each [] individually (O(N) Python-level loop).
        normal_df["command_sequence"] = [[] for _ in range(len(normal_df))]
    else:
        normal_df = pd.DataFrame(columns=list(EVENT_FIELDS) + ["_label", "_is_anomaly", "_confounder_type", "_attack_stage", "episode_id"])

    total_normal = len(normal_df)
    log.info("Normal events generated: %d", total_normal)

    # ---- Collect all resources for insider_drift ----
    all_resource_pool: list[tuple[str, str]] = []
    for cohort in COHORTS.values():
        all_resource_pool.extend(cohort.resources)

    profile_list = list(profiles.values())

    def _pick_profile(entity_type: str | None = None) -> EntityProfile:
        if entity_type:
            candidates = [p for p in profile_list if p.entity_type == entity_type]
        else:
            candidates = profile_list
        return candidates[int(rng.integers(0, len(candidates)))]

    def _pick_split_day(spl: str) -> datetime:
        if spl == "train":
            di = int(rng.integers(0, train_end))
        elif spl == "val":
            di = int(rng.integers(train_end, val_end))
        else:
            di = int(rng.integers(val_end, total_days))
        return _day_dt(cfg, di)

    def _random_split() -> str:
        r = rng.random()
        if r < 0.55:
            return "train"
        if r < 0.75:
            return "val"
        return "test"

    # ---- Inject attacks (still list[dict] — O(episodes), not O(events)) ----
    log.info("Injecting attacks...")
    attack_rows: list[dict[str, Any]] = []

    # Cap episode counts: rates × total_normal can produce hundreds of thousands
    # of episodes with a large corpus (5M+ events), which would take hours in
    # pure-Python list[dict] construction.  We clamp at a per-type ceiling that
    # preserves diversity (1 episode per ~10 entities) while keeping runtime under
    # a minute.  Anomaly rate is preserved because attack events are a tiny
    # fraction of normal events at any episode count above ~50.
    _ep_cap = max(50, n_total_entities // 2)

    # Brute force
    bf_spec = cfg.attacks["brute_force"]
    n_bf = min(_ep_cap, max(1, int(total_normal * bf_spec.rate)))
    for _ in range(n_bf):
        spl = _random_split()
        attack_rows.extend(inject_brute_force(_pick_profile("user"), _pick_split_day(spl), spl, rng, bf_spec.params))

    # Impossible travel
    it_spec = cfg.attacks["impossible_travel"]
    n_it = min(_ep_cap, max(1, int(total_normal * it_spec.rate)))
    for _ in range(n_it):
        spl = _random_split()
        attack_rows.extend(inject_impossible_travel(_pick_profile("user"), _pick_split_day(spl), spl, rng, it_spec.params))

    # Credential stuffing
    cs_spec = cfg.attacks["credential_stuffing"]
    n_cs = min(_ep_cap, max(1, int(total_normal * cs_spec.rate)))
    user_profiles = [p for p in profile_list if p.entity_type == "user"]
    for _ in range(n_cs):
        spl = _random_split()
        attack_rows.extend(inject_credential_stuffing(user_profiles, _pick_split_day(spl), spl, rng, cs_spec.params))

    # Lateral movement
    lm_spec = cfg.attacks["lateral_movement"]
    n_lm = min(_ep_cap, max(1, int(total_normal * lm_spec.rate)))
    for _ in range(n_lm):
        spl = _random_split()
        attack_rows.extend(inject_lateral_movement(_pick_profile("user"), profile_list, _pick_split_day(spl), spl, rng, lm_spec.params))

    # Device spoofing
    ds_spec = cfg.attacks["device_spoofing"]
    n_ds = min(_ep_cap, max(1, int(total_normal * ds_spec.rate)))
    for _ in range(n_ds):
        spl = _random_split()
        attack_rows.extend(inject_device_spoofing(_pick_profile("edge_device"), _pick_split_day(spl), spl, rng, ds_spec.params))

    # Low and slow exfil
    lse_spec = cfg.attacks["low_and_slow_exfil"]
    n_lse = min(_ep_cap, max(1, int(total_normal * lse_spec.rate)))
    for _ in range(n_lse):
        start_di = int(rng.integers(0, max(1, train_end - lse_spec.params["days_max"])))
        rows = inject_low_and_slow_exfil(_pick_profile(), _day_dt(cfg, start_di), "train", rng, lse_spec.params)
        for row in rows:
            di = (row["timestamp"].date() - cfg.time.start_date).days
            row["split"] = _split_for_day(di, cfg)
        attack_rows.extend(rows)

    log.info("Attacks injected; attack events: %d (episodes: bf=%d it=%d cs=%d lm=%d ds=%d lse=%d)",
             len(attack_rows), n_bf, n_it, n_cs, n_lm, n_ds, n_lse)

    # ---- Inject confounders ----
    log.info("Injecting confounders...")
    conf_cfg = cfg.confounders
    confounder_rows: list[dict[str, Any]] = []
    normal_user_profiles = [p for p in profile_list if p.entity_type == "user" and not p.cold_start]
    svc_profiles = [p for p in profile_list if p.entity_type == "service_account" and not p.cold_start]
    if not svc_profiles:
        svc_profiles = profile_list

    travel_candidates = [p for p in normal_user_profiles if p.cohort == "sales_field"] or normal_user_profiles

    n_lt = min(_ep_cap, max(1, int(total_normal * conf_cfg["legit_travel"])))
    for _ in range(n_lt):
        spl = _random_split()
        prof = travel_candidates[int(rng.integers(0, len(travel_candidates)))]
        confounder_rows.extend(inject_legit_travel(prof, _pick_split_day(spl), spl, rng))

    n_nde = min(_ep_cap, max(1, int(total_normal * conf_cfg["new_device_enrollment"])))
    for _ in range(n_nde):
        spl = _random_split()
        prof = normal_user_profiles[int(rng.integers(0, len(normal_user_profiles)))]
        confounder_rows.extend(inject_new_device_enrollment(prof, _pick_split_day(spl), spl, rng))

    n_pr = min(_ep_cap, max(1, int(total_normal * conf_cfg["password_rotation"])))
    for _ in range(n_pr):
        spl = _random_split()
        prof = normal_user_profiles[int(rng.integers(0, len(normal_user_profiles)))]
        confounder_rows.extend(inject_password_rotation(prof, _pick_split_day(spl), spl, rng))

    n_vr = min(_ep_cap, max(1, int(total_normal * conf_cfg["vacation_return"])))
    for _ in range(n_vr):
        spl = _random_split()
        prof = normal_user_profiles[int(rng.integers(0, len(normal_user_profiles)))]
        confounder_rows.extend(inject_vacation_return(prof, _pick_split_day(spl), spl, rng))

    n_mb = min(_ep_cap, max(1, int(total_normal * conf_cfg["maintenance_burst"])))
    for _ in range(n_mb):
        spl = _random_split()
        prof = svc_profiles[int(rng.integers(0, len(svc_profiles)))]
        confounder_rows.extend(inject_maintenance_burst(prof, _pick_split_day(spl), spl, rng))

    # Insider drift
    drift_spec = cfg.edge_cases["insider_drift"]
    n_drift_eps = max(1, int(len(normal_user_profiles) * drift_spec.rate))
    ramp_days_val = drift_spec.params["ramp_days"]
    new_res_frac = drift_spec.params["new_resource_fraction"]
    for _ in range(n_drift_eps):
        start_di = int(rng.integers(0, max(1, train_end - ramp_days_val)))
        prof = normal_user_profiles[int(rng.integers(0, len(normal_user_profiles)))]
        rows = inject_insider_drift(prof, _day_dt(cfg, start_di), ramp_days_val, new_res_frac, "train", rng, all_resource_pool)
        for row in rows:
            di = (row["timestamp"].date() - cfg.time.start_date).days
            row["split"] = _split_for_day(di, cfg)
        confounder_rows.extend(rows)

    log.info("Confounders injected; confounder events: %d", len(confounder_rows))

    # ---- Build combined DataFrame ----
    log.info("Building DataFrames...")

    # Convert attack + confounder rows to DataFrame
    extra_rows = attack_rows + confounder_rows
    if extra_rows:
        extra_df = pd.DataFrame(extra_rows)
    else:
        extra_df = pd.DataFrame(columns=list(EVENT_FIELDS) + ["_label", "_is_anomaly", "_confounder_type", "_attack_stage", "episode_id"])

    # Ensure timestamp column in extra_df is tz-aware UTC to match normal_df
    if len(extra_df) > 0 and "timestamp" in extra_df.columns:
        if extra_df["timestamp"].dt.tz is None:
            extra_df["timestamp"] = extra_df["timestamp"].dt.tz_localize(UTC)
        else:
            extra_df["timestamp"] = extra_df["timestamp"].dt.tz_convert(UTC)

    combined_df = pd.concat([normal_df, extra_df], ignore_index=True)
    log.info("Combined rows: %d", len(combined_df))

    # Split into events + labels
    events_df, labels_df = _df_to_frames(combined_df)
    events_df = events_df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    labels_df = labels_df.set_index("event_id").loc[events_df["event_id"]].reset_index()

    # ---- Write parquet ----
    log.info("Writing parquet to %s ...", out_path)
    write_events(events_df, out_path / "events.parquet")
    write_labels(labels_df, out_path / "labels.parquet")

    # ---- Manifest ----
    n_anomaly = int(labels_df["is_anomaly"].sum())
    n_total_events = len(events_df)
    manifest = {
        "seed": seed,
        "config_hash": _config_hash(cfg),
        "total_events": n_total_events,
        "anomaly_rate": round(n_anomaly / max(1, n_total_events), 6),
        "label_counts": labels_df["label"].value_counts().to_dict(),
        "split_counts": events_df["split"].value_counts().to_dict(),
        "n_entities": n_total_entities,
        "cold_start_entities": len(cold_entities),
        "drifting_entities": sum(1 for p in profiles.values() if p.drifting),
        "entity_counts_by_type": {
            etype: sum(1 for p in profiles.values() if p.entity_type == etype)
            for etype in ["user", "service_account", "edge_device"]
        },
    }
    with (out_path / "generation_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    log.info("Done. %d events, anomaly_rate=%.4f", n_total_events, manifest["anomaly_rate"])
    return events_df, labels_df
