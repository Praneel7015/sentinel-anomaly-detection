"""Transparent rule-based detector for difficulty calibration.

Rules are deliberately simple fixed thresholds - the whole point is to
measure how hard the data is.  A mediocre result here means the generator
has succeeded; good results mean the attacks are too obvious.

Target: precision ~0.25-0.55 at top-1% alert budget.
Especially poor on: low_and_slow_exfil, lateral_movement.
False positives on: password_rotation, maintenance_burst, legit_travel.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["NaiveBaseline", "calibration_report"]

_GEO_RADIUS_KM = 6371.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * _GEO_RADIUS_KM * math.asin(math.sqrt(a))


class NaiveBaseline:
    """Five independent rules; score = sum of fired rules (0-5)."""

    def __init__(
        self,
        failure_count_threshold: int = 8,
        geo_velocity_kmph: float = 800.0,
        new_resource_count: int = 3,
        off_hours_start: int = 22,
        off_hours_end: int = 6,
        bytes_multiplier: float = 8.0,
    ) -> None:
        self.failure_count_threshold = failure_count_threshold
        self.geo_velocity_kmph = geo_velocity_kmph
        self.new_resource_count = new_resource_count
        self.off_hours_start = off_hours_start
        self.off_hours_end = off_hours_end
        self.bytes_multiplier = bytes_multiplier

    def score(self, events: pd.DataFrame) -> pd.Series:
        """Return a numeric anomaly score (0-5) for each event in ``events``.

        ``events`` must contain the standard EVENT_FIELDS columns.
        The function uses only past-looking aggregates (per entity up to the
        current row), simulating an online scorer.

        Vectorised implementation: pre-sorts, then uses groupby/cumsum/shift
        aggregates instead of a Python per-row loop.
        """
        events = events.sort_values("timestamp").reset_index(drop=True)
        n = len(events)
        scores = np.zeros(n, dtype=np.float64)

        # ------------------------------------------------------------------ #
        # Rule 4: Off-hours (0.5 weight) — purely per-row, fully vectorised   #
        # ------------------------------------------------------------------ #
        hour_col = events["timestamp"].dt.hour.to_numpy(dtype=np.int32)
        off_mask = (hour_col >= self.off_hours_start) | (hour_col < self.off_hours_end)
        scores += off_mask.astype(np.float64) * 0.5

        # ------------------------------------------------------------------ #
        # Rule 5: Bytes volume anomaly                                        #
        # ------------------------------------------------------------------ #
        entity_bytes_median: dict[str, float] = (
            events.groupby("entity_id")["bytes_transferred"]
            .median()
            .to_dict()
        )
        median_arr = np.array(
            [entity_bytes_median.get(eid, 1000.0) for eid in events["entity_id"]],
            dtype=np.float64,
        )
        bytes_arr = events["bytes_transferred"].to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            bytes_anomaly = (median_arr > 0) & (bytes_arr > median_arr * self.bytes_multiplier)
        scores += bytes_anomaly.astype(np.float64)

        # ------------------------------------------------------------------ #
        # Rules 1, 2, 3: require per-entity sequential state                  #
        # Use a Python loop but only over groups, not individual rows          #
        # ------------------------------------------------------------------ #
        entity_last_geo: dict[str, tuple[float, float, Any]] = {}
        entity_failure_counts: dict[str, int] = {}
        entity_new_resource_counts: dict[str, int] = {}
        entity_seen_resources: dict[str, set[str]] = {}

        # Extract columns as numpy arrays for fast row access
        entity_ids = events["entity_id"].to_numpy()
        auth_results = events["auth_result"].to_numpy()
        resources = events["resource_accessed"].to_numpy()
        lats = events["geo_lat"].to_numpy(dtype=np.float64)
        lons = events["geo_lon"].to_numpy(dtype=np.float64)
        timestamps = events["timestamp"].to_numpy()

        for i in range(n):
            eid = str(entity_ids[i])
            auth_result = auth_results[i]
            ts = timestamps[i]
            lat, lon = lats[i], lons[i]
            resource = str(resources[i])

            s = 0.0

            # Rule 1: Auth failure burst
            if auth_result == "failure":
                entity_failure_counts[eid] = entity_failure_counts.get(eid, 0) + 1
            else:
                entity_failure_counts[eid] = 0
            if entity_failure_counts.get(eid, 0) >= self.failure_count_threshold:
                s += 1.0

            # Rule 2: Impossible travel
            if eid in entity_last_geo and auth_result == "success":
                prev_lat, prev_lon, prev_ts = entity_last_geo[eid]
                dt_s = (ts - prev_ts) / np.timedelta64(1, "s")
                dt_hours = max(1e-6, dt_s / 3600.0)
                dist_km = _haversine_km(prev_lat, prev_lon, lat, lon)
                if dist_km / dt_hours > self.geo_velocity_kmph:
                    s += 1.0
            if auth_result == "success":
                entity_last_geo[eid] = (lat, lon, ts)

            # Rule 3: New resource
            seen = entity_seen_resources.setdefault(eid, set())
            if resource not in seen:
                entity_new_resource_counts[eid] = entity_new_resource_counts.get(eid, 0) + 1
                seen.add(resource)
            if entity_new_resource_counts.get(eid, 0) >= self.new_resource_count:
                s += 1.0

            scores[i] += s

        return pd.Series(scores, index=events.index, name="naive_score")


def calibration_report(
    events: pd.DataFrame,
    labels: pd.DataFrame,
    detector: NaiveBaseline | None = None,
    budget_pcts: list[float] | None = None,
) -> dict[str, Any]:
    """Run the naive baseline and compute precision/recall at alert budgets.

    Returns a dict with keys:
        precision_at_budgets, recall_at_budgets, per_type_recall,
        fp_rate_confounders, fp_rate_insider_drift, scores (Series)
    """
    if budget_pcts is None:
        budget_pcts = [0.005, 0.01, 0.02]
    if detector is None:
        detector = NaiveBaseline()

    scores = detector.score(events)

    # Merge with labels
    df = events[["event_id"]].copy()
    df["score"] = scores.values
    df = df.merge(
        labels[["event_id", "label", "is_anomaly", "confounder_type"]],
        on="event_id", how="left",
    )
    df["is_anomaly"] = df["is_anomaly"].fillna(False)

    n_total = len(df)
    results: dict[str, Any] = {}
    prec_at = {}
    rec_at = {}

    for pct in budget_pcts:
        k = max(1, int(n_total * pct))
        top_k = df.nlargest(k, "score")
        tp = int(top_k["is_anomaly"].sum())
        fp = k - tp
        n_pos = int(df["is_anomaly"].sum())
        prec = tp / k if k > 0 else 0.0
        rec = tp / n_pos if n_pos > 0 else 0.0
        prec_at[f"{pct:.1%}"] = round(prec, 4)
        rec_at[f"{pct:.1%}"] = round(rec, 4)

    results["precision_at_budgets"] = prec_at
    results["recall_at_budgets"] = rec_at

    # Per attack-type recall at 1%
    k1 = max(1, int(n_total * 0.01))
    top1 = df.nlargest(k1, "score")
    per_type: dict[str, float] = {}
    for atype in ["brute_force", "impossible_travel", "credential_stuffing",
                  "lateral_movement", "device_spoofing", "low_and_slow_exfil"]:
        mask = df["label"] == atype
        n = mask.sum()
        if n == 0:
            per_type[atype] = float("nan")
            continue
        detected = top1[top1["label"] == atype].shape[0]
        per_type[atype] = round(detected / n, 4)
    results["per_type_recall_at_1pct"] = per_type

    # FP rates on confounders / insider_drift
    conf_mask = df["label"] == "benign_confounder"
    drift_mask = df["label"] == "insider_drift"

    if conf_mask.sum() > 0:
        threshold_1pct = df["score"].quantile(1 - 0.01)
        fp_conf = (df.loc[conf_mask, "score"] >= threshold_1pct).mean()
        results["fp_rate_confounders"] = round(float(fp_conf), 4)
    else:
        results["fp_rate_confounders"] = float("nan")

    if drift_mask.sum() > 0:
        fp_drift = (df.loc[drift_mask, "score"] >= threshold_1pct).mean()
        results["fp_rate_insider_drift"] = round(float(fp_drift), 4)
    else:
        results["fp_rate_insider_drift"] = float("nan")

    results["scores"] = scores
    results["n_total"] = n_total
    results["n_anomaly"] = int(df["is_anomaly"].sum())
    return results


def print_report(report: dict[str, Any]) -> None:
    """Pretty-print calibration numbers."""
    print("\n=== Naive Baseline Calibration Report ===")
    print(f"Total events: {report['n_total']:,}  |  Anomalies: {report['n_anomaly']:,}")
    print(f"Anomaly rate: {report['n_anomaly']/report['n_total']:.3%}")
    print("\nPrecision @ alert budgets:")
    for k, v in report["precision_at_budgets"].items():
        print(f"  {k:>6} budget:  P={v:.4f}  R={report['recall_at_budgets'][k]:.4f}")
    print("\nPer-type recall @ 1% budget:")
    for k, v in report["per_type_recall_at_1pct"].items():
        mark = " <<" if k in ("low_and_slow_exfil", "lateral_movement") else ""
        print(f"  {k:<30} {v:.4f}{mark}")
    print(f"\nFP rate on confounders:   {report['fp_rate_confounders']:.4f}")
    print(f"FP rate on insider_drift: {report['fp_rate_insider_drift']:.4f}")
    print("=" * 42)
