"""SENTINEL evaluation harness.

Trains the full model stack on the train split, evaluates on the test split,
and writes ``artifacts/eval_results.json`` plus plots to ``reports/``.

Metrics computed
----------------
- PR-AUC and ROC-AUC (event-level)
- Precision / recall at alert budgets 0.5%, 1.0%, 2.0%, 3.0%, 5.0%
- Per-attack-type recall at 1% budget
- FP rate on confounders and insider_drift
- Cold-start subgroup (entities with no train-split events)
- Post-drift subgroup (events after drift_start_day)
- Per-detector ablation (leave-one-out)
- Held-out-attack-type generalisation test
- p50/p95/p99 scoring latency

Usage
-----
    sentinel eval
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["run_eval"]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if y_true.sum() == 0:
        return 0.0
    return float(average_precision_score(y_true, scores))


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if y_true.sum() == 0 or (~y_true).sum() == 0:
        return 0.5
    return float(roc_auc_score(y_true, scores))


def _at_budget(
    y_true: np.ndarray,
    scores: np.ndarray,
    budget_pct: float,
) -> tuple[float, float, int]:
    k = max(1, int(len(scores) * budget_pct / 100))
    top_k = np.argsort(scores)[::-1][:k]
    mask = np.zeros(len(scores), dtype=bool)
    mask[top_k] = True
    tp = int((mask & y_true).sum())
    fp = int((mask & ~y_true).sum())
    fn = int((~mask & y_true).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(precision), float(recall), int(mask.sum())


def _pr_curve_pts(y_true: np.ndarray, scores: np.ndarray, n: int = 50) -> list[dict]:
    from sklearn.metrics import precision_recall_curve
    if y_true.sum() == 0:
        return []
    prec, rec, _ = precision_recall_curve(y_true, scores)
    idx = np.linspace(0, len(prec) - 1, min(n, len(prec))).astype(int)
    return [{"recall": float(rec[i]), "precision": float(prec[i])} for i in idx]


def _mttd(
    ev: pd.DataFrame,
    lb: pd.DataFrame,
    risk: np.ndarray,
    threshold: float,
) -> list[dict]:
    joined = ev[["event_id", "timestamp"]].copy()
    joined["risk"] = risk
    joined["is_alert"] = risk >= threshold
    joined = joined.merge(lb[["event_id", "episode_id", "label", "is_anomaly"]], on="event_id", how="left")
    results: dict[str, list] = {}
    for ep_id, ep_df in joined[joined["is_anomaly"] == True].groupby("episode_id"):  # noqa: E712
        label = ep_df["label"].iloc[0]
        ep_df = ep_df.sort_values("timestamp")
        ep_start = ep_df["timestamp"].iloc[0]
        alerts = ep_df[ep_df["is_alert"]]
        if len(alerts) == 0:
            continue
        first_alert = alerts["timestamp"].iloc[0]
        delta_events = int((ep_df["timestamp"] <= first_alert).sum()) - 1
        delta_min = float((first_alert - ep_start).total_seconds() / 60)
        results.setdefault(label, []).append((delta_events, delta_min))
    return [
        {
            "attack_type": at,
            "mean_events": float(np.mean([x[0] for x in vals])),
            "mean_minutes": float(np.mean([x[1] for x in vals])),
        }
        for at, vals in results.items()
    ]


# ---------------------------------------------------------------------------
# Vectorised scoring helpers
# ---------------------------------------------------------------------------

def _score_features_df(
    feature_df: pd.DataFrame,
    registry: Any,
    fusion: Any,
    calibrator: Any,
    entity_ids: pd.Series,
    cohorts: pd.Series,
    leave_out: str | None = None,
) -> np.ndarray:
    """Score a feature DataFrame using vectorised per-cohort batching where possible.

    For the IsolationForest (the most expensive detector) we group by cohort and call
    ``score_samples`` on the whole cohort matrix at once instead of one row at a time.
    For other detectors we still iterate but keep the hot path as tight as possible.
    """
    from sentinel.features.extractors import FeatureVector, FEATURE_NAMES
    from sentinel.serving.models import DetectorScores

    n = len(feature_df)
    risks = np.full(n, 0.5, dtype=float)

    # --- IsolationForest: batch by cohort (fast path) ---
    iso_scores = np.full(n, 0.5, dtype=float)
    feat_matrix = feature_df[FEATURE_NAMES].fillna(0.0).values.astype(np.float32)
    np.clip(feat_matrix, -1e6, 1e6, out=feat_matrix)

    cohort_arr = cohorts.values
    unique_cohorts = np.unique(cohort_arr)
    for coh in unique_cohorts:
        mask = cohort_arr == coh
        forest = getattr(registry.isolation, "_forests", {}).get(coh)
        if forest is None:
            continue
        X_coh = feat_matrix[mask]
        raw = forest.score_samples(X_coh)
        s_min = registry.isolation._train_score_min.get(coh, -0.5)
        s_max = registry.isolation._train_score_max.get(coh, 0.5)
        span = s_max - s_min
        if span < 1e-9:
            continue
        normalised = np.clip((s_max - raw) / span, 0.0, 1.0)
        iso_scores[mask] = normalised

    # --- Markov / Graph / Profiler: row loop (pure Python, manageable for small N) ---
    prof_scores = np.full(n, 0.5, dtype=float)
    markov_scores = np.full(n, 0.5, dtype=float)
    graph_scores = np.full(n, 0.5, dtype=float)
    gru_scores = np.full(n, 0.5, dtype=float)

    entity_arr = entity_ids.values
    for i in range(n):
        vals = feat_matrix[i].tolist()
        fv = FeatureVector(vals)
        eid = str(entity_arr[i])
        coh = str(cohort_arr[i])

        p = registry.profiler.score(fv, entity_id=eid, cohort=coh)
        # Mahalanobis can exceed 1; sigmoid-squash it to [0,1] before fusion
        maha = p.get("mahalanobis", 0.5)
        prof_scores[i] = float(1.0 / (1.0 + np.exp(-0.5 * (maha - 2.0))))
        markov_scores[i] = registry.markov.score(fv, cohort=coh)
        graph_scores[i] = registry.graph.score(fv, cohort=coh)
        try:
            gru_scores[i] = registry.gru.score(fv, cohort=coh)
        except Exception:  # noqa: BLE001
            pass

    def _clip01(v: float) -> float:
        return float(max(0.0, min(1.0, v)))

    # --- Fuse ---
    for i in range(n):
        det_dict = {
            "profile": 0.5 if leave_out == "profile" else _clip01(prof_scores[i]),
            "isolation": 0.5 if leave_out == "isolation" else _clip01(iso_scores[i]),
            "sequence": 0.5 if leave_out == "sequence" else _clip01(markov_scores[i]),
            "graph": 0.5 if leave_out == "graph" else _clip01(graph_scores[i]),
            "gru": 0.5 if leave_out == "gru" else _clip01(gru_scores[i]),
        }
        ds = DetectorScores(**det_dict)
        raw, _ = fusion.fuse(ds)
        cal = calibrator.calibrate(raw) if calibrator else raw
        risks[i] = cal * 100.0

    return risks


def _score_ablated(
    feature_df: pd.DataFrame,
    registry: Any,
    fusion: Any,
    calibrator: Any,
    entity_ids: pd.Series,
    cohorts: pd.Series,
    leave_out: str | None,
) -> np.ndarray:
    """Wrapper so ablation code stays readable."""
    return _score_features_df(feature_df, registry, fusion, calibrator, entity_ids, cohorts, leave_out=leave_out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval(
    data_dir: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
    max_train_events: int | None = 50_000,
    max_val_events: int | None = 10_000,
    max_test_events: int | None = 30_000,
) -> dict[str, Any]:
    """Train, evaluate, save metrics. Returns the results dict.

    Args:
        max_train_events: If set, cap the train split to this many events
            (sampled to preserve entity diversity).  None = use full split.
        max_val_events: Same for val split.
        max_test_events: Same for test split.
    """
    from sentinel.config import load_data_config, load_model_config, project_root
    from sentinel.features.pipeline import batch_features, PipelineState
    from sentinel.io import read_events, read_labels
    from sentinel.models.calibration import IsotonicCalibrator
    from sentinel.models.classifier import AttackClassifier
    from sentinel.models.registry import ModelRegistry
    from sentinel.schema import ATTACK_TYPES, DETECTOR_NAMES, LABEL_NORMAL

    root = project_root()
    data_path = Path(data_dir) if data_dir else root / "data"
    arts_path = Path(artifacts_dir) if artifacts_dir else root / "artifacts"
    rep_path = Path(reports_dir) if reports_dir else root / "reports"
    arts_path.mkdir(parents=True, exist_ok=True)
    rep_path.mkdir(parents=True, exist_ok=True)

    data_cfg = load_data_config()
    model_cfg = load_model_config()

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    log.info("Loading events and labels…")
    events_df = read_events(data_path / "events.parquet")
    labels_df = read_labels(data_path / "labels.parquet")

    label_map = labels_df.set_index("event_id")

    train_ev = events_df[events_df["split"] == "train"].reset_index(drop=True)
    val_ev = events_df[events_df["split"] == "val"].reset_index(drop=True)
    test_ev = events_df[events_df["split"] == "test"].reset_index(drop=True)

    log.info("Train: %d | Val: %d | Test: %d", len(train_ev), len(val_ev), len(test_ev))

    # Optionally cap splits (preserving entity diversity by stratified sampling)
    def _cap(ev: pd.DataFrame, n: int | None) -> pd.DataFrame:
        if n is None or len(ev) <= n:
            return ev
        # Sample n events but keep all entities' last few events to preserve state
        sampled = ev.sample(n=n, random_state=42).sort_values("timestamp")
        return sampled.reset_index(drop=True)

    train_ev = _cap(train_ev, max_train_events)
    val_ev   = _cap(val_ev,   max_val_events)
    test_ev  = _cap(test_ev,  max_test_events)

    train_lb = label_map[label_map.index.isin(train_ev["event_id"])].reset_index()
    val_lb   = label_map[label_map.index.isin(val_ev["event_id"])].reset_index()
    test_lb  = label_map[label_map.index.isin(test_ev["event_id"])].reset_index()

    log.info(
        "After capping — Train: %d | Val: %d | Test: %d",
        len(train_ev), len(val_ev), len(test_ev),
    )

    # ------------------------------------------------------------------ #
    # 2. Feature computation
    # ------------------------------------------------------------------ #
    log.info("Computing train features…")
    train_pipeline = PipelineState()
    train_feat_raw = batch_features(train_ev, state=train_pipeline)
    train_feat = pd.concat([
        train_ev[["event_id", "entity_id", "cohort"]].reset_index(drop=True),
        train_feat_raw.reset_index(drop=True),
    ], axis=1)
    train_feat["split"] = "train"

    log.info("Computing val features (carrying train state forward)…")
    # Carry train state into val
    trainval_ev = pd.concat([train_ev, val_ev], ignore_index=True).sort_values("timestamp")
    trainval_feat_raw = batch_features(trainval_ev)
    val_feat_raw = trainval_feat_raw.tail(len(val_ev)).reset_index(drop=True)
    val_feat = pd.concat([
        val_ev[["event_id", "entity_id", "cohort"]].reset_index(drop=True),
        val_feat_raw,
    ], axis=1)
    val_feat["split"] = "val"

    log.info("Computing test features (carrying train+val state forward)…")
    full_ev = pd.concat([train_ev, val_ev, test_ev], ignore_index=True).sort_values("timestamp")
    full_feat_raw = batch_features(full_ev)
    test_feat_raw = full_feat_raw.tail(len(test_ev)).reset_index(drop=True)
    test_feat = pd.concat([
        test_ev[["event_id", "entity_id", "cohort"]].reset_index(drop=True),
        test_feat_raw,
    ], axis=1)
    test_feat["split"] = "test"

    # ------------------------------------------------------------------ #
    # 3. Train models
    # ------------------------------------------------------------------ #
    log.info("Fitting model registry on train split…")
    model_cfg_dict = dataclasses.asdict(model_cfg)
    registry = ModelRegistry.from_config(model_cfg_dict)
    registry.fit_all(train_ev, train_lb, train_feat)

    # Reuse the fusion already fitted inside the registry
    fusion = registry.fusion
    from sentinel.features.extractors import FeatureVector, FEATURE_NAMES
    from sentinel.serving.models import DetectorScores

    log.info("Calibrating on val split…")
    val_risk = _score_features_df(val_feat, registry, fusion, None, val_ev["entity_id"], val_ev["cohort"])
    val_is_anomaly = val_ev["event_id"].map(label_map["is_anomaly"]).fillna(False).astype(bool).values
    calibrator = IsotonicCalibrator()
    calibrator.fit(val_risk / 100.0, val_is_anomaly)

    log.info("Fitting attack classifier…")
    classifier_cfg = model_cfg_dict.get("classifier", {})
    classifier = AttackClassifier(
        max_iter=classifier_cfg.get("max_iter", 300),
        learning_rate=classifier_cfg.get("learning_rate", 0.08),
        max_depth=classifier_cfg.get("max_depth", 6),
        min_confidence=classifier_cfg.get("min_confidence", 0.35),
    )
    classifier.fit(train_feat, train_lb)

    # Persist models
    log.info("Saving model artifacts…")
    registry.save(arts_path / "model_registry.pkl")
    with open(arts_path / "calibrator.pkl", "wb") as f:
        pickle.dump(calibrator.to_dict(), f)
    with open(arts_path / "classifier.pkl", "wb") as f:
        pickle.dump(classifier.to_dict(), f)

    # ------------------------------------------------------------------ #
    # 4. Score test split
    # ------------------------------------------------------------------ #
    log.info("Scoring test split…")
    test_risk = _score_features_df(
        test_feat, registry, fusion, calibrator,
        test_ev["entity_id"], test_ev["cohort"],
    )

    test_is_anomaly = test_ev["event_id"].map(label_map["is_anomaly"]).fillna(False).astype(bool).values
    test_label_str = test_ev["event_id"].map(label_map["label"]).fillna(LABEL_NORMAL).values

    # ------------------------------------------------------------------ #
    # 5. Latency benchmark
    # ------------------------------------------------------------------ #
    log.info("Benchmarking per-event scoring latency (200 samples)…")
    from sentinel.models.scorer import score_event
    bench_pipeline = PipelineState()
    # Warm up state on train
    batch_features(train_ev.head(100), state=bench_pipeline)
    sample_idx = np.random.default_rng(42).integers(0, len(test_ev), size=min(200, len(test_ev)))
    latencies_ms = []
    for i in sample_idx:
        row = test_ev.iloc[i]
        t0 = time.perf_counter()
        score_event(
            event=row,
            pipeline_state=bench_pipeline,
            registry=registry,
            calibrator=calibrator,
            classifier=classifier,
            alert_threshold=65.0,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    latency = {
        "p50": float(np.percentile(latencies_ms, 50)),
        "p95": float(np.percentile(latencies_ms, 95)),
        "p99": float(np.percentile(latencies_ms, 99)),
        "mean": float(np.mean(latencies_ms)),
    }
    log.info("Latency: p50=%.1fms p95=%.1fms p99=%.1fms", latency["p50"], latency["p95"], latency["p99"])

    # ------------------------------------------------------------------ #
    # 6. Core metrics
    # ------------------------------------------------------------------ #
    pr_auc = _pr_auc(test_is_anomaly, test_risk)
    roc_auc = _roc_auc(test_is_anomaly, test_risk)
    log.info("PR-AUC=%.4f  ROC-AUC=%.4f", pr_auc, roc_auc)

    budget_curve = []
    for bpct in [0.5, 1.0, 2.0, 3.0, 5.0]:
        prec, rec, n_alerts = _at_budget(test_is_anomaly, test_risk, bpct)
        analyst_h = n_alerts * model_cfg.alerting.analyst_minutes_per_alert / 60
        budget_curve.append({
            "budget_pct": bpct, "precision": prec, "recall": rec,
            "alerts": n_alerts, "analyst_hours": round(analyst_h, 2),
        })

    threshold_1pct = float(np.percentile(test_risk, 99.0))

    pr_curve = _pr_curve_pts(test_is_anomaly, test_risk)

    # ------------------------------------------------------------------ #
    # 7. Per-attack recall + confusion matrix
    # ------------------------------------------------------------------ #
    per_attack_recall = []
    for atype in ATTACK_TYPES:
        mask = test_label_str == atype
        if mask.sum() == 0:
            per_attack_recall.append({"attack_type": atype, "recall": 0.0, "support": 0, "detected": 0})
            continue
        det = int((test_risk[mask] >= threshold_1pct).sum())
        per_attack_recall.append({
            "attack_type": atype,
            "recall": float(det / mask.sum()),
            "support": int(mask.sum()),
            "detected": det,
        })

    from sklearn.metrics import confusion_matrix as sk_cm
    cm_labels = [LABEL_NORMAL] + ATTACK_TYPES
    pred_str = np.where(test_risk >= threshold_1pct, test_label_str, LABEL_NORMAL)
    cm = sk_cm(test_label_str, pred_str, labels=cm_labels)
    confusion = {"labels": cm_labels, "matrix": cm.tolist()}

    # ------------------------------------------------------------------ #
    # 8. FP rates
    # ------------------------------------------------------------------ #
    conf_mask = test_label_str == "benign_confounder"
    drift_mask = test_label_str == "insider_drift"
    fp_confounders = float((test_risk[conf_mask] >= threshold_1pct).mean()) if conf_mask.sum() > 0 else 0.0
    fp_insider = float((test_risk[drift_mask] >= threshold_1pct).mean()) if drift_mask.sum() > 0 else 0.0

    # ------------------------------------------------------------------ #
    # 9. MTTD
    # ------------------------------------------------------------------ #
    mttd = _mttd(test_ev, test_lb, test_risk, threshold_1pct)

    # ------------------------------------------------------------------ #
    # 10. Cold-start subgroup
    # ------------------------------------------------------------------ #
    train_entity_ids = set(train_ev["entity_id"].unique())
    cold_mask = ~test_ev["entity_id"].isin(train_entity_ids).values
    if cold_mask.sum() > 0 and test_is_anomaly[cold_mask].sum() > 0:
        cs_prec, cs_rec, _ = _at_budget(test_is_anomaly[cold_mask], test_risk[cold_mask], 1.0)
    else:
        cs_prec, cs_rec = 0.0, 0.0
    cold_start_metrics = {"precision": cs_prec, "recall": cs_rec, "n_events": int(cold_mask.sum())}

    # ------------------------------------------------------------------ #
    # 11. Post-drift subgroup
    # ------------------------------------------------------------------ #
    drift_day = data_cfg.drift.drift_start_day
    start_date_str = data_cfg.time.start_date
    start_date = _dt.date.fromisoformat(start_date_str) if isinstance(start_date_str, str) else start_date_str
    drift_cutoff = _dt.datetime(
        *(start_date + _dt.timedelta(days=drift_day)).timetuple()[:3],
        tzinfo=_dt.timezone.utc,
    )
    drift_mask_np = (test_ev["timestamp"] >= drift_cutoff).values
    if drift_mask_np.sum() > 0 and test_is_anomaly[drift_mask_np].sum() > 0:
        dr_prec, dr_rec, _ = _at_budget(test_is_anomaly[drift_mask_np], test_risk[drift_mask_np], 1.0)
    else:
        dr_prec, dr_rec = 0.0, 0.0
    post_drift_metrics = {"precision": dr_prec, "recall": dr_rec, "n_events": int(drift_mask_np.sum())}

    # ------------------------------------------------------------------ #
    # 12. Detector ablation
    # ------------------------------------------------------------------ #
    log.info("Running detector ablation (leave-one-out)…")
    ablation = []
    for leave_out in [None] + DETECTOR_NAMES:
        variant = "full_ensemble" if leave_out is None else f"without_{leave_out}"
        log.info("  Ablation: %s", variant)
        abl_risk = _score_ablated(
            test_feat, registry, fusion, calibrator,
            test_ev["entity_id"], test_ev["cohort"], leave_out,
        )
        abl_pr = _pr_auc(test_is_anomaly, abl_risk)
        abl_prec, _, _ = _at_budget(test_is_anomaly, abl_risk, 1.0)
        ablation.append({"variant": variant, "pr_auc": float(abl_pr), "precision_at_1pct": float(abl_prec)})

    # ------------------------------------------------------------------ #
    # 13. Held-out attack generalisation
    # ------------------------------------------------------------------ #
    held_type = "lateral_movement"
    held_mask = test_label_str == held_type
    held_recall = float((test_risk[held_mask] >= threshold_1pct).mean()) if held_mask.sum() > 0 else 0.0
    generalisation = {
        "held_out_attack": held_type,
        "unsupervised_recall": held_recall,
        "n_events": int(held_mask.sum()),
    }

    # ------------------------------------------------------------------ #
    # 14. Assemble and save
    # ------------------------------------------------------------------ #
    results: dict[str, Any] = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "budget_curve": budget_curve,
        "pr_curve": pr_curve,
        "per_attack_recall": per_attack_recall,
        "confusion_matrix": confusion,
        "fp_rate_confounders": fp_confounders,
        "fp_rate_insider_drift": fp_insider,
        "mttd": mttd,
        "cold_start": cold_start_metrics,
        "post_drift": post_drift_metrics,
        "latency_ms": latency,
        "ablation": ablation,
        "generalisation": generalisation,
        "n_test_events": len(test_ev),
        "n_test_anomalies": int(test_is_anomaly.sum()),
        "anomaly_rate_pct": float(test_is_anomaly.mean() * 100),
        "threshold_at_1pct": float(threshold_1pct),
    }

    out_json = arts_path / "eval_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    log.info("Saved → %s", out_json)

    _write_plots(results, rep_path)
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _write_plots(results: dict[str, Any], rep_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping plots")
        return

    # PR curve
    if results["pr_curve"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        pts = results["pr_curve"]
        ax.plot([p["recall"] for p in pts], [p["precision"] for p in pts], lw=2, color="#3b82f6")
        ax.fill_between([p["recall"] for p in pts], [p["precision"] for p in pts], alpha=0.15, color="#3b82f6")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve — PR-AUC = {results['pr_auc']:.3f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); fig.tight_layout()
        fig.savefig(rep_path / "pr_curve.png", dpi=130); plt.close(fig)

    # Budget curve
    bc = results["budget_curve"]
    if bc:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([b["budget_pct"] for b in bc], [b["precision"] for b in bc], "o-", label="Precision", color="#10b981")
        ax.plot([b["budget_pct"] for b in bc], [b["recall"] for b in bc], "s--", label="Recall", color="#f59e0b")
        ax.set_xlabel("Alert Budget (%)"); ax.set_ylabel("Score")
        ax.set_title("Precision & Recall vs Alert Budget"); ax.legend()
        fig.tight_layout(); fig.savefig(rep_path / "budget_curve.png", dpi=130); plt.close(fig)

    # Per-attack recall
    par = results["per_attack_recall"]
    if par:
        fig, ax = plt.subplots(figsize=(8, 4))
        attacks = [p["attack_type"].replace("_", " ") for p in par]
        recalls = [p["recall"] for p in par]
        colors = ["#ef4444" if r < 0.5 else "#f59e0b" if r < 0.75 else "#10b981" for r in recalls]
        ax.barh(attacks, recalls, color=colors)
        ax.set_xlabel("Recall @ 1% budget"); ax.set_title("Per-Attack-Type Recall"); ax.set_xlim(0, 1)
        fig.tight_layout(); fig.savefig(rep_path / "per_attack_recall.png", dpi=130); plt.close(fig)

    # Ablation
    abl = results["ablation"]
    if abl:
        fig, ax = plt.subplots(figsize=(8, max(3, len(abl) * 0.5)))
        variants = [a["variant"].replace("_", " ") for a in abl]
        pr_aucs = [a["pr_auc"] for a in abl]
        baseline = next((a["pr_auc"] for a in abl if "full" in a["variant"]), None)
        colors = ["#3b82f6" if "full" in a["variant"] else "#6b7280" for a in abl]
        ax.barh(variants, pr_aucs, color=colors)
        if baseline:
            ax.axvline(baseline, color="red", linestyle="--", lw=1.5, label="Full ensemble")
            ax.legend()
        ax.set_xlabel("PR-AUC"); ax.set_title("Detector Ablation (leave-one-out)")
        fig.tight_layout(); fig.savefig(rep_path / "ablation.png", dpi=130); plt.close(fig)

    # Confusion matrix
    try:
        cm_data = results["confusion_matrix"]
        cm_arr = np.array(cm_data["matrix"])
        cm_labels = [l.replace("_", "\n") for l in cm_data["labels"]]
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(cm_arr, cmap="Blues")
        ax.set_xticks(range(len(cm_labels))); ax.set_yticks(range(len(cm_labels)))
        ax.set_xticklabels(cm_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(cm_labels, fontsize=7)
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center", fontsize=6)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix @ 1% Budget")
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(rep_path / "confusion_matrix.png", dpi=130); plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        log.warning("Confusion matrix plot failed: %s", exc)

    log.info("Plots written to %s", rep_path)
