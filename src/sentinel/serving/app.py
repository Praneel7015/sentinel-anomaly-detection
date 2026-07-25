"""FastAPI application for the SENTINEL scoring and alert service.

Implements every endpoint in ``docs/CONTRACTS.md § 3``.

Scorer import strategy
----------------------
This module tries to import the real scorer from ``sentinel.models.scorer``.
If those modules are not yet fitted / installed, it falls back silently to a
**stub scorer** that returns plausible :class:`~sentinel.serving.models.ScoredEvent`
objects so the entire API is runnable and testable without a trained model.

TODO: swap stub when models are ready — change the ``try`` block below.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from sentinel import __version__
from sentinel.schema import ATTACK_TYPES, AccessEvent, risk_band
from sentinel.serving.models import (
    AlertDetailResponse,
    AlertSort,
    AlertsResponse,
    AblationRow,
    BudgetPoint,
    ConfusionMatrix,
    Contribution,
    Counterfactual,
    DetectorScores,
    DriftState,
    EntityDetailResponse,
    EntityListResponse,
    EntitySummary,
    FeedbackRequest,
    FeedbackResponse,
    GroundTruth,
    HealthResponse,
    LatencyStats,
    MetricsResponse,
    MttdEntry,
    PeerComparison,
    PerAttackRecall,
    ProfileSummaryItem,
    PrPoint,
    ResourceUsage,
    RiskTimelinePoint,
    ScoredEvent,
    StatsResponse,
    StreamControlRequest,
    StreamControlResponse,
    SubgroupMetrics,
)
from sentinel.serving.replay import StreamReplay
from sentinel.serving.store import AlertStore, EntityStore, FeedbackStore, StatsTracker

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Scorer: try the real stack, fall back to stub
# --------------------------------------------------------------------------- #

_MODEL_LOADED = False
_TORCH_AVAILABLE = False

try:
    import torch as _torch  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_pipeline_state = None
_model_registry = None

# Only attempt model import if the registry artifact actually exists —
# this avoids slow/hanging torch imports in stub mode.
_registry_path_check = Path("artifacts") / "model_registry.pkl"
if _registry_path_check.exists():
    try:
        # TODO: swap stub when models are ready — load artifacts here once
        #       `sentinel train` has been run and artifacts exist.
        from sentinel.models.scorer import score_event as _real_score_event  # noqa: F401
        from sentinel.models.registry import ModelRegistry  # noqa: F401

        import joblib  # type: ignore[import]
        _model_registry = joblib.load(_registry_path_check)
        _MODEL_LOADED = True
        logger.info("Loaded real model registry from %s", _registry_path_check)
    except Exception as _exc:  # noqa: BLE001
        logger.warning("Failed to load model registry: %s; running in stub mode", _exc)
else:
    # TODO: swap stub when models are ready — remove this branch once
    #       sentinel.models.scorer exists and artifacts are fitted.
    logger.warning("No model_registry.pkl found; running in stub mode (scores are synthetic)")


def _stub_score_event(event: AccessEvent) -> ScoredEvent:
    """Deterministic-ish stub scorer returning a plausible ScoredEvent.

    Uses the event_id to seed randomness so repeated calls for the same
    event return the same score (idempotent within a process).
    """
    try:
        seed = int(event.event_id[:8], 16)
    except (ValueError, TypeError):
        seed = hash(event.event_id) & 0xFFFFFFFF
    rng = random.Random(seed)
    risk = round(rng.uniform(5.0, 95.0), 2)
    band = risk_band(risk)
    attack = rng.choice(ATTACK_TYPES + ["normal"])
    conf = round(rng.uniform(0.3, 0.99), 3)
    det = DetectorScores(
        profile=round(rng.uniform(0.0, 1.0), 3),
        isolation=round(rng.uniform(0.0, 1.0), 3),
        sequence=round(rng.uniform(0.0, 1.0), 3),
        graph=round(rng.uniform(0.0, 1.0), 3),
        gru=None,
    )
    contributions = [
        Contribution(
            feature="stub_feature",
            display_name="Stub feature",
            value=0.5,
            display_value="0.50",
            contribution=rng.uniform(-1.0, 1.0),
            direction="increases",
            description="Stub scorer placeholder",
        )
    ]
    counterfactuals = [
        Counterfactual(
            feature="stub_feature",
            display_name="Stub feature",
            neutralised_risk=max(0.0, risk - 10.0),
            delta=-10.0,
        )
    ]
    # Default threshold: medium band = 40
    is_alert = risk >= _alert_threshold()
    return ScoredEvent(
        event_id=event.event_id,
        entity_id=event.entity_id,
        entity_type=event.entity_type,
        cohort=event.cohort,
        timestamp=event.timestamp,
        risk_score=risk,
        risk_band=band,
        is_alert=is_alert,
        predicted_attack_type=attack,
        attack_type_confidence=conf,
        classifier_agreement=True,
        is_novel=False,
        detector_scores=det,
        contributions=contributions,
        narrative=(
            f"[STUB] Entity {event.entity_id} scored {risk:.1f} risk. "
            "No trained model is loaded; this score is synthetic."
        ),
        counterfactuals=counterfactuals,
        cold_start=_entity_store.get(event.entity_id) is None,
        entity_event_count=_entity_store.get(event.entity_id).event_count if _entity_store.get(event.entity_id) else 0,
        event=event,
        ground_truth=None,
    )


def _score_event(event: AccessEvent) -> ScoredEvent:
    """Route to real or stub scorer."""
    if _MODEL_LOADED and _model_registry is not None:
        # TODO: swap stub when models are ready — inject real pipeline_state
        # return _real_score_event(event, pipeline_state, _model_registry)
        pass
    return _stub_score_event(event)


async def _async_score_event(row: dict) -> ScoredEvent:
    """Async wrapper for the stream replay driver."""
    # Convert raw parquet row dict to AccessEvent (best-effort)
    try:
        # Normalise timestamp
        ts = row.get("timestamp")
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if ts is not None and hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        row = dict(row)
        row["timestamp"] = ts
        # command_sequence may come as numpy array
        cs = row.get("command_sequence", [])
        if hasattr(cs, "tolist"):
            cs = cs.tolist()
        row["command_sequence"] = list(cs) if cs is not None else []
        event = AccessEvent(**row)
    except Exception as exc:  # noqa: BLE001
        logger.debug("StreamReplay: event parse error: %s", exc)
        raise

    loop = asyncio.get_event_loop()
    scored = await loop.run_in_executor(None, _score_event, event)
    _alert_store.append(scored)
    _entity_store.update(scored)
    _stats.record_event(scored)
    _stats.set_stream_state(_replay.position, _replay.total)
    return scored


# --------------------------------------------------------------------------- #
# Global singletons
# --------------------------------------------------------------------------- #

_ARTIFACTS_DIR = Path(os.environ.get("SENTINEL_ARTIFACTS_DIR", "artifacts"))
_DATA_DIR = Path(os.environ.get("SENTINEL_DATA_DIR", "data"))
_DEFAULT_BUDGET_PCT: float = 1.0  # loaded from config if available
_BUDGET_THRESHOLDS: dict[float, float] = {}  # budget_pct -> risk threshold

_alert_store = AlertStore(max_size=50_000)
_entity_store = EntityStore()
_stats = StatsTracker()
_feedback_store = FeedbackStore()
_sse_queue: asyncio.Queue[ScoredEvent] = asyncio.Queue(maxsize=2000)

_replay = StreamReplay(
    score_fn=_async_score_event,
    data_dir=_DATA_DIR,
    queue=_sse_queue,
    default_speed=50.0,
    auto_start_task=False,  # enabled only when running under uvicorn
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _alert_threshold(budget_pct: float | None = None) -> float:
    """Return the risk threshold for the given budget percentage."""
    pct = budget_pct if budget_pct is not None else _DEFAULT_BUDGET_PCT
    if pct in _BUDGET_THRESHOLDS:
        return _BUDGET_THRESHOLDS[pct]
    # Fallback: map 0.5% -> 65, 1% -> 50, 2% -> 40
    _defaults: dict[float, float] = {0.5: 65.0, 1.0: 50.0, 2.0: 40.0}
    return _defaults.get(pct, 50.0)


def _load_budget_thresholds() -> None:
    """Load thresholds from artifacts/thresholds.json if present."""
    global _BUDGET_THRESHOLDS, _DEFAULT_BUDGET_PCT
    path = _ARTIFACTS_DIR / "thresholds.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            _BUDGET_THRESHOLDS = {float(k): float(v) for k, v in data.items()}
            logger.info("Loaded budget thresholds from %s", path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load thresholds.json: %s", exc)


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def _lifespan(application: FastAPI):  # noqa: ARG001
    _load_budget_thresholds()
    yield


app = FastAPI(
    title="SENTINEL",
    version=__version__,
    description="Explainable behavioural anomaly detection — REST + SSE API",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=_lifespan,
)

_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:4173",
    "https://sentinel-soc.vercel.app",
    "https://sentinel-anomaly-detection-phi.vercel.app",
    *[o.strip() for o in os.environ.get("SENTINEL_CORS_ORIGINS", "").split(",") if o.strip()],
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api")


# --------------------------------------------------------------------------- #
# GET /api/health
# --------------------------------------------------------------------------- #


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _MODEL_LOADED else "degraded",
        version=__version__,
        model_loaded=_MODEL_LOADED,
        torch_available=_TORCH_AVAILABLE,
    )


# --------------------------------------------------------------------------- #
# GET /api/stats
# --------------------------------------------------------------------------- #


@router.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    snap = _stats.snapshot()
    return StatsResponse(**snap)


# --------------------------------------------------------------------------- #
# POST /api/score
# --------------------------------------------------------------------------- #


@router.post("/score", response_model=ScoredEvent)
async def score(event: AccessEvent) -> ScoredEvent:
    loop = asyncio.get_event_loop()
    t0 = time.monotonic()
    scored = await loop.run_in_executor(None, _score_event, event)
    _alert_store.append(scored)
    _entity_store.update(scored)
    _stats.record_event(scored)
    return scored


# --------------------------------------------------------------------------- #
# GET /api/alerts
# --------------------------------------------------------------------------- #


@router.get("/alerts", response_model=AlertsResponse)
async def alerts(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    min_risk: float | None = Query(default=None, ge=0.0, le=100.0),
    attack_type: str | None = Query(default=None),
    entity_id: str | None = Query(default=None),
    budget_pct: float | None = Query(default=None),
    sort: AlertSort = Query(default="risk_desc"),
) -> AlertsResponse:
    threshold: float | None = None
    if budget_pct is not None:
        threshold = _alert_threshold(budget_pct)

    page, total = _alert_store.query(
        limit=limit,
        offset=offset,
        min_risk=min_risk,
        attack_type=attack_type,
        entity_id=entity_id,
        threshold=threshold,
        sort=sort,
    )
    return AlertsResponse(alerts=page, total=total)


# --------------------------------------------------------------------------- #
# GET /api/alerts/{event_id}
# --------------------------------------------------------------------------- #


@router.get("/alerts/{event_id}", response_model=AlertDetailResponse)
async def alert_detail(event_id: str) -> AlertDetailResponse:
    scored = _alert_store.get(event_id)
    if scored is None:
        raise HTTPException(status_code=404, detail=f"Alert {event_id!r} not found")

    # entity summary
    rec = _entity_store.get(scored.entity_id)
    if rec is not None:
        entity_summary = EntitySummary(
            entity_id=rec.entity_id,
            entity_type=rec.entity_type,
            cohort=rec.cohort,
            event_count=rec.event_count,
            cold_start=rec.cold_start,
            first_seen=rec.first_seen,
            last_seen=rec.last_seen,
            alert_count=rec.alert_count,
            mean_risk=round(rec.mean_risk, 2),
        )
    else:
        entity_summary = EntitySummary(
            entity_id=scored.entity_id,
            entity_type=scored.entity_type,
            cohort=scored.cohort,
            event_count=scored.entity_event_count,
            cold_start=scored.cold_start,
        )

    similar = _alert_store.similar(scored)

    return AlertDetailResponse(
        **scored.model_dump(),
        entity_summary=entity_summary,
        similar_alerts=similar,
    )


# --------------------------------------------------------------------------- #
# GET /api/entities  (list)
# --------------------------------------------------------------------------- #


@router.get("/entities", response_model=EntityListResponse)
async def entity_list(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    sort: str = Query(default="risk_desc", pattern="^(risk_desc|risk_asc|name_asc|event_desc)$"),
) -> EntityListResponse:
    records = _entity_store.list_all()
    # sort
    if sort == "risk_desc":
        records.sort(key=lambda r: r.mean_risk, reverse=True)
    elif sort == "risk_asc":
        records.sort(key=lambda r: r.mean_risk)
    elif sort == "name_asc":
        records.sort(key=lambda r: r.entity_id)
    elif sort == "event_desc":
        records.sort(key=lambda r: r.event_count, reverse=True)

    total = len(records)
    page = records[offset : offset + limit]
    summaries = [
        EntitySummary(
            entity_id=r.entity_id,
            entity_type=r.entity_type,
            cohort=r.cohort,
            event_count=r.event_count,
            cold_start=r.cold_start,
            first_seen=r.first_seen,
            last_seen=r.last_seen,
            alert_count=r.alert_count,
            mean_risk=round(r.mean_risk, 2),
            max_risk=round(max((rs for _, rs, _ in r.risk_timeline), default=0.0), 2),
        )
        for r in page
    ]
    return EntityListResponse(entities=summaries, total=total)


# --------------------------------------------------------------------------- #
# GET /api/entities/{entity_id}
# --------------------------------------------------------------------------- #


@router.get("/entities/{entity_id}", response_model=EntityDetailResponse)
async def entity_detail(entity_id: str) -> EntityDetailResponse:
    rec = _entity_store.get(entity_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id!r} not found")

    # risk timeline (cap at 200 most recent)
    timeline = [
        RiskTimelinePoint(timestamp=ts, risk_score=rs, is_alert=ia)
        for ts, rs, ia in rec.risk_timeline[-200:]
    ]

    # top resources (top 10 by count, mark is_new if first_seen is recent)
    sorted_res = sorted(rec.resource_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    last_seen_limit = rec.last_seen
    top_resources = [
        ResourceUsage(
            resource=r,
            count=c,
            is_new=(rec.resource_first_seen.get(r, rec.first_seen) >= last_seen_limit),
        )
        for r, c in sorted_res
    ]

    # profile summary (simple stub since full profiler is in models/)
    profile_summary = [
        ProfileSummaryItem(
            label="event_count",
            value=float(rec.event_count),
            cohort_value=float(rec.event_count),
        ),
        ProfileSummaryItem(
            label="mean_risk",
            value=round(rec.mean_risk, 2),
            cohort_value=50.0,
        ),
    ]

    # peer comparison stubs
    peer_comparison = [
        PeerComparison(axis="mean_risk", entity=round(rec.mean_risk, 2), cohort_median=50.0),
        PeerComparison(
            axis="alert_rate",
            entity=round(rec.alert_count / max(rec.event_count, 1), 3),
            cohort_median=0.01,
        ),
    ]

    drift_state = DriftState(
        drifting=rec.drifting,
        detected_at=rec.drift_detected_at,
        adapted=rec.drift_adapted,
    )

    return EntityDetailResponse(
        entity_id=rec.entity_id,
        entity_type=rec.entity_type,
        cohort=rec.cohort,
        first_seen=rec.first_seen,
        last_seen=rec.last_seen,
        event_count=rec.event_count,
        cold_start=rec.cold_start,
        profile_summary=profile_summary,
        risk_timeline=timeline,
        activity_by_hour=rec.activity_by_hour,
        top_resources=top_resources,
        peer_comparison=peer_comparison,
        drift_state=drift_state,
    )


# --------------------------------------------------------------------------- #
# POST /api/feedback
# --------------------------------------------------------------------------- #


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest) -> FeedbackResponse:
    scored = _alert_store.get(req.event_id)
    entity_id = scored.entity_id if scored is not None else None

    updated_threshold = _feedback_store.add(
        event_id=req.event_id,
        verdict=req.verdict,
        note=req.note,
        entity_id=entity_id,
        alert_store=_alert_store,
    )
    return FeedbackResponse(ok=True, updated_threshold=updated_threshold)


# --------------------------------------------------------------------------- #
# GET /api/metrics
# --------------------------------------------------------------------------- #


@router.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    eval_path = _ARTIFACTS_DIR / "eval_results.json"
    if eval_path.exists():
        try:
            data = json.loads(eval_path.read_text())
            return MetricsResponse(**data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse eval_results.json: %s", exc)

    # Return zeros with a warning note when eval has not been run
    empty_matrix = ConfusionMatrix(labels=["normal", *ATTACK_TYPES], matrix=[[0] * 7 for _ in range(7)])
    return MetricsResponse(
        pr_auc=0.0,
        roc_auc=0.0,
        budget_curve=[
            BudgetPoint(budget_pct=p, precision=0.0, recall=0.0, alerts=0, analyst_hours=0.0)
            for p in (0.5, 1.0, 2.0)
        ],
        per_attack_recall=[
            PerAttackRecall(attack_type=at, recall=0.0, support=0, detected=0)
            for at in ATTACK_TYPES
        ],
        confusion_matrix=empty_matrix,
        pr_curve=[PrPoint(recall=0.0, precision=0.0)],
        fp_rate_confounders=0.0,
        fp_rate_insider_drift=0.0,
        mttd=[MttdEntry(attack_type=at, mean_events=0.0, mean_minutes=0.0) for at in ATTACK_TYPES],
        cold_start=SubgroupMetrics(precision=0.0, recall=0.0),
        post_drift=SubgroupMetrics(precision=0.0, recall=0.0),
        latency_ms=LatencyStats(p50=0.0, p95=0.0, p99=0.0, mean=0.0),
        ablation=[AblationRow(variant="full", pr_auc=0.0, precision_at_1pct=0.0)],
    )


# --------------------------------------------------------------------------- #
# GET /api/stream  (SSE)
# --------------------------------------------------------------------------- #

_SSE_HEARTBEAT_S: float = float(os.environ.get("SENTINEL_SSE_HEARTBEAT_S", "15"))
_SSE_STATS_INTERVAL_S: float = 5.0


async def _event_generator() -> AsyncIterator[str]:
    """Generate SSE frames from the queue plus periodic stats and heartbeats."""
    last_stats = time.monotonic()
    last_heartbeat = time.monotonic()

    while True:
        now = time.monotonic()

        # Drain any queued alerts
        try:
            scored: ScoredEvent = _sse_queue.get_nowait()
            if scored.is_alert:
                payload = scored.model_dump_json()
                yield f"event: alert\ndata: {payload}\n\n"
            _sse_queue.task_done()
        except asyncio.QueueEmpty:
            pass

        # Stats every 5 seconds
        if now - last_stats >= _SSE_STATS_INTERVAL_S:
            snap = _stats.snapshot()
            stats_json = json.dumps(snap)
            yield f"event: stats\ndata: {stats_json}\n\n"
            last_stats = now

        # Heartbeat
        if now - last_heartbeat >= _SSE_HEARTBEAT_S:
            yield "event: heartbeat\ndata: {}\n\n"
            last_heartbeat = now

        await asyncio.sleep(0.05)


@router.get("/stream")
async def stream() -> StreamingResponse:
    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# --------------------------------------------------------------------------- #
# POST /api/stream/control
# --------------------------------------------------------------------------- #


@router.post("/stream/control", response_model=StreamControlResponse)
async def stream_control(req: StreamControlRequest) -> StreamControlResponse:
    if req.speed is not None:
        _replay.set_speed(req.speed)

    if req.action == "start":
        _replay.start()
    elif req.action == "pause":
        _replay.pause()
    elif req.action == "reset":
        _replay.reset()

    return StreamControlResponse(ok=True, state=_replay.state)


# --------------------------------------------------------------------------- #
# Mount router
# --------------------------------------------------------------------------- #

app.include_router(router)


# Redirect root to API docs for convenience
from fastapi.responses import RedirectResponse  # noqa: E402

@app.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/api/docs")


# --------------------------------------------------------------------------- #
# Programmatic entry point (used by cli.py)
# --------------------------------------------------------------------------- #


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
    artifacts_dir: Path | None = None,
    data_dir: Path | None = None,
) -> None:  # pragma: no cover
    """Launch uvicorn programmatically (called from ``sentinel serve``)."""
    global _ARTIFACTS_DIR, _DATA_DIR, _replay

    if artifacts_dir is not None:
        _ARTIFACTS_DIR = artifacts_dir
    if data_dir is not None:
        _DATA_DIR = data_dir
        _replay = StreamReplay(
            score_fn=_async_score_event,
            data_dir=_DATA_DIR,
            queue=_sse_queue,
            default_speed=50.0,
            auto_start_task=True,  # real server — background task is safe
        )

    _load_budget_thresholds()

    uvicorn.run(
        "sentinel.serving.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
