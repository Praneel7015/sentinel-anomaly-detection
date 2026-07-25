"""Async stream replay driver for the SENTINEL SSE endpoint.

Loads ``data/events.parquet``, replays events through the scorer (or stub),
and pushes :class:`~sentinel.serving.models.ScoredEvent` objects onto an
``asyncio.Queue`` that the ``GET /api/stream`` SSE handler drains.

Speed semantics
---------------
``speed=1.0``  → real-time: respects original inter-event timestamps.
``speed=100``  → 100× faster than real-time.
``speed=0``    → as fast as possible (no sleep).

Control
-------
Pause / resume / reset are handled via three ``asyncio.Event`` flags.
The driver loop checks these flags before processing each event so it is
responsive to control signals without blocking.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Awaitable

if TYPE_CHECKING:
    from sentinel.serving.models import ScoredEvent

logger = logging.getLogger(__name__)

__all__ = ["StreamReplay", "ReplayState"]

ReplayState = str  # "running" | "paused" | "stopped"


class StreamReplay:
    """Async stream replay driver.

    Parameters
    ----------
    score_fn:
        ``async (event_dict) -> ScoredEvent`` callable.  The app layer injects
        its stub or real scorer so this module stays scorer-agnostic.
    data_dir:
        Directory containing ``events.parquet``.
    queue:
        The :class:`asyncio.Queue` the SSE endpoint reads from.
    default_speed:
        Initial replay speed multiplier.
    """

    def __init__(
        self,
        score_fn: Callable[[dict], Awaitable[ScoredEvent]],
        data_dir: Path,
        queue: asyncio.Queue[ScoredEvent],
        default_speed: float = 50.0,
        auto_start_task: bool = True,
    ) -> None:
        self._score_fn = score_fn
        self._data_dir = data_dir
        self._queue = queue
        self._speed = max(0.0, default_speed)
        self._auto_start_task = auto_start_task

        # Control flags
        self._running = asyncio.Event()     # set = allowed to run
        self._reset_requested = asyncio.Event()

        self._state: ReplayState = "stopped"
        self._position: int = 0
        self._total: int = 0
        self._task: asyncio.Task | None = None

    # -- public API ---------------------------------------------------------- #

    @property
    def state(self) -> ReplayState:
        return self._state

    @property
    def position(self) -> int:
        return self._position

    @property
    def total(self) -> int:
        return self._total

    def set_speed(self, speed: float) -> None:
        self._speed = max(0.0, speed)

    def start(self) -> None:
        """Start or resume the replay."""
        if self._state == "stopped":
            self._state = "running"
            self._running.set()
            if self._auto_start_task and (self._task is None or self._task.done()):
                try:
                    loop = asyncio.get_running_loop()
                    self._task = loop.create_task(self._run())
                except RuntimeError:
                    logger.debug("StreamReplay.start: no running event loop, skipping task")
        elif self._state == "paused":
            self._state = "running"
            self._running.set()

    def pause(self) -> None:
        if self._state == "running":
            self._state = "paused"
            self._running.clear()

    def reset(self) -> None:
        self._state = "stopped"
        self._running.clear()
        self._reset_requested.set()
        self._position = 0

    # -- internals ----------------------------------------------------------- #

    async def _load_events(self) -> list[dict]:
        """Load ``events.parquet`` from disk.

        Returns an empty list if the file does not exist (common in test mode).
        When the file exists, loading is done in a thread-pool executor to avoid
        blocking the event loop during disk I/O.
        """
        parquet_path = self._data_dir / "events.parquet"

        # Short-circuit: if file doesn't exist don't even bother with the executor
        if not parquet_path.exists():
            logger.warning("events.parquet not found at %s; stream will be empty", parquet_path)
            return []

        def _read() -> list[dict]:
            try:
                import pandas as pd  # noqa: PLC0415
                df = pd.read_parquet(parquet_path)
                df = df.sort_values("timestamp")
                return df.to_dict(orient="records")
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to load events.parquet: %s", exc)
                return []

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _read)

    async def _run(self) -> None:
        """Main replay loop."""
        logger.info("StreamReplay: loading events…")
        events = await self._load_events()
        self._total = len(events)
        self._position = 0
        logger.info("StreamReplay: %d events loaded, speed=%.1fx", self._total, self._speed)

        prev_event_ts: float | None = None

        while self._position < self._total:
            # Handle reset
            if self._reset_requested.is_set():
                self._reset_requested.clear()
                self._position = 0
                self._state = "stopped"
                logger.info("StreamReplay: reset")
                return

            # Pause: wait until running again
            if not self._running.is_set():
                await self._running.wait()
                # After resume, skip sleeping for inter-event gap (avoid huge catch-up)
                prev_event_ts = None
                if self._reset_requested.is_set():
                    continue

            row = events[self._position]

            # Inter-event timing (real-time mode)
            if self._speed > 0 and prev_event_ts is not None:
                try:
                    import pandas as pd  # noqa: PLC0415
                    ts_val = row.get("timestamp")
                    if hasattr(ts_val, "timestamp"):
                        cur_ts = ts_val.timestamp()
                    elif isinstance(ts_val, pd.Timestamp):
                        cur_ts = ts_val.timestamp()
                    else:
                        cur_ts = float(ts_val) / 1e9
                    gap = (cur_ts - prev_event_ts) / self._speed
                    if gap > 0:
                        await asyncio.sleep(min(gap, 1.0))  # cap at 1s so control is responsive
                    prev_event_ts = cur_ts
                except Exception:  # noqa: BLE001
                    prev_event_ts = None
            elif self._speed > 0 and prev_event_ts is None:
                try:
                    ts_val = row.get("timestamp")
                    if hasattr(ts_val, "timestamp"):
                        prev_event_ts = ts_val.timestamp()
                except Exception:  # noqa: BLE001
                    pass

            # Score the event
            try:
                scored = await self._score_fn(row)
                await self._queue.put(scored)
            except Exception as exc:  # noqa: BLE001
                logger.debug("StreamReplay: score error at position %d: %s", self._position, exc)

            self._position += 1
            # Yield to the event loop every 50 events even at max speed
            if self._position % 50 == 0:
                await asyncio.sleep(0)

        logger.info("StreamReplay: finished all %d events", self._total)
        self._state = "stopped"
