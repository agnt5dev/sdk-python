"""``ctx.progress``: report how far a run has got.

A report is a ``progress.update`` record in the run's journal. Studio and
``get_run`` show the latest one, and an MCP client that called the run's
tool with a progress token hears each one as ``notifications/progress``.

Reports are side-band: nothing reads them back into the run, so they can't
change what a workflow does when it replays. They are cheap to make often:
each run writes at most one record a second, always the latest report, and
sends it without holding up the run. Progress never goes backwards: a
report below the last one is dropped, as MCP requires.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

PROGRESS_EVENT_TYPE = "progress.update"

#: The shortest time between two progress records of one run.
MIN_INTERVAL_SECONDS = 1.0

#: The longest message a report keeps.
MAX_MESSAGE_CHARS = 1000


def _number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"ctx.progress: {name} must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"ctx.progress: {name} must be finite, got {value!r}")
    return number


def progress_payload(
    progress: float, total: Optional[float] = None, message: Optional[str] = None
) -> dict[str, Any]:
    """Check a report and return what its journal record carries.

    Raises ``TypeError`` for a non-number ``progress`` or ``total`` or a
    non-string ``message``, and ``ValueError`` for a value that isn't finite
    or a ``total`` that isn't positive.
    """
    payload: dict[str, Any] = {"progress": _number("progress", progress)}
    if total is not None:
        payload["total"] = _number("total", total)
        if payload["total"] <= 0:
            raise ValueError(f"ctx.progress: total must be positive, got {total!r}")
    if message is not None:
        if not isinstance(message, str):
            raise TypeError(
                f"ctx.progress: message must be a string, got {type(message).__name__}"
            )
        if message:
            payload["message"] = message[:MAX_MESSAGE_CHARS]
    return payload


class ProgressReporter:
    """Coalesces one run's reports into at most one record per interval.

    ``report`` never blocks and may be called from any thread (a sync
    function runs in a worker thread). The first report goes out at once;
    later ones wait out the interval and only the latest is sent. ``close``
    drops anything not yet sent: once a run has finished, a report would
    only say something stale.
    """

    def __init__(
        self,
        send: Callable[[dict[str, Any]], Awaitable[None]],
        *,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        interval: Optional[float] = None,
    ) -> None:
        self._send = send
        self._loop = loop
        self._interval = MIN_INTERVAL_SECONDS if interval is None else interval
        self._lock = threading.Lock()
        self._last: Optional[dict[str, Any]] = None
        self._pending: Optional[dict[str, Any]] = None
        self._scheduled = False
        self._closed = False
        self._next_at = 0.0
        self._task: Optional[asyncio.Task[None]] = None

    @property
    def closed(self) -> bool:
        return self._closed

    def report(self, payload: dict[str, Any]) -> bool:
        """Queue a report. False when it was dropped: it went backwards,
        repeated the last one, or the run has finished."""
        with self._lock:
            if self._closed or not self._moves_on(payload):
                return False
            self._last = payload
            self._pending = payload
            if self._scheduled:
                return True
            loop = self._loop_for_caller()
            if loop is None or loop.is_closed():
                # Nowhere to send from; keep the latest for the next try.
                return True
            self._scheduled = True
        self._start(loop)
        return True

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._pending = None
        task = self._task
        if task is not None and not task.done():
            loop = task.get_loop()
            try:
                if _running_loop() is loop:
                    task.cancel()
                else:
                    loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass

    def _moves_on(self, payload: dict[str, Any]) -> bool:
        last = self._last
        if last is None:
            return True
        if payload["progress"] < last["progress"]:
            return False
        return payload != last

    def _loop_for_caller(self) -> Optional[asyncio.AbstractEventLoop]:
        running = _running_loop()
        if self._loop is None:
            self._loop = running
        return self._loop

    def _start(self, loop: asyncio.AbstractEventLoop) -> None:
        def start() -> None:
            self._task = loop.create_task(self._pump())

        try:
            if _running_loop() is loop:
                start()
            else:
                loop.call_soon_threadsafe(start)
        except RuntimeError:
            # The loop closed under us: the run is over.
            with self._lock:
                self._scheduled = False

    async def _pump(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while True:
                delay = self._next_at - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                with self._lock:
                    payload, self._pending = self._pending, None
                    if payload is None or self._closed:
                        self._scheduled = False
                        return
                self._next_at = loop.time() + self._interval
                try:
                    await self._send(payload)
                except Exception:
                    # Progress is best effort: never fail the run over it.
                    logger.debug("ctx.progress: report not delivered", exc_info=True)
        finally:
            with self._lock:
                self._scheduled = False


def _running_loop() -> Optional[asyncio.AbstractEventLoop]:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None
