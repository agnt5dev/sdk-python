"""Durable timer client."""

from __future__ import annotations

import asyncio
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

if TYPE_CHECKING:
    from ..context import Context, _GatewayTransport


class TimerClient:
    """Durable timer helpers for scheduling and cancelling waits."""

    def __init__(self, context: "Context") -> None:
        self._context = context
        self._transport = _GatewayTransport(context)

    async def schedule(
        self,
        key: str,
        *,
        run_id: Optional[str] = None,
        waiting_step: Optional[str] = None,
        delay: Optional[timedelta] = None,
        fire_at: Optional[datetime] = None,
        schedule_type: Optional[str] = None,
        dedupe_id: Optional[str] = None,
    ) -> Mapping[str, Any]:
        if not key:
            raise ValueError("timer key must be provided")

        run = run_id or self._context.run_id
        if not run:
            raise RuntimeError("run_id is required to schedule a timer")

        schedule: Dict[str, Any] = {}
        stype = (schedule_type or "").lower()

        if fire_at is not None:
            stype = stype or "absolute"
            schedule["fire_at"] = fire_at.astimezone(timezone.utc).isoformat()
        elif delay is not None:
            stype = stype or "delay"
            schedule["delay_ms"] = int(delay.total_seconds() * 1000)
        else:
            raise ValueError("either delay or fire_at must be provided")

        body: Dict[str, Any] = {
            "timer_key": key,
            "schedule_type": stype,
            "schedule_payload": schedule,
        }
        if waiting_step:
            body["waiting_step"] = waiting_step
        if dedupe_id:
            body["dedupe_id"] = dedupe_id

        response = await asyncio.to_thread(
            self._transport.request,
            "POST",
            f"/runs/{run}/timers",
            body,
        )
        return response

    async def cancel(
        self,
        key: str,
        *,
        run_id: Optional[str] = None,
        waiting_step: Optional[str] = None,
    ) -> Mapping[str, Any]:
        if not key:
            raise ValueError("timer key must be provided")

        run = run_id or self._context.run_id
        if not run:
            raise RuntimeError("run_id is required to cancel a timer")

        path = f"/runs/{run}/timers/{urllib.parse.quote(key, safe='')}"
        if waiting_step:
            path = f"{path}?step={urllib.parse.quote(waiting_step, safe='')}"

        response = await asyncio.to_thread(
            self._transport.request,
            "DELETE",
            path,
            None,
        )
        return response


@dataclass
