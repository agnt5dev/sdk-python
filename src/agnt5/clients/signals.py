"""Signal handling client."""

from __future__ import annotations

import asyncio
import urllib.parse
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from ..context import Context, _GatewayTransport


class SignalClient:
    """Durable signal helpers backed by the coordinator gateway."""

    def __init__(self, context: "Context") -> None:
        self._context = context
        self._transport = _GatewayTransport(context)

    async def emit(
        self,
        name: str,
        payload: Any = None,
        *,
        run_id: Optional[str] = None,
        invocation_id: Optional[str] = None,
        dedupe_id: Optional[str] = None,
    ) -> Mapping[str, Any]:
        if not name:
            raise ValueError("signal name must be provided")

        run = run_id or self._context.run_id
        if not run:
            raise RuntimeError("run_id is required to emit a signal")

        body: Dict[str, Any] = {"signal_name": name}
        if payload is not None:
            body["payload"] = payload
        if invocation_id:
            body["invocation_id"] = invocation_id
        if dedupe_id:
            body["dedupe_id"] = dedupe_id

        response = await asyncio.to_thread(
            self._transport.request,
            "POST",
            f"/runs/{run}/signals",
            body,
        )
        return response

    async def wait(
        self,
        name: str,
        *,
        run_id: Optional[str] = None,
        waiting_step: Optional[str] = None,
        timeout: Optional[timedelta] = None,
        auto_ack: bool = True,
        dedupe_id: Optional[str] = None,
    ) -> Any:
        if not name:
            raise ValueError("signal name must be provided")

        run = run_id or self._context.run_id
        if not run:
            raise RuntimeError("run_id is required to wait for a signal")

        body: Dict[str, Any] = {
            "signal_name": name,
            "auto_ack": auto_ack,
        }
        if waiting_step:
            body["waiting_step"] = waiting_step
        if timeout is not None:
            body["timeout_ms"] = int(timeout.total_seconds() * 1000)
        if dedupe_id:
            body["dedupe_id"] = dedupe_id

        response = await asyncio.to_thread(
            self._transport.request,
            "POST",
            f"/runs/{run}/signals/wait",
            body,
        )

        if not response.get("delivered"):
            raise TimeoutError("signal wait completed without delivery")

        return response.get("payload")

    async def acknowledge(
        self,
        name: str,
        *,
        run_id: Optional[str] = None,
        waiting_step: str,
        dedupe_id: Optional[str] = None,
        acked_by: Optional[str] = None,
    ) -> Mapping[str, Any]:
        if not name:
            raise ValueError("signal name must be provided")

        run = run_id or self._context.run_id
        if not run:
            raise RuntimeError("run_id is required to acknowledge a signal")

        body: Dict[str, Any] = {
            "waiting_step": waiting_step,
        }
        if dedupe_id:
            body["dedupe_id"] = dedupe_id
        if acked_by:
            body["acked_by"] = acked_by

        response = await asyncio.to_thread(
            self._transport.request,
            "POST",
            f"/runs/{run}/signals/{urllib.parse.quote(name, safe='')}/ack",
            body,
        )

        return response
