"""Human-in-the-loop client for approval workflows."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

if TYPE_CHECKING:
    from ..context import Context, _GatewayTransport


@dataclass
class ApprovalResult:
    """Result of an approval request."""

    approval_id: str
    decision: str
    decided_by: Optional[str]
    reason: Optional[str]
    payload: Any
    decided_at: Optional[datetime]


class HumanClient:
    """Helpers for human approval workflows backed by the gateway."""

    def __init__(self, context: Context) -> None:
        # Import here to avoid circular dependency
        from ..context import _GatewayTransport

        self._context = context
        self._transport = _GatewayTransport(context)
        try:
            interval = float(os.getenv("AGNT5_APPROVAL_POLL_INTERVAL", "0.5"))
        except ValueError:
            interval = 0.5
        self._poll_interval = max(0.1, interval)
        # Default timeout expressed in seconds; 0 disables waiting
        try:
            self._default_timeout = float(os.getenv("AGNT5_APPROVAL_TIMEOUT_SECONDS", "900"))
        except ValueError:
            self._default_timeout = 900.0

    async def approval(
        self,
        name: str,
        payload: Any | None = None,
        *,
        run_id: Optional[str] = None,
        timeout: Optional[timedelta] = None,
        required_roles: Optional[Sequence[str]] = None,
        notify_channels: Optional[Sequence[str]] = None,
    ) -> ApprovalResult:
        """Request human approval and wait for decision.

        Args:
            name: Name/description of what needs approval
            payload: Optional data to include with approval request
            run_id: Run ID (defaults to context run_id)
            timeout: How long to wait for decision
            required_roles: List of roles that can approve
            notify_channels: Notification channels to alert

        Returns:
            ApprovalResult with decision or pending status

        Raises:
            ValueError: If name is empty
            RuntimeError: If run_id not available
            TimeoutError: If timeout expires before decision
        """
        if not name:
            raise ValueError("approval name must be provided")

        run = run_id or self._context.run_id
        if not run:
            raise RuntimeError("run_id is required to request approval")

        body: Dict[str, Any] = {"name": name}
        if payload is not None:
            body["payload"] = payload
        if required_roles:
            body["required_roles"] = list(required_roles)
        if notify_channels:
            body["notify_channels"] = list(notify_channels)
        if timeout is not None:
            timeout_seconds = max(timeout.total_seconds(), 0.0)
            body["timeout_ms"] = int(timeout_seconds * 1000)
        elif self._default_timeout > 0:
            body["timeout_ms"] = int(self._default_timeout * 1000)

        response = await asyncio.to_thread(
            self._transport.request,
            "POST",
            f"/runs/{run}/approvals",
            body,
        )

        approval_id = response.get("approval_id")
        if not approval_id:
            raise RuntimeError("gateway did not return approval_id")

        poll_timeout = timeout.total_seconds() if timeout is not None else self._default_timeout
        poll_timeout = max(poll_timeout, 0.0)
        wait_for_decision = poll_timeout > 0
        deadline = time.monotonic() + poll_timeout if wait_for_decision else None
        pending_payload = copy.deepcopy(payload) if payload is not None else None

        while True:
            list_resp = await asyncio.to_thread(
                self._transport.request,
                "GET",
                f"/runs/{run}/approvals?status=decided",
                None,
            )
            for item in list_resp.get("approvals", []):
                if item.get("approval_id") != approval_id:
                    continue
                decision = str(item.get("decision", "")).lower() or "pending"
                decided_by = item.get("decided_by")
                reason = item.get("reason")
                decided_at = self._parse_timestamp(item.get("decided_at"))
                payload_value = self._decode_payload(item.get("payload"))
                return ApprovalResult(
                    approval_id=approval_id,
                    decision=decision,
                    decided_by=decided_by,
                    reason=reason,
                    payload=payload_value,
                    decided_at=decided_at,
                )

            if not wait_for_decision:
                break

            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"approval decision not received before timeout (approval_id={approval_id})"
                )

            await asyncio.sleep(self._poll_interval)

        return ApprovalResult(
            approval_id=approval_id,
            decision="pending",
            decided_by=None,
            reason=None,
            payload=pending_payload,
            decided_at=None,
        )

    def _decode_payload(self, value: Any) -> Any:
        """Decode payload from gateway response."""
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray)):
            try:
                return json.loads(value.decode("utf-8"))
            except Exception:  # pragma: no cover - fall back to raw
                return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return ""
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
        return value

    @staticmethod
    def _parse_timestamp(value: Any) -> Optional[datetime]:
        """Parse timestamp from various formats."""
        if not value:
            return None
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
            except ValueError:
                return None
        return None