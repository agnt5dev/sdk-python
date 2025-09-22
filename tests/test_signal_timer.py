from __future__ import annotations

import pytest

from datetime import datetime, timedelta, timezone

from agnt5.context import Context


@pytest.fixture
def context() -> Context:
    return Context(
        run_id="run-1",
        step_id="step-1",
        attempt=1,
        invocation_id="inv-1",
        service_name="svc",
        component_name="handler",
        metadata={"tenant_id": "tenant-1"},
    )


@pytest.mark.asyncio
async def test_signal_emit_uses_gateway(monkeypatch, context: Context):
    captured = {}

    def fake_request(method: str, path: str, body):
        captured.update({"method": method, "path": path, "body": body})
        return {"event_id": "evt-123"}

    monkeypatch.setattr(context.signal._transport, "request", fake_request)

    payload = {"message": "hello"}
    response = await context.signal.emit("done", payload)

    assert response["event_id"] == "evt-123"
    assert captured["method"] == "POST"
    assert captured["path"] == "/runs/run-1/signals"
    assert captured["body"]["signal_name"] == "done"
    assert captured["body"]["payload"] == payload


@pytest.mark.asyncio
async def test_signal_wait_returns_payload(monkeypatch, context: Context):
    expected = {"greeting": "hi"}

    def fake_request(method: str, path: str, body):
        assert path == "/runs/run-1/signals/wait"
        return {"delivered": True, "payload": expected}

    monkeypatch.setattr(context.signal._transport, "request", fake_request)

    result = await context.signal.wait("done")
    assert result == expected


@pytest.mark.asyncio
async def test_timer_schedule_delay(monkeypatch, context: Context):
    captured = {}

    def fake_request(method: str, path: str, body):
        captured.update({"method": method, "path": path, "body": body})
        return {"event_id": "timer-evt"}

    monkeypatch.setattr(context.timer._transport, "request", fake_request)

    await context.timer.schedule("followup", delay=timedelta(seconds=5))

    assert captured["path"] == "/runs/run-1/timers"
    assert captured["body"]["timer_key"] == "followup"
    assert captured["body"]["schedule_type"] == "delay"
    assert captured["body"]["schedule_payload"]["delay_ms"] == 5000


@pytest.mark.asyncio
async def test_timer_schedule_absolute(monkeypatch, context: Context):
    received = {}

    def fake_request(method: str, path: str, body):
        received.update(body)
        return {"event_id": "timer-evt"}

    monkeypatch.setattr(context.timer._transport, "request", fake_request)

    run_at = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    await context.timer.schedule("reminder", fire_at=run_at)

    assert received["schedule_type"] == "absolute"
    assert "fire_at" in received["schedule_payload"]


@pytest.mark.asyncio
async def test_timer_cancel(monkeypatch, context: Context):
    captured = {}

    def fake_request(method: str, path: str, body):
        captured.update({"method": method, "path": path, "body": body})
        return {"event_id": "cancel-evt"}

    monkeypatch.setattr(context.timer._transport, "request", fake_request)

    await context.timer.cancel("followup", waiting_step="await_response")

    assert captured["method"] == "DELETE"
    assert captured["path"].startswith("/runs/run-1/timers/followup")


@pytest.mark.asyncio
async def test_context_sleep_uses_durable_timer(monkeypatch, context: Context):
    schedule_calls: list[dict] = []
    wait_calls: list[tuple[str, str]] = []

    def fake_timer_request(method: str, path: str, body=None):
        schedule_calls.append({
            "method": method,
            "path": path,
            "body": body,
        })
        return {"event_id": "timer-evt"}

    wait_responses = [
        {"timers": [{"timer_key": "step-1:sleep:1", "waiting_step": "step-1"}]},
        {"timers": []},
    ]

    def fake_wait_request(self, method: str, path: str, body=None):  # type: ignore[no-redef]
        wait_calls.append((method, path))
        assert method == "GET"
        assert path == "/runs/run-1/waits"
        if wait_responses:
            return wait_responses.pop(0)
        return {"timers": []}

    monkeypatch.setattr(context.timer._transport, "request", fake_timer_request)
    monkeypatch.setattr("agnt5.context._GatewayTransport.request", fake_wait_request)

    await context.sleep(0.1)

    assert schedule_calls, "timer schedule should be invoked"
    schedule_body = schedule_calls[0]["body"]
    assert schedule_body["timer_key"].startswith("step-1:sleep:")
    assert schedule_body["waiting_step"] == "step-1"
    assert schedule_body["schedule_payload"]["delay_ms"] == pytest.approx(100, rel=0.2)
    assert len(wait_calls) >= 1

    checkpoint = context._checkpoints.get_completed("__sleep__", "step-1:sleep:1")
    assert checkpoint is not None
    assert checkpoint.result["seconds"] == pytest.approx(0.1)

@pytest.mark.asyncio
async def test_human_approval_flow(monkeypatch, context: Context):
    calls = {"get": 0}

    def fake_request(method: str, path: str, body):
        if method == "POST" and path == "/runs/run-1/approvals":
            return {"approval_id": "appr-1", "event_id": "evt-1"}
        if method == "GET" and path == "/runs/run-1/approvals?status=decided":
            calls["get"] += 1
            if calls["get"] < 2:
                return {"approvals": []}
            return {
                "approvals": [
                    {
                        "approval_id": "appr-1",
                        "decision": "APPROVED",
                        "decided_by": "reviewer",
                        "reason": "",
                        "payload": {"note": "ok"},
                        "decided_at": "2025-01-01T00:00:00Z",
                    }
                ]
            }
        raise AssertionError(f"unexpected request {method} {path}")

    monkeypatch.setattr(context.human._transport, "request", fake_request)
    context.human._poll_interval = 0.0

    result = await context.human.approval("manager")
    assert result.approval_id == "appr-1"
    assert result.decision == "approved"
    assert result.payload == {"note": "ok"}


@pytest.mark.asyncio
async def test_human_approval_includes_optional_fields(monkeypatch, context: Context):
    captured = {}

    def fake_request(method: str, path: str, body):
        if method == "POST" and path == "/runs/run-1/approvals":
            captured["body"] = body
            return {"approval_id": "appr-2", "event_id": "evt-2"}
        if method == "GET" and path == "/runs/run-1/approvals?status=decided":
            return {
                "approvals": [
                    {
                        "approval_id": "appr-2",
                        "decision": "DENIED",
                        "decided_by": "reviewer-1",
                        "reason": "policy",
                        "payload": None,
                        "decided_at": "2025-01-01T00:00:05Z",
                    }
                ]
            }
        raise AssertionError(f"unexpected request {method} {path}")

    monkeypatch.setattr(context.human._transport, "request", fake_request)

    await context.human.approval(
        "manager",
        payload={"draft": "hello"},
        timeout=timedelta(minutes=5),
        required_roles=["manager"],
        notify_channels=["slack:reviews"],
    )

    body = captured["body"]
    assert body["name"] == "manager"
    assert body["payload"] == {"draft": "hello"}
    assert body["required_roles"] == ["manager"]
    assert body["notify_channels"] == ["slack:reviews"]
    assert body["timeout_ms"] == 5 * 60 * 1000


@pytest.mark.asyncio
async def test_human_approval_zero_timeout_returns_pending(monkeypatch, context: Context):
    calls = {"get": 0}

    def fake_request(method: str, path: str, body):
        if method == "POST" and path == "/runs/run-1/approvals":
            return {"approval_id": "appr-3"}
        if method == "GET" and path == "/runs/run-1/approvals?status=decided":
            calls["get"] += 1
            return {"approvals": []}
        raise AssertionError(f"unexpected request {method} {path}")

    monkeypatch.setattr(context.human._transport, "request", fake_request)
    context.human._poll_interval = 0.0

    result = await context.human.approval(
        "manager",
        payload={"draft": "pending"},
        timeout=timedelta(seconds=0),
    )

    assert calls["get"] == 1
    assert result.decision == "pending"
    assert result.payload == {"draft": "pending"}
    assert result.decided_at is None


@pytest.mark.asyncio
async def test_human_approval_timeout_raises(monkeypatch, context: Context):
    def fake_request(method: str, path: str, body):
        if method == "POST" and path == "/runs/run-1/approvals":
            return {"approval_id": "appr-4"}
        if method == "GET" and path == "/runs/run-1/approvals?status=decided":
            return {"approvals": []}
        raise AssertionError(f"unexpected request {method} {path}")

    monkeypatch.setattr(context.human._transport, "request", fake_request)
    context.human._poll_interval = 0.01

    with pytest.raises(TimeoutError):
        await context.human.approval("manager", timeout=timedelta(milliseconds=5))
