from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any

import pytest

from agnt5 import FunctionRegistry, WorkflowRegistry, function, workflow
from agnt5.serverless import serve


@pytest.fixture(autouse=True)
def clear_registries():
    FunctionRegistry.clear()
    WorkflowRegistry.clear()
    yield
    FunctionRegistry.clear()
    WorkflowRegistry.clear()


@pytest.mark.asyncio
async def test_serverless_manifest_includes_python_workflows_and_functions() -> None:
    @function
    async def uppercase(text: str) -> dict[str, str]:
        return {"text": text.upper()}

    @workflow
    async def hello(ctx, name: str = "world") -> dict[str, str]:
        return {"message": f"hello {name}"}

    app = serve(service_name="python-workerless", service_version="test")

    status, body, _headers = await call_asgi(app, "GET", "/.well-known/agnt5")

    assert status == 200
    assert body["protocol_version"] == "workerless.v1"
    assert body["service_name"] == "python-workerless"
    assert body["service_version"] == "test"
    assert any(
        component["name"] == "hello"
        and component["type"] == "workflow"
        and component["component_type"] == "workflow"
        for component in body["components"]
    )
    assert any(
        component["name"] == "uppercase"
        and component["type"] == "function"
        and component["component_type"] == "function"
        for component in body["components"]
    )


@pytest.mark.asyncio
async def test_serverless_invokes_python_workflow() -> None:
    @workflow
    async def hello(ctx, name: str = "world") -> dict[str, str]:
        return {"message": f"hello {name}"}

    app = serve(service_name="python-workerless")

    status, body, _headers = await call_asgi(
        app,
        "POST",
        "/agnt5/invoke",
        {
            "protocol_version": "workerless.v1",
            "run_id": "python-workerless-hello",
            "component_type": "workflow",
            "component_name": "hello",
            "input": {"name": "Ada"},
        },
    )

    assert status == 200
    assert body["status"] == "completed"
    assert body["output"] == {"message": "hello Ada"}


@pytest.mark.asyncio
async def test_serverless_verifies_signed_invokes() -> None:
    @workflow
    async def hello(ctx, name: str = "world") -> dict[str, str]:
        return {"message": f"hello {name}"}

    app = serve(service_name="python-workerless", signing_secret="secret-123")
    payload = {
        "protocol_version": "workerless.v1",
        "run_id": "python-workerless-signed",
        "component_type": "workflow",
        "component_name": "hello",
        "input": {"name": "Grace"},
    }

    unsigned_status, unsigned_body, _headers = await call_asgi(app, "POST", "/agnt5/invoke", payload)
    assert unsigned_status == 401
    assert unsigned_body["error"]["code"] == "WORKERLESS_SIGNATURE_MISSING"

    status, body, _headers = await call_asgi(
        app,
        "POST",
        "/agnt5/invoke",
        payload,
        headers=signed_headers("secret-123", payload),
    )

    assert status == 200
    assert body["status"] == "completed"
    assert body["output"] == {"message": "hello Grace"}


@pytest.mark.asyncio
async def test_serverless_budget_suspension_and_checkpoint_resume() -> None:
    fetch_count = 0

    @workflow
    async def research(ctx, title: str = "AGNT5") -> dict[str, Any]:
        nonlocal fetch_count

        async def fetch() -> dict[str, Any]:
            nonlocal fetch_count
            fetch_count += 1
            return {"title": title, "fetch_count": fetch_count}

        page = await ctx.step("fetch", fetch)
        if ctx.attempt == 0:
            await ctx.yield_if_needed()
        return {"summary": f"summary:{page['title']}", "fetch_count": page["fetch_count"]}

    app = serve(service_name="python-workerless")

    suspended_status, suspended, _headers = await call_asgi(
        app,
        "POST",
        "/agnt5/invoke",
        {
            "protocol_version": "workerless.v1",
            "run_id": "python-workerless-research",
            "component_type": "workflow",
            "component_name": "research",
            "attempt": 0,
            "input": {"title": "AGNT5"},
            "budget": {"deadline_ms": int(time.time() * 1000) - 1, "yield_before_timeout_ms": 0},
        },
    )

    assert suspended_status == 200
    assert suspended["status"] == "suspended"
    assert suspended["reason"] == "budget"
    assert suspended["checkpoint"]["steps"]["step:fetch"] == {"title": "AGNT5", "fetch_count": 1}

    completed_status, completed, _headers = await call_asgi(
        app,
        "POST",
        "/agnt5/invoke",
        {
            "protocol_version": "workerless.v1",
            "run_id": "python-workerless-research",
            "component_type": "workflow",
            "component_name": "research",
            "attempt": 1,
            "input": {"title": "AGNT5"},
            "checkpoint": suspended["checkpoint"],
            "budget": {"deadline_ms": int(time.time() * 1000) + 60_000, "yield_before_timeout_ms": 0},
        },
    )

    assert completed_status == 200
    assert completed["status"] == "completed"
    assert completed["output"] == {"summary": "summary:AGNT5", "fetch_count": 1}
    assert fetch_count == 1


async def call_asgi(
    app,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else b""
    sent: list[dict[str, Any]] = []
    received = False

    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "scheme": "http",
        "server": ("127.0.0.1", 8787),
        "client": ("127.0.0.1", 12345),
        "headers": [
            (key.lower().encode("latin-1"), value.encode("latin-1"))
            for key, value in {"host": "127.0.0.1:8787", **(headers or {})}.items()
        ],
    }

    async def receive() -> dict[str, Any]:
        nonlocal received
        if received:
            return {"type": "http.request", "body": b"", "more_body": False}
        received = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(scope, receive, send)

    start = next(message for message in sent if message["type"] == "http.response.start")
    body_parts = [message.get("body", b"") for message in sent if message["type"] == "http.response.body"]
    response_headers = {
        key.decode("latin-1"): value.decode("latin-1")
        for key, value in start.get("headers", [])
    }
    return start["status"], json.loads(b"".join(body_parts).decode("utf-8")), response_headers


def signed_headers(secret: str, payload: dict[str, Any]) -> dict[str, str]:
    body = json.dumps(payload).encode("utf-8")
    timestamp = str(int(time.time() * 1000))
    attempt_id = f"{payload['run_id']}:{payload.get('attempt', 0)}"
    signature = hmac.new(
        secret.encode("utf-8"),
        f"{timestamp}.{attempt_id}.".encode("utf-8") + body,
        hashlib.sha256,
    ).hexdigest()
    return {
        "content-type": "application/json",
        "x-agnt5-signature-version": "workerless-hmac-sha256.v1",
        "x-agnt5-timestamp": timestamp,
        "x-agnt5-attempt-id": attempt_id,
        "x-agnt5-signature": f"sha256={signature}",
    }
