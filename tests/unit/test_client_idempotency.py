from __future__ import annotations

import httpx
import pytest

from agnt5.client import AsyncClient, Client


def _response_for(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/run"):
        return httpx.Response(
            200,
            json={"run_id": "run-1", "status": "completed", "output": {"ok": True}},
        )
    if path.endswith("/submit"):
        return httpx.Response(202, json={"run_id": "run-1", "status": "enqueued"})
    if path.endswith("/batch"):
        return httpx.Response(
            200,
            json={"batch_id": "batch-1", "status": "queued", "results": []},
        )
    if path.endswith("/stream"):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                "event: output.delta\n"
                'data: {"event_type":"output.delta","run_id":"run-1","data":{"content":"ok"}}\n\n'
                "event: run.completed\n"
                'data: {"event_type":"run.completed","run_id":"run-1","data":{"output_data":"ok"}}\n\n'
            ).encode(),
        )
    raise AssertionError(f"unexpected request path: {path}")


def test_sync_invocation_methods_send_explicit_idempotency_key() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response_for(request)

    client = Client("http://gateway.test")
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        client.run("checkout", {"value": 1}, idempotency_key="run-key")
        client.submit("checkout", {"value": 1}, idempotency_key="submit-key")
        list(client.stream("checkout", {"value": 1}, idempotency_key="stream-key"))
        client.batch("checkout", [{"value": 1}], idempotency_key="batch-key")
    finally:
        client.close()

    assert [request.headers["Idempotency-Key"] for request in requests] == [
        "run-key",
        "submit-key",
        "stream-key",
        "batch-key",
    ]


@pytest.mark.asyncio
async def test_async_invocation_methods_send_explicit_idempotency_key() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response_for(request)

    client = AsyncClient("http://gateway.test")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await client.run("checkout", {"value": 1}, idempotency_key="run-key")
        await client.submit("checkout", {"value": 1}, idempotency_key="submit-key")
        _ = [
            event
            async for event in client.stream_events(
                "checkout", {"value": 1}, idempotency_key="stream-key"
            )
        ]
        await client.batch("checkout", [{"value": 1}], idempotency_key="batch-key")
    finally:
        await client.close()

    assert [request.headers["Idempotency-Key"] for request in requests] == [
        "run-key",
        "submit-key",
        "stream-key",
        "batch-key",
    ]


def test_explicit_idempotency_key_overrides_legacy_custom_header() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Idempotency-Key"])
        return _response_for(request)

    client = Client("http://gateway.test")
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        client.run(
            "checkout",
            headers={"Idempotency-Key": "legacy-key"},
            idempotency_key="explicit-key",
        )
    finally:
        client.close()

    assert seen == ["explicit-key"]
