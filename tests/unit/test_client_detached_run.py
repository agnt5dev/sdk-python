from __future__ import annotations

import httpx
import pytest

from agnt5.client import AsyncClient, Client
from agnt5.responses import RunStatus, parse_run_response


def _detached_response(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/run"):
        return httpx.Response(202, json={"run_id": "run-detached", "status": "queued"})
    if request.url.path == "/v1/status/run-detached":
        return httpx.Response(
            200,
            json={"run_id": "run-detached", "status": "completed"},
        )
    if request.url.path == "/v1/result/run-detached":
        return httpx.Response(
            200,
            json={
                "run_id": "run-detached",
                "status": "completed",
                "output": {"ok": True},
            },
        )
    raise AssertionError(f"unexpected request path: {request.url.path}")


def test_queued_run_response_is_pending() -> None:
    response = parse_run_response({"run_id": "run-1", "status": "queued"})

    assert response.status == RunStatus.QUEUED
    assert response.is_pending


def test_sync_run_waits_after_gateway_detaches() -> None:
    requests: list[str] = []
    result_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal result_attempts
        requests.append(request.url.path)
        if request.url.path == "/v1/result/run-detached":
            result_attempts += 1
            if result_attempts == 1:
                return httpx.Response(
                    404,
                    json={"status": "completed", "error": "result not projected yet"},
                )
        return _detached_response(request)

    client = Client("http://gateway.test")
    client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        result = client.run("noop", timeout=1.0)
    finally:
        client.close()

    assert result.is_success
    assert result.output == {"ok": True}
    assert requests == [
        "/v1/functions/noop/run",
        "/v1/status/run-detached",
        "/v1/result/run-detached",
        "/v1/result/run-detached",
    ]


@pytest.mark.asyncio
async def test_async_run_waits_after_gateway_detaches() -> None:
    requests: list[str] = []
    result_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal result_attempts
        requests.append(request.url.path)
        if request.url.path == "/v1/result/run-detached":
            result_attempts += 1
            if result_attempts == 1:
                return httpx.Response(
                    404,
                    json={"status": "completed", "error": "result not projected yet"},
                )
        return _detached_response(request)

    client = AsyncClient("http://gateway.test", timeout=1.0)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await client.run("noop")
    finally:
        await client.close()

    assert result.is_success
    assert result.output == {"ok": True}
    assert requests == [
        "/v1/functions/noop/run",
        "/v1/status/run-detached",
        "/v1/result/run-detached",
        "/v1/result/run-detached",
    ]
