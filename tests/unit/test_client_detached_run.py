from __future__ import annotations

import httpx
import pytest

from agnt5.client import AsyncClient, Client
from agnt5.responses import RunStatus, parse_run_response


def test_queued_run_response_is_pending() -> None:
    response = parse_run_response({"run_id": "run-1", "status": "queued"})
    assert response.status == RunStatus.QUEUED
    assert response.is_pending


@pytest.mark.parametrize("wait", [0, 60, 600])
def test_sync_run_returns_receipt_without_another_wait(wait: float) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        assert request.headers["X-AGNT5-Wait-Timeout-Ms"] == str(wait * 1000)
        assert request.extensions["timeout"]["read"] == max(45, wait + 10)
        return httpx.Response(202, json={"run_id": "run-wait", "status": "pending"})

    client = Client("http://gateway.test")
    client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        result = client.run("noop", wait_timeout=wait)
        assert result.is_pending
        assert result.status == RunStatus.PENDING
        assert result.run_id == "run-wait"
        assert requests == ["/v1/functions/noop/run"]
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_run_returns_receipt_without_another_wait() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        assert request.headers["X-AGNT5-Wait-Timeout-Ms"] == "60000"
        assert request.extensions["timeout"]["read"] == 70
        return httpx.Response(202, json={"run_id": "run-wait", "status": "pending"})

    client = AsyncClient("http://gateway.test")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await client.run("noop", wait_timeout=60)
        assert result.is_pending
        assert result.status == RunStatus.PENDING
        assert requests == ["/v1/functions/noop/run"]
    finally:
        await client.close()


@pytest.mark.parametrize("wait", [-1, float("nan"), float("inf"), True, 86401])
def test_invalid_wait_rejected_before_request(wait: float) -> None:
    with Client("http://gateway.test") as client:
        with pytest.raises(ValueError, match="wait_timeout"):
            client.run("noop", wait_timeout=wait)
        with pytest.raises(ValueError, match="wait_timeout"):
            list(client.stream_events("noop", wait_timeout=wait))


@pytest.mark.parametrize("accepted", [False, True])
def test_stream_wait_receipt_preserves_run_id(accepted: bool) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-AGNT5-Wait-Timeout-Ms"] == "300000"
        assert request.extensions["timeout"]["read"] == 310
        if accepted:
            return httpx.Response(202, json={"run_id": "run-wait", "status": "pending"})
        return httpx.Response(200, content=b'event: stream.wait_expired\ndata: {"run_id":"run-wait","status":"pending"}\n\n')

    client = Client("http://gateway.test")
    client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        events = list(client.stream_events("noop"))
        assert len(events) == 1
        assert events[0].run_id == "run-wait"
        assert events[0].event_type == ("stream.detached" if accepted else "stream.wait_expired")
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_stream_supports_custom_wait_and_receipt() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-AGNT5-Wait-Timeout-Ms"] == "600000"
        assert request.extensions["timeout"]["read"] == 610
        return httpx.Response(202, json={"run_id": "run-wait", "status": "pending"})

    client = AsyncClient("http://gateway.test")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        events = [event async for event in client.stream_events("noop", wait_timeout=600)]
        assert events[0].event_type == "stream.detached"
        assert events[0].run_id == "run-wait"
    finally:
        await client.close()
