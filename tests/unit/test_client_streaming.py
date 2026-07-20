from __future__ import annotations

import httpx
import pytest

from agnt5.client import Client, RunError


def gateway_sse(*events: tuple[str, str]) -> bytes:
    lines: list[str] = []
    for event_type, data in events:
        lines.extend((f"event: {event_type}", f"data: {data}", ""))
    return ("\n".join(lines) + "\n").encode()


def streaming_client(body: bytes) -> Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/functions/generate/stream"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    client = Client("http://gateway.test")
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def test_stream_unwraps_gateway_event_envelopes() -> None:
    body = gateway_sse(
        (
            "output.delta",
            '{"event_type":"output.delta","run_id":"run-1","data":{"content":"hel","index":0}}',
        ),
        (
            "output.delta",
            '{"event_type":"output.delta","run_id":"run-1","data":{"content":"lo","index":0}}',
        ),
        (
            "run.completed",
            '{"event_type":"run.completed","run_id":"run-1","data":{"output_data":"hello"}}',
        ),
    )
    client = streaming_client(body)
    try:
        assert list(client.stream("generate")) == ["hel", "lo"]
        events = list(client.stream_events("generate"))
    finally:
        client.close()

    deltas = [event for event in events if event.event_type == "output.delta"]
    assert [event.data["content"] for event in deltas] == ["hel", "lo"]
    assert all("event_type" not in event.data for event in deltas)
    assert [event.sequence for event in deltas] == [1, 2]
    assert all(event.run_id == "run-1" for event in deltas)


def test_stream_raises_for_enveloped_run_failure() -> None:
    client = streaming_client(
        gateway_sse(
            (
                "run.failed",
                '{"event_type":"run.failed","run_id":"run-2","data":{"error_message":"boom","error_code":"FUNCTION_ERROR"}}',
            ),
        )
    )
    try:
        with pytest.raises(RunError, match="boom") as raised:
            list(client.stream("generate"))
    finally:
        client.close()

    assert raised.value.run_id == "run-2"
    assert raised.value.error_code == "FUNCTION_ERROR"
