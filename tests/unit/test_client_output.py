"""Large output references and asynchronous client parity at the HTTP seam."""

import httpx
import pytest

from agnt5.client import AsyncClient, Client

OUTPUT_REF = {"ref": "s3://outputs/run-1.json", "size_bytes": 100000, "sha256": "abc"}


def gateway(request):
    if request.url.path == "/v1/status/run-1":
        return httpx.Response(200, json={"run_id": "run-1", "status": "completed"})
    if request.url.path == "/v1/result/run-1":
        return httpx.Response(
            200, json={"run_id": "run-1", "status": "completed", "output_ref": OUTPUT_REF}
        )
    if request.url.path == "/v1/runs/run-1/output":
        return httpx.Response(200, json={"output": {"text": "large output"}})
    raise AssertionError(f"Unexpected request: {request.url}")


def test_sync_client_resolves_large_output():
    with Client("http://gateway.test") as client:
        client._client.close()
        client._client = httpx.Client(transport=httpx.MockTransport(gateway))
        result = client.get_result("run-1")
        assert result.has_output_ref
        assert result.output_ref.ref == OUTPUT_REF["ref"]
        assert result.output_ref.size_bytes == 100000
        assert client.resolve_output(result) == {"text": "large output"}
        assert client.wait_for_output("run-1") == {"text": "large output"}


async def test_async_client_resolves_large_output():
    client = AsyncClient("http://gateway.test")
    async with client:
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(gateway))
        result = await client.get_result("run-1")
        assert result.has_output_ref
        assert result.output_ref.ref == OUTPUT_REF["ref"]
        assert await client.resolve_output(result) == {"text": "large output"}
        assert await client.wait_for_output("run-1") == {"text": "large output"}


@pytest.mark.parametrize("output", [None, False, 0, "", [], {"ok": True}])
async def test_resolve_inline_output_does_not_fetch(output):
    from agnt5.responses import parse_run_response

    def unexpected_request(request):
        raise AssertionError("Inline output must not trigger another request")

    result = parse_run_response({"run_id": "run-1", "status": "completed", "output": output})
    with Client("http://gateway.test") as client:
        client._client.close()
        client._client = httpx.Client(transport=httpx.MockTransport(unexpected_request))
        assert client.resolve_output(result) == output
    async with AsyncClient("http://gateway.test") as client:
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(unexpected_request))
        assert await client.resolve_output(result) == output


async def test_async_proxies_invoke_and_stream():
    import json

    seen = []

    def handler(request):
        seen.append((request.url.path, json.loads(request.content), dict(request.headers)))
        if request.url.path.endswith("/stream"):
            return httpx.Response(
                200,
                text='event: output.delta\ndata: {"content":"hello"}\n\nevent: done\ndata: {}\n\n',
            )
        output = {"response": "hello"}
        if request.url.path.endswith("get_history"):
            output = [{"role": "user", "content": "hi"}]
        return httpx.Response(
            200, json={"run_id": "run-1", "status": "completed", "output": output}
        )

    async with AsyncClient("http://gateway.test") as client:
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        workflow = client.workflow("support")
        assert (await workflow.run(topic="SDK", session_id="session", user_id="user")).is_success
        assert (await workflow.chat("hi", session_id="session")).is_success
        assert (await workflow.submit(topic="SDK")).run_id == "run-1"
        assert [event.event_type async for event in workflow.stream_events(topic="SDK")] == [
            "output.delta"
        ]
        assert (await client.entity("Counter", "one").increment(amount=2)).is_success
        session = client.session("Conversation", "one")
        assert await session.chat("hi") == "hello"
        assert await session.get_history() == [{"role": "user", "content": "hi"}]
        assert (await session.add_message("user", "hi")).is_success
        assert (await session.clear_history()).is_success
        assert [chunk async for chunk in client.stream("greet")] == ["hello"]

    assert seen[0][0] == "/v1/workflows/support/run"
    assert seen[0][1] == {"topic": "SDK"}
    assert seen[0][2]["x-session-id"] == "session"
    assert seen[0][2]["x-user-id"] == "user"
    assert seen[4][0] == "/v1/entity/Counter/one/increment"
    assert seen[4][1] == {"amount": 2}


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(202, json={"run_id": "run-1"}),
        httpx.Response(
            200,
            text='event: run.failed\ndata: {"run_id":"run-1","data":{"error_message":"broken"}}\n\n',
        ),
        httpx.Response(200, text='event: stream.wait_expired\ndata: {"run_id":"run-1"}\n\n'),
    ],
)
async def test_async_chunk_stream_surfaces_failure_or_detach(response):
    from agnt5.client import RunError

    async with AsyncClient("http://gateway.test") as client:
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response))
        with pytest.raises(RunError) as error:
            async for _ in client.stream("greet"):
                pass
        assert error.value.run_id == "run-1"


async def test_async_wait_for_result_polls_and_times_out(monkeypatch):
    import agnt5.client as client_module

    polls = []

    def handler(request):
        if request.url.path.startswith("/v1/status"):
            polls.append(request.url.path)
            return httpx.Response(
                200,
                json={"run_id": "run-1", "status": "running" if len(polls) == 1 else "completed"},
            )
        return gateway(request)

    async def sleep(delay):
        assert delay == 0.01

    monkeypatch.setattr(client_module.asyncio, "sleep", sleep)
    async with AsyncClient("http://gateway.test") as client:
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        assert (await client.wait_for_result("run-1", poll_interval=0.01)).is_success
        assert len(polls) == 2
        result = await client.wait_for_result("run-1", timeout=0)
        assert result.is_error and result.error.code == "TIMEOUT"


def test_async_client_has_sync_public_methods():
    sync_methods = {
        name for name in dir(Client) if not name.startswith("_") and callable(getattr(Client, name))
    }
    async_methods = {
        name
        for name in dir(AsyncClient)
        if not name.startswith("_") and callable(getattr(AsyncClient, name))
    }
    assert sync_methods <= async_methods
