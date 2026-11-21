"""A chat run's bot token is fetched with the worker's workload identity.

The run's input names the integration and carries no token, because run input
is stored with the run's history.
"""

import json

import httpx
import pytest

from agnt5.chat.credentials import BotTokenSource


def broker(calls: list, *, status: int = 200, value: str = "xoxb-fetched"):
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((request.url.path, body, request.headers.get("authorization")))
        if request.url.path == "/internal/workload/token":
            return httpx.Response(200, json={"token": "workload-token"})
        if status != 200:
            return httpx.Response(status, json={"error": "integration not found"})
        return httpx.Response(
            200,
            json={
                "value": value,
                "expires_at": "2999-01-01T00:00:00Z",
                "provider": "slack",
                "integration_id": body["ref"],
            },
        )

    return httpx.MockTransport(handle)


@pytest.fixture
def token_file(tmp_path):
    path = tmp_path / "token"
    path.write_text("projected-token\n")
    return str(path)


async def test_the_token_is_fetched_once_for_its_integration(token_file):
    calls: list = []
    source = BotTokenSource("http://broker", token_file, transport=broker(calls))

    assert await source.get("integration-1") == "xoxb-fetched"
    assert await source.get("integration-1") == "xoxb-fetched"

    paths = [path for path, _, _ in calls]
    assert paths == ["/internal/workload/token", "/internal/workload/credential"]
    exchange, credential = calls[0][1], calls[1][1]
    assert exchange == {"token": "projected-token", "audience": ["agnt5-credentials"]}
    assert credential == {
        "scope": "chat_bot_token",
        "ref": "integration-1",
        "purpose": "chat_reply",
    }
    assert calls[1][2] == "Bearer workload-token"


async def test_a_refused_fetch_falls_back(token_file, caplog):
    calls: list = []
    source = BotTokenSource("http://broker", token_file, transport=broker(calls, status=404))

    assert await source.get("integration-1") is None
    assert "integration not found" in caplog.text


async def test_without_a_workload_identity_nothing_is_fetched(token_file):
    calls: list = []
    assert (
        await BotTokenSource("", token_file, transport=broker(calls)).get("integration-1") is None
    )
    assert (
        await BotTokenSource("http://broker", "", transport=broker(calls)).get("integration-1")
        is None
    )
    assert (
        await BotTokenSource("http://broker", token_file, transport=broker(calls)).get("") is None
    )
    assert calls == []


async def test_the_worker_replies_with_the_fetched_token(monkeypatch):
    from agnt5.chat import credentials
    from agnt5.worker._core import Worker

    received = {}

    class Bot:
        name = "support-bot"

        async def handle_webhook(self, source, headers, body, bot_token):
            received.update(source=source, bot_token=bot_token)
            return None

    class Source:
        async def get(self, integration_id):
            received["integration_id"] = integration_id
            return "xoxb-fetched"

    monkeypatch.setattr(credentials, "_default_source", Source())
    envelope = json.dumps(
        {
            "source": "slack",
            "event_type": "slack.message",
            "headers": {},
            "body": "{}",
            "integration_id": "integration-1",
        }
    ).encode()
    request = type("Request", (), {"invocation_id": "run-1", "attempt": 0})()

    response = await Worker._execute_chat_webhook(None, Bot(), envelope, request)

    assert response.success
    assert received == {
        "source": "slack",
        "bot_token": "xoxb-fetched",
        "integration_id": "integration-1",
    }
