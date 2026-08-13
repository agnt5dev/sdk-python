from __future__ import annotations

import asyncio
import hashlib
import hmac
import io
import json
import time
from typing import Any, cast

import pytest

from agnt5 import (
    Agent,
    AgentRegistry,
    FunctionRegistry,
    ToolRegistry,
    WorkflowRegistry,
    function,
    tool,
    workflow,
)
from agnt5.lm import GenerateRequest, GenerateResponse, LanguageModel
from agnt5.lm.events import LMCompleted, LMContentBlockDelta
from agnt5.serverless import WorkerlessContext, _WorkerlessAgentContext, serve


class FakeAgentCompleted:
    event_type = "agent.completed"

    def __init__(self, output: str) -> None:
        self.output_data = {"output": output, "tool_calls": []}

    def to_response_fields(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "output_data": json.dumps(self.output_data),
        }


class FakeAgent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.model_name = "test-model"

    async def stream(self, user_message, context=None, history=None):
        prior_users = [
            str(message.get("content", ""))
            for message in history or []
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        output = "seen:" + "|".join([*prior_users, user_message])
        yield FakeAgentCompleted(output)


class ServerlessTestModel(LanguageModel):
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        return GenerateResponse(text=self._response_text(request))

    async def stream(self, request: GenerateRequest):
        text = self._response_text(request)
        yield LMContentBlockDelta(
            name="serverless-test-model",
            correlation_id="serverless-test-correlation",
            parent_correlation_id="",
            content=text,
            block_type="text",
            index=0,
        )
        yield LMCompleted(
            name="serverless-test-model",
            correlation_id="serverless-test-correlation",
            parent_correlation_id="",
            model="serverless-test-model",
            provider="test",
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            output_data={"text": text},
        )

    @staticmethod
    def _response_text(request: GenerateRequest) -> str:
        user_messages = [
            message.content
            for message in request.messages
            if getattr(message.role, "value", message.role) == "user"
        ]
        return "actual:" + "|".join(user_messages)


def test_workerless_agent_observed_event_preserves_capture_fields() -> None:
    parent = WorkerlessContext(
        invocation_id="workerless-run-1",
        run_id="run-1",
        attempt=1,
        component_name="agent",
    )
    context = _WorkerlessAgentContext(parent, "agent", "session-1")
    event = LMCompleted(
        name="openai/gpt-4o-mini",
        correlation_id="lm-1",
        parent_correlation_id="agent-1",
        model="openai/gpt-4o-mini",
        provider="openai",
        input_tokens=2,
        output_tokens=3,
        total_tokens=5,
        finish_reason="stop",
        output_data={"output": "done"},
        metadata={"source": "openai", "provider": "openai"},
    )

    context.emit_observed(event)

    [captured] = parent.events_snapshot()
    assert captured["event_type"] == "lm.completed"
    assert captured["correlation_id"] == "lm-1"
    assert captured["metadata"]["source"] == "openai"
    assert captured["metadata"]["parent_correlation_id"] == "agent-1"
    assert captured["data"]["finish_reason"] == "stop"
    assert captured["data"]["total_tokens"] == 5


@pytest.fixture(autouse=True)
def clear_registries():
    FunctionRegistry.clear()
    WorkflowRegistry.clear()
    ToolRegistry.clear()
    AgentRegistry.clear()
    yield
    FunctionRegistry.clear()
    WorkflowRegistry.clear()
    ToolRegistry.clear()
    AgentRegistry.clear()


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


def test_serverless_wsgi_app_serves_manifest_and_signed_invoke() -> None:
    @workflow
    async def hello(ctx, name: str = "world") -> dict[str, str]:
        return {"message": f"hello {name}"}

    app = serve(
        service_name="python-wsgi",
        service_version="wsgi-test",
        signing_secret="wsgi-signing-secret",
    )

    manifest_status, manifest, _headers = call_wsgi(app, "GET", "/.well-known/agnt5")
    assert manifest_status == 200
    assert manifest["service_name"] == "python-wsgi"
    assert manifest["service_version"] == "wsgi-test"

    payload = {
        "protocol_version": "workerless.v1",
        "run_id": "python-wsgi-hello",
        "component_type": "workflow",
        "component_name": "hello",
        "input": {"name": "WSGI"},
    }
    unsigned_status, unsigned, _headers = call_wsgi(app, "POST", "/agnt5/invoke", payload)
    assert unsigned_status == 401
    assert unsigned["error"]["code"] == "WORKERLESS_SIGNATURE_MISSING"

    status, body, _headers = call_wsgi(
        app,
        "POST",
        "/agnt5/invoke",
        payload,
        signed_headers("wsgi-signing-secret", payload),
    )
    assert status == 200
    assert body["status"] == "completed"
    assert body["output"] == {"message": "hello WSGI"}


def test_serverless_mount_starlette_registers_protocol_routes() -> None:
    routes: list[tuple[str, list[str], bool]] = []

    class FakeStarlette:
        def add_route(self, path, _handler, *, methods, include_in_schema):
            routes.append((path, methods, include_in_schema))

    app = serve(workflows=[])
    app.mount_starlette(FakeStarlette())

    assert routes == [
        ("/.well-known/agnt5", ["GET"], False),
        ("/agnt5/invoke", ["POST"], False),
    ]


def test_serverless_mount_flask_serves_signed_protocol() -> None:
    flask = pytest.importorskip("flask")

    @workflow
    async def hello(ctx, name: str = "world") -> dict[str, str]:
        return {"message": f"hello {name}"}

    app = flask.Flask(__name__)
    serverless = serve(signing_secret="flask-signing-secret")
    serverless.mount_flask(app)
    client = app.test_client()

    manifest_response = client.get("/.well-known/agnt5")
    assert manifest_response.status_code == 200
    assert manifest_response.get_json()["protocol_version"] == "workerless.v1"

    payload = {
        "protocol_version": "workerless.v1",
        "run_id": "python-flask-hello",
        "component_type": "workflow",
        "component_name": "hello",
        "input": {"name": "Flask"},
    }
    response = client.post(
        "/agnt5/invoke",
        data=json.dumps(payload),
        content_type="application/json",
        headers=signed_headers("flask-signing-secret", payload),
    )
    assert response.status_code == 200
    assert response.get_json()["output"] == {"message": "hello Flask"}


def test_serverless_django_view_serves_signed_protocol() -> None:
    django = pytest.importorskip("django")
    from django.conf import settings
    from django.test import RequestFactory

    if not settings.configured:
        settings.configure(
            ALLOWED_HOSTS=["testserver"],
            DEFAULT_CHARSET="utf-8",
            SECRET_KEY="agnt5-serverless-test",
        )
        django.setup()

    @workflow
    async def hello(ctx, name: str = "world") -> dict[str, str]:
        return {"message": f"hello {name}"}

    serverless = serve(signing_secret="django-signing-secret")
    patterns = serverless.django_urlpatterns()
    assert [pattern.name for pattern in patterns] == [
        "agnt5-workerless-manifest",
        "agnt5-workerless-invoke",
    ]

    payload = {
        "protocol_version": "workerless.v1",
        "run_id": "python-django-hello",
        "component_type": "workflow",
        "component_name": "hello",
        "input": {"name": "Django"},
    }
    request = RequestFactory().generic(
        "POST",
        "/agnt5/invoke",
        data=json.dumps(payload),
        content_type="application/json",
        headers=signed_headers("django-signing-secret", payload),
    )
    response = asyncio.run(serverless.handle_django_request(request))
    assert response.status_code == 200
    assert json.loads(response.content)["output"] == {"message": "hello Django"}


@pytest.mark.asyncio
async def test_serverless_exposes_and_invokes_selected_tools() -> None:
    @tool(description="Double a number")
    async def selected_tool(ctx, value: int) -> dict[str, int]:
        ctx.emit({"event_type": "tool.custom", "data": {"value": value}})
        return {"value": value * 2}

    @tool(description="Must stay private")
    async def hidden_tool(ctx) -> dict[str, bool]:
        return {"hidden": True}

    app = serve(tools=[selected_tool], agents=[])
    manifest_status, manifest, _headers = await call_asgi(app, "GET", "/.well-known/agnt5")

    assert manifest_status == 200
    assert manifest["components"] == [
        {
            "name": "selected_tool",
            "type": "tool",
            "component_type": "tool",
            "input_schema": selected_tool.input_schema,
            "output_schema": selected_tool.output_schema,
            "metadata": {
                "description": "Double a number",
                "requires_confirmation": False,
            },
        }
    ]

    status, body, _headers = await call_asgi(
        app,
        "POST",
        "/agnt5/invoke",
        {
            "protocol_version": "workerless.v1",
            "run_id": "python-workerless-tool",
            "component_type": "tool",
            "component_name": "selected_tool",
            "input": {"value": 4},
        },
    )

    assert status == 200
    assert body["output"] == {"value": 8}
    assert any(event["event_type"] == "tool.custom" for event in body["events"])
    hidden_status, hidden, _headers = await call_asgi(
        app,
        "POST",
        "/agnt5/invoke",
        {
            "protocol_version": "workerless.v1",
            "run_id": "python-workerless-hidden-tool",
            "component_type": "tool",
            "component_name": hidden_tool.name,
            "input": {},
        },
    )
    assert hidden_status == 404
    assert hidden["error"]["code"] == "WORKERLESS_COMPONENT_NOT_FOUND"


@pytest.mark.asyncio
async def test_serverless_agent_checkpoint_resume() -> None:
    selected_agent = cast(Agent, FakeAgent("selected-agent"))
    hidden_agent = cast(Agent, FakeAgent("hidden-agent"))
    AgentRegistry.register(selected_agent)
    AgentRegistry.register(hidden_agent)
    app = serve(tools=[], agents=[selected_agent])

    manifest_status, manifest, _headers = await call_asgi(app, "GET", "/.well-known/agnt5")
    assert manifest_status == 200
    assert manifest["components"] == [
        {
            "name": "selected-agent",
            "type": "agent",
            "component_type": "agent",
            "metadata": {"model": "test-model"},
        }
    ]

    first_status, first, _headers = await call_asgi(
        app,
        "POST",
        "/agnt5/invoke",
        {
            "protocol_version": "workerless.v1",
            "run_id": "python-workerless-agent",
            "component_type": "agent",
            "component_name": "selected-agent",
            "input": {"message": "hello", "session_id": "session-1"},
        },
    )

    assert first_status == 200
    assert first["output"] == "seen:hello"
    assert first["checkpoint"]["agent_sessions"]["session-1"]["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "seen:hello"},
    ]
    assert any(event["event_type"] == "agent.completed" for event in first["events"])
    assert any(event["event_type"] == "session.created" for event in first["events"])

    second_status, second, _headers = await call_asgi(
        app,
        "POST",
        "/agnt5/invoke",
        {
            "protocol_version": "workerless.v1",
            "run_id": "python-workerless-agent",
            "component_type": "agent",
            "component_name": "selected-agent",
            "input": {"message": "again", "session_id": "session-1"},
            "checkpoint": first["checkpoint"],
        },
    )

    assert second_status == 200
    assert second["output"] == "seen:hello|again"
    assert len(second["checkpoint"]["agent_sessions"]["session-1"]["messages"]) == 4


@pytest.mark.asyncio
async def test_serverless_invokes_sdk_agent_without_platform_session_io() -> None:
    agent = Agent(
        name="actual-agent",
        model=ServerlessTestModel(),
        model_name="serverless-test-model",
        instructions="Answer briefly.",
    )
    app = serve(functions=[], workflows=[], tools=[], agents=[agent])

    status, body, _headers = await call_asgi(
        app,
        "POST",
        "/agnt5/invoke",
        {
            "protocol_version": "workerless.v1",
            "run_id": "python-workerless-actual-agent",
            "component_type": "agent",
            "component_name": "actual-agent",
            "input": {"message": "hello", "session_id": "actual-session"},
        },
    )

    assert status == 200
    assert body["output"] == "actual:hello"
    assert body["checkpoint"]["agent_sessions"]["actual-session"]["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "actual:hello"},
    ]


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

    unsigned_status, unsigned_body, _headers = await call_asgi(
        app, "POST", "/agnt5/invoke", payload
    )
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
            "budget": {
                "deadline_ms": int(time.time() * 1000) + 60_000,
                "yield_before_timeout_ms": 0,
            },
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
    body_parts = [
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    ]
    response_headers = {
        key.decode("latin-1"): value.decode("latin-1") for key, value in start.get("headers", [])
    }
    return start["status"], json.loads(b"".join(body_parts).decode("utf-8")), response_headers


def call_wsgi(
    app,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else b""
    environ: dict[str, Any] = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "SCRIPT_NAME": "",
        "QUERY_STRING": "",
        "SERVER_NAME": "127.0.0.1",
        "SERVER_PORT": "8787",
        "HTTP_HOST": "127.0.0.1:8787",
        "wsgi.url_scheme": "http",
        "wsgi.input": io.BytesIO(body),
        "CONTENT_LENGTH": str(len(body)),
    }
    for key, value in (headers or {}).items():
        normalized = key.upper().replace("-", "_")
        if normalized == "CONTENT_TYPE":
            environ["CONTENT_TYPE"] = value
        else:
            environ[f"HTTP_{normalized}"] = value

    response_status = ""
    response_headers: dict[str, str] = {}

    def start_response(status: str, values: list[tuple[str, str]], _exc_info=None) -> None:
        nonlocal response_status, response_headers
        response_status = status
        response_headers = {key.lower(): value for key, value in values}

    response_body = b"".join(app.wsgi_app(environ, start_response))
    return int(response_status.split(" ", 1)[0]), json.loads(response_body), response_headers


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
