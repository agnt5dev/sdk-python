import asyncio
import json

import pytest

from agnt5 import Context, durable
from agnt5.function import clear_registry, execute_component
from agnt5.durable import BackoffPolicy, RetryPolicy, get_registry
from agnt5.llm import (
    ChatMessage,
    LlmError,
    LlmResponse,
    LlmStreamChunk,
    LlmUsage,
    LlmClient,
    TransportResponse,
)


@pytest.fixture(autouse=True)
def _clear_registry():
    clear_registry()
    yield
    clear_registry()


def test_durable_function_registration():
    retry_policy = RetryPolicy(max_attempts=5)
    backoff_policy = BackoffPolicy(
        strategy="fixed", initial_delay_seconds=2.0, max_delay_seconds=10.0
    )

    @durable.function(name="demo", retries=retry_policy, backoff=backoff_policy)
    async def handler(
        ctx: Context, value: int
    ) -> int:  # pragma: no cover - executed via execute_component
        return value

    definition = get_registry().get("demo")
    assert definition is not None
    assert definition.retries.max_attempts == 5
    assert definition.backoff.strategy == "fixed"


def test_duplicate_registration_raises():
    @durable.function(name="duplicate")
    async def handler(ctx: Context) -> None:  # pragma: no cover
        return None

    with pytest.raises(ValueError):

        @durable.function(name="duplicate")
        async def handler_2(ctx: Context) -> None:  # pragma: no cover
            return None


@pytest.mark.asyncio
async def test_context_step_uses_existing_checkpoint():
    checkpoint_payload = [
        {
            "name": "fetch",
            "key": None,
            "status": "SUCCEEDED",
            "attempt": 1,
            "result": {"value": 42},
            "updated_at": "2024-11-30T12:34:56+00:00",
        }
    ]
    ctx = Context(
        run_id="run-123",
        step_id="step-root",
        attempt=1,
        invocation_id="inv-1",
        service_name="svc",
        component_name="demo",
        step_checkpoints=checkpoint_payload,
    )

    async def failing_step():  # pragma: no cover - should not be executed
        raise AssertionError("should not execute")

    result = await ctx.step("fetch", failing_step)
    assert result == {"value": 42}


@pytest.mark.asyncio
async def test_context_surface_matches_spec_contract():
    resource_marker = object()

    class StubClient:
        async def complete(self, *, prompt: str, model: str, schema, context) -> LlmResponse:
            return LlmResponse(model=model, output=prompt, usage=LlmUsage())

        async def chat(
            self, *, messages, model: str, tools, context, stream: bool = False, on_chunk=None
        ):
            return LlmResponse(model=model, output="chat", usage=LlmUsage())

    metadata = {
        "headers": {"X-Tenant": "acme"},
        "secrets": {"OPENAI_API_KEY": "s3cr3t"},
        "config": {"feature_x": True, "variants": {"ab": "A"}},
        "resources": {"db": resource_marker},
        "parent_run_id": "parent-1",
        "tenant_id": "tenant-42",
    }

    ctx = Context(
        run_id="run-ctx",
        step_id="step-ctx",
        attempt=2,
        invocation_id="inv-ctx",
        service_name="svc",
        component_name="component",
        metadata=metadata,
        llm_client=StubClient(),
    )

    assert ctx.parent_run_id == "parent-1"
    assert ctx.headers()["x-tenant"] == "acme"
    assert ctx.secrets().get("OPENAI_API_KEY") == "s3cr3t"
    assert ctx.config().get("feature_x") is True
    assert ctx.config().variant("ab") == "A"
    assert ctx.metadata["tenant_id"] == "tenant-42"
    assert ctx.resources.db is resource_marker

    tool = ctx.tools.register("echo", handler=lambda *_: "ok", schema={"type": "object"})
    assert tool.name == "echo"
    assert ctx.tools.get("echo") is tool
    assert ctx.tools.handler("echo")() == "ok"
    assert len(ctx.tools.list()) == 1

    await ctx.memory.set("stats", {"count": 1})
    current = await ctx.memory.get("stats")
    assert current == {"count": 1}

    await ctx.memory.append("events", {"type": "created"})
    await ctx.memory.append("events", {"type": "processed"}, limit=10)
    memory_update = ctx.memory.export_state_update()
    assert memory_update is not None
    snapshot = json.loads(memory_update.new_state.decode("utf-8"))
    assert snapshot["stats"] == {"count": 1}
    assert isinstance(memory_update.transitions, list)
    assert memory_update.transitions

    ctx.eval.metric("quality", 0.9)
    await ctx.sleep(0)

    with pytest.raises(KeyError):
        ctx.secrets().get("MISSING")


def test_execute_component_emits_checkpoints():
    @durable.function(name="add")
    async def add(ctx: Context, a: int, b: int) -> int:
        return await ctx.step("sum", lambda: a + b)

    payload = json.dumps({"a": 1, "b": 2}).encode("utf-8")
    context = {
        "invocation_id": "inv-123",
        "service_name": "svc",
        "handler_name": "add",
        "metadata": {"run_id": "run-1", "attempt": "1"},
    }

    result = execute_component("add", payload, context)
    assert json.loads(result.output_data.decode("utf-8")) == 3
    assert "step_checkpoints" in result.metadata

    replay_context = {
        "invocation_id": "inv-456",
        "service_name": "svc",
        "handler_name": "add",
        "metadata": {
            "run_id": "run-1",
            "attempt": "2",
            "step_checkpoints": result.metadata["step_checkpoints"],
        },
    }

    replay_result = execute_component("add", payload, replay_context)
    assert json.loads(replay_result.output_data.decode("utf-8")) == 3
    assert replay_result.metadata.get("step_checkpoints", "[]") == "[]"


@pytest.mark.asyncio
async def test_context_llm_facade_uses_injected_client():
    class StubClient:
        def __init__(self) -> None:
            self.called_with = None

        async def complete(self, *, prompt: str, model: str, schema, context) -> LlmResponse:
            self.called_with = {
                "prompt": prompt,
                "model": model,
                "schema": schema,
                "run_id": context.run_id,
            }
            return LlmResponse(
                model=model,
                output="ok",
                usage=LlmUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            )

        async def chat(
            self, *, messages, model: str, tools, context, stream: bool = False, on_chunk=None
        ):
            self.called_with = {
                "messages": messages,
                "model": model,
                "tools": tools,
                "stream": stream,
                "on_chunk": on_chunk,
                "run_id": context.run_id,
            }
            if stream:
                chunk1 = LlmStreamChunk(index=0, content="hello", model=model)
                chunk2 = LlmStreamChunk(
                    index=0, content=" world", model=model, finish_reason="stop"
                )
                if on_chunk:
                    maybe = on_chunk(chunk1)
                    if asyncio.iscoroutine(maybe):
                        await maybe
                    maybe = on_chunk(chunk2)
                    if asyncio.iscoroutine(maybe):
                        await maybe
                response = LlmResponse(model=model, output="hello world", usage=LlmUsage())
                response.raw["events"] = [chunk1.raw, chunk2.raw]
                return response

            return LlmResponse(model=model, output="chat", usage=LlmUsage())

    client = StubClient()
    ctx = Context(
        run_id="run-llm",
        step_id="step",
        attempt=1,
        invocation_id="inv-llm",
        service_name="svc",
        component_name="component",
        llm_client=client,
    )

    response = await ctx.llm.complete(prompt="hi", model="gpt", schema={"type": "string"})
    assert response.output == "ok"
    assert client.called_with == {
        "prompt": "hi",
        "model": "gpt",
        "schema": {"type": "string"},
        "run_id": "run-llm",
    }

    chat_response = await ctx.llm.chat([ChatMessage(role="user", content="hey")], model="gpt")
    assert chat_response.output == "chat"
    assert client.called_with["stream"] is False

    observed = []

    async def on_chunk(chunk: LlmStreamChunk) -> None:
        observed.append(chunk.content)

    streamed_response = await ctx.llm.chat(
        [ChatMessage(role="user", content="hey")],
        model="gpt",
        stream=True,
        on_chunk=on_chunk,
    )

    assert client.called_with["stream"] is True
    assert "".join(filter(lambda x: isinstance(x, str), observed)) == "hello world"
    assert streamed_response.output == "hello world"

    # Legacy methods remain available for backwards compatibility
    legacy = await ctx.llm_complete(prompt="bye", model="gpt")
    assert legacy.output == "ok"


def test_context_spawn_and_join():
    @durable.function(name="child_double")
    async def child_double(ctx: Context, value: int) -> int:
        return await ctx.step("double", lambda: value * 2)

    @durable.function(name="parent_sum_spawn")
    async def parent_sum_spawn(ctx: Context, numbers: list[int]) -> int:
        handles = [ctx.spawn(child_double, value=number) for number in numbers]
        totals = await ctx.join(handles)
        return sum(totals)

    payload = json.dumps({"numbers": [1, 2, 3]}).encode("utf-8")
    context = {
        "invocation_id": "inv-spawn",
        "service_name": "svc",
        "handler_name": "parent_sum_spawn",
        "metadata": {"run_id": "run-spawn", "attempt": "1"},
    }

    result = execute_component("parent_sum_spawn", payload, context)
    assert json.loads(result.output_data.decode("utf-8")) == 12


def test_context_spawn_handle_await_and_parallel():
    @durable.function(name="child_increment")
    async def child_increment(ctx: Context, value: int) -> int:
        return await ctx.step("increment", lambda: value + 1)

    @durable.function(name="parent_spawn_parallel")
    async def parent_spawn_parallel(ctx: Context, base: int) -> dict[str, list[int]]:
        first = ctx.spawn(child_increment, value=base)
        second = ctx.spawn(child_increment, value=base + 1)
        results = await ctx.parallel(first, second)
        remaining = await ctx.join()
        return {"results": results, "remaining": remaining}

    payload = json.dumps({"base": 4}).encode("utf-8")
    context = {
        "invocation_id": "inv-parallel",
        "service_name": "svc",
        "handler_name": "parent_spawn_parallel",
        "metadata": {"run_id": "run-parallel", "attempt": "1"},
    }

    result = execute_component("parent_spawn_parallel", payload, context)
    decoded = json.loads(result.output_data.decode("utf-8"))
    assert sorted(decoded["results"]) == [5, 6]
    assert decoded["remaining"] == []


@pytest.mark.asyncio
async def test_llm_client_error_mapping():
    async def failing_transport(method, url, *, headers, data):
        payload = {
            "error": {
                "code": "rate_limit",
                "message": "Too many requests",
                "details": {"retry_after": 1},
            }
        }
        return TransportResponse(status=429, headers={}, body=json.dumps(payload).encode("utf-8"))

    client = LlmClient(base_url="http://example.com", transport=failing_transport)

    with pytest.raises(LlmError) as exc_info:
        await client.chat(
            messages=[ChatMessage(role="user", content="hi")],
            model="gpt",
            tools=None,
            context=None,
        )

    err = exc_info.value
    assert err.status == 429
    assert err.code == "rate_limit"
