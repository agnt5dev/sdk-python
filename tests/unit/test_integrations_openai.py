"""Unit tests for third-party OpenAI SDK capture (agnt5.integrations)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("openai")

from openai.resources.chat.completions import AsyncCompletions, Completions  # noqa: E402

import agnt5.integrations as integrations  # noqa: E402
from agnt5.context import Context, _current_context, set_current_context  # noqa: E402
from agnt5.integrations import openai as openai_integration  # noqa: E402
from agnt5.integrations._common import build_lm_completed  # noqa: E402


class RecordingEmitter:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event):
        self.events.append(event)
        return event

    async def emit_async(self, event):
        self.events.append(event)
        return event


@pytest.fixture(autouse=True)
def _restore_openai():
    from openai.resources.embeddings import AsyncEmbeddings, Embeddings
    from openai.resources.responses import AsyncResponses, Responses

    openai_integration.disable()
    patched_classes = (
        Completions,
        AsyncCompletions,
        Embeddings,
        AsyncEmbeddings,
        Responses,
        AsyncResponses,
    )
    real = {cls: cls.create for cls in patched_classes}
    yield
    openai_integration._patched = False
    openai_integration._originals.clear()
    openai_integration._wrappers.clear()
    integrations._auto_enabled = False
    for cls, method in real.items():
        cls.create = method


@pytest.fixture
def ctx():
    context = Context(run_id="run-1", correlation_id="root-cid", parent_correlation_id="run-1")
    context._emitter = RecordingEmitter()
    token = set_current_context(context)
    yield context
    _current_context.reset(token)


def fake_response(model="gpt-4o-mini-2024-07-18", tool_calls=None):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=34,
            total_tokens=46,
            prompt_tokens_details=SimpleNamespace(cached_tokens=5),
        ),
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="hi there", tool_calls=tool_calls),
            )
        ],
    )


def test_completed_metadata_matches_cross_sdk_contract():
    event = build_lm_completed(
        source="openai",
        name="openai/gpt-4o-mini",
        model="openai/gpt-4o-mini",
        provider="openai",
        correlation_id="lm-1",
        parent_correlation_id="fn-1",
        duration_ms=12,
        input_tokens=5,
        output_tokens=2,
        total_tokens=7,
        cached_tokens=3,
    )

    assert event.metadata == {
        "name": "openai/gpt-4o-mini",
        "source": "openai",
        "capture_mode": "observed",
        "model": "openai/gpt-4o-mini",
        "provider": "openai",
        "duration_ms": "12",
        "input_tokens": "5",
        "output_tokens": "2",
        "total_tokens": "7",
        "cached_tokens": "3",
    }


def stub_sync(calls, response=None, error=None):
    def create(self, *args, **kwargs):
        calls.append(kwargs)
        if error is not None:
            raise error
        return response if response is not None else fake_response()

    Completions.create = create


def stub_async(calls, response=None, error=None):
    async def create(self, *args, **kwargs):
        calls.append(kwargs)
        if error is not None:
            raise error
        return response if response is not None else fake_response()

    AsyncCompletions.create = create


def make_call(**kwargs):
    inst = Completions.__new__(Completions)
    return inst.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hello"},
        ],
        **kwargs,
    )


def test_sync_capture_emits_started_and_completed(ctx):
    calls = []
    stub_sync(calls)
    assert openai_integration.enable() is True

    result = make_call(temperature=0.2, max_tokens=64)

    assert len(calls) == 1
    assert result.model == "gpt-4o-mini-2024-07-18"
    started, completed = ctx._emitter.events
    assert started.event_type == "lm.started"
    assert completed.event_type == "lm.completed"

    # Same span, parented to the enclosing component execution.
    assert started.correlation_id == completed.correlation_id
    assert started.parent_correlation_id == "root-cid"
    assert completed.parent_correlation_id == "root-cid"

    # Response model preferred over request alias, provider-prefixed.
    assert completed.name == "openai/gpt-4o-mini-2024-07-18"
    assert started.name == "openai/gpt-4o-mini"

    # The runtime reads only string-valued metadata.
    assert completed.metadata["model"] == "openai/gpt-4o-mini-2024-07-18"
    assert completed.metadata["provider"] == "openai"
    assert completed.metadata["input_tokens"] == "12"
    assert completed.metadata["output_tokens"] == "34"
    assert completed.metadata["total_tokens"] == "46"
    assert completed.metadata["cached_tokens"] == "5"
    assert completed.metadata["source"] == "openai"
    assert completed.metadata["capture_mode"] == "observed"
    assert all(isinstance(v, str) for v in completed.metadata.values())

    # Native LM client payload shape.
    assert started.input_data["system_prompt"] == "be nice"
    assert started.input_data["messages"] == [{"role": "user", "content": "hello"}]
    assert started.input_data["temperature"] == 0.2
    assert started.input_data["max_tokens"] == 64
    assert completed.output_data["output"] == "hi there"
    assert completed.finish_reason == "stop"


async def test_async_capture_emits_started_and_completed(ctx):
    calls = []
    stub_async(calls)
    openai_integration.enable()

    inst = AsyncCompletions.__new__(AsyncCompletions)
    await inst.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    assert len(calls) == 1
    started, completed = ctx._emitter.events
    assert started.event_type == "lm.started"
    assert completed.event_type == "lm.completed"
    assert completed.metadata["input_tokens"] == "12"


def test_error_emits_failed_and_propagates(ctx):
    calls = []
    stub_sync(calls, error=RuntimeError("boom"))
    openai_integration.enable()

    with pytest.raises(RuntimeError, match="boom"):
        make_call()

    started, failed = ctx._emitter.events
    assert started.event_type == "lm.started"
    assert failed.event_type == "lm.failed"
    assert failed.error_code == "RuntimeError"
    assert failed.error_message == "boom"
    assert failed.error_traceback is not None
    assert failed.correlation_id == started.correlation_id


def test_tool_calls_serialized(ctx):
    tool_calls = [
        SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="get_weather", arguments='{"city": "SF"}'),
        )
    ]
    stub_sync([], response=fake_response(tool_calls=tool_calls))
    openai_integration.enable()

    make_call()

    completed = ctx._emitter.events[1]
    assert completed.output_data["tool_calls"] == [
        {"id": "call_1", "name": "get_weather", "arguments": '{"city": "SF"}'}
    ]


def test_no_context_passthrough():
    calls = []
    stub_sync(calls)
    openai_integration.enable()

    result = make_call()

    assert len(calls) == 1
    assert result.model == "gpt-4o-mini-2024-07-18"


def chat_chunk(content=None, finish_reason=None, usage=None, tool_calls=None):
    return SimpleNamespace(
        model="gpt-4o-mini-2024-07-18",
        usage=usage,
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                delta=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ],
    )


def final_usage():
    return SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=34,
        total_tokens=46,
        prompt_tokens_details=SimpleNamespace(cached_tokens=5),
    )


class FakeStream:
    def __init__(self, chunks, error=None):
        self._chunks = iter(chunks)
        self._error = error
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            if self._error is not None:
                raise self._error from None
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        self.closed = True


class FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = iter(chunks)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()
        return False

    async def close(self):
        self.closed = True


def test_streaming_capture_accumulates_and_completes(ctx):
    chunks = [
        chat_chunk(content="Hel"),
        chat_chunk(content="lo"),
        chat_chunk(finish_reason="stop", usage=final_usage()),
    ]
    stub_sync([], response=FakeStream(chunks))
    openai_integration.enable()

    received = list(make_call(stream=True))

    assert len(received) == 3
    started, completed = ctx._emitter.events
    assert started.event_type == "lm.started"
    assert completed.event_type == "lm.completed"
    assert completed.name == "openai/gpt-4o-mini-2024-07-18"
    assert completed.output_data["output"] == "Hello"
    assert completed.metadata["input_tokens"] == "12"
    assert completed.metadata["output_tokens"] == "34"
    assert completed.metadata["cached_tokens"] == "5"
    assert completed.finish_reason == "stop"


def test_streaming_without_usage_still_sets_model(ctx):
    stub_sync([], response=FakeStream([chat_chunk(content="hi", finish_reason="stop")]))
    openai_integration.enable()

    list(make_call(stream=True))

    completed = ctx._emitter.events[1]
    # Model must be non-empty or the runtime drops the event.
    assert completed.metadata["model"] == "openai/gpt-4o-mini-2024-07-18"
    assert completed.metadata["input_tokens"] == "0"


def test_streaming_tool_call_deltas_aggregate(ctx):
    def tc(index, id=None, name=None, arguments=None):
        return SimpleNamespace(
            index=index,
            id=id,
            function=SimpleNamespace(name=name, arguments=arguments),
        )

    chunks = [
        chat_chunk(tool_calls=[tc(0, id="call_1", name="get_weather", arguments='{"ci')]),
        chat_chunk(tool_calls=[tc(0, arguments='ty": "SF"}')]),
        chat_chunk(finish_reason="tool_calls"),
    ]
    stub_sync([], response=FakeStream(chunks))
    openai_integration.enable()

    list(make_call(stream=True))

    completed = ctx._emitter.events[1]
    assert completed.output_data["tool_calls"] == [
        {"id": "call_1", "name": "get_weather", "arguments": '{"city": "SF"}'}
    ]


def test_streaming_error_emits_failed(ctx):
    stub_sync([], response=FakeStream([chat_chunk(content="x")], error=RuntimeError("mid-stream")))
    openai_integration.enable()

    with pytest.raises(RuntimeError, match="mid-stream"):
        list(make_call(stream=True))

    started, failed = ctx._emitter.events
    assert failed.event_type == "lm.failed"
    assert failed.error_message == "mid-stream"


def test_streaming_early_close_emits_failed(ctx):
    inner = FakeStream([chat_chunk(content="partial"), chat_chunk(content="never-read")])
    stub_sync([], response=inner)
    openai_integration.enable()

    with make_call(stream=True) as stream:
        next(iter(stream))

    assert inner.closed
    failed = ctx._emitter.events[1]
    assert failed.event_type == "lm.failed"
    assert failed.error_message == "OpenAI stream closed before exhaustion"
    # Emitted exactly once even though close + exit both fire.
    assert len(ctx._emitter.events) == 2


def test_streaming_close_after_terminal_chunk_emits_completed(ctx):
    inner = FakeStream([chat_chunk(content="done", finish_reason="stop", usage=final_usage())])
    stub_sync([], response=inner)
    openai_integration.enable()

    with make_call(stream=True) as stream:
        next(iter(stream))

    completed = ctx._emitter.events[1]
    assert completed.event_type == "lm.completed"
    assert completed.output_data["output"] == "done"
    assert completed.finish_reason == "stop"


async def test_async_streaming_capture(ctx):
    chunks = [chat_chunk(content="async"), chat_chunk(finish_reason="stop", usage=final_usage())]
    stub_async([], response=FakeAsyncStream(chunks))
    openai_integration.enable()

    inst = AsyncCompletions.__new__(AsyncCompletions)
    stream = await inst.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    received = [chunk async for chunk in stream]

    assert len(received) == 2
    started, completed = ctx._emitter.events
    assert completed.event_type == "lm.completed"
    assert completed.output_data["output"] == "async"
    assert completed.metadata["total_tokens"] == "46"


def responses_response():
    return SimpleNamespace(
        model="gpt-4o-2024-08-06",
        status="completed",
        usage=SimpleNamespace(
            input_tokens=20,
            output_tokens=30,
            total_tokens=50,
            input_tokens_details=SimpleNamespace(cached_tokens=4),
        ),
        output_text="responses says hi",
        output=[
            SimpleNamespace(type="function_call", call_id="call_9", name="lookup", arguments="{}")
        ],
    )


def test_responses_capture(ctx):
    from openai.resources.responses import Responses

    def create(self, *args, **kwargs):
        return responses_response()

    Responses.create = create
    openai_integration.enable()

    inst = Responses.__new__(Responses)
    inst.create(model="gpt-4o", input="hello", instructions="be brief")

    started, completed = ctx._emitter.events
    assert started.input_data["system_prompt"] == "be brief"
    assert started.input_data["messages"] == "hello"
    assert completed.name == "openai/gpt-4o-2024-08-06"
    assert completed.metadata["input_tokens"] == "20"
    assert completed.metadata["output_tokens"] == "30"
    assert completed.metadata["cached_tokens"] == "4"
    assert completed.finish_reason == "completed"
    assert completed.output_data["output"] == "responses says hi"
    assert completed.output_data["tool_calls"] == [
        {"id": "call_9", "name": "lookup", "arguments": "{}"}
    ]


def test_responses_streaming_capture(ctx):
    from openai.resources.responses import Responses

    events = [
        SimpleNamespace(type="response.output_text.delta", delta="hi"),
        SimpleNamespace(type="response.completed", response=responses_response()),
    ]

    def create(self, *args, **kwargs):
        return FakeStream(events)

    Responses.create = create
    openai_integration.enable()

    inst = Responses.__new__(Responses)
    list(inst.create(model="gpt-4o", input="hello", stream=True))

    completed = ctx._emitter.events[1]
    assert completed.event_type == "lm.completed"
    assert completed.metadata["total_tokens"] == "50"
    assert completed.output_data["output"] == "responses says hi"


@pytest.mark.parametrize("status", ["failed", "incomplete", "in_progress"])
def test_responses_nonterminal_status_is_not_reported_completed(ctx, status):
    from openai.resources.responses import Responses

    response = responses_response()
    response.status = status
    response.error = (
        SimpleNamespace(message="provider rejected request") if status == "failed" else None
    )
    Responses.create = lambda self, *args, **kwargs: response
    openai_integration.enable()

    Responses.__new__(Responses).create(model="gpt-4o", input="hello")

    failed = ctx._emitter.events[1]
    assert failed.event_type == "lm.failed"
    expected = "provider rejected request" if status == "failed" else f"response status is {status}"
    assert failed.error_message == expected


def test_responses_stream_without_terminal_event_fails(ctx):
    from openai.resources.responses import Responses

    Responses.create = lambda self, *args, **kwargs: FakeStream(
        [SimpleNamespace(type="response.output_text.delta", delta="partial")]
    )
    openai_integration.enable()

    list(Responses.__new__(Responses).create(model="gpt-4o", input="hello", stream=True))

    failed = ctx._emitter.events[1]
    assert failed.event_type == "lm.failed"
    assert failed.error_message == "Responses API stream ended without a terminal response"


def test_embeddings_capture(ctx):
    from openai.resources.embeddings import Embeddings

    def create(self, *args, **kwargs):
        return SimpleNamespace(
            model="text-embedding-3-small",
            usage=SimpleNamespace(prompt_tokens=8, total_tokens=8),
            data=[object(), object(), object()],
        )

    Embeddings.create = create
    openai_integration.enable()

    inst = Embeddings.__new__(Embeddings)
    inst.create(model="text-embedding-3-small", input=["a", "b", "c"])

    started, completed = ctx._emitter.events
    assert started.input_data == {"input": ["a", "b", "c"], "input_count": 3}
    assert completed.name == "openai/text-embedding-3-small"
    assert completed.metadata["input_tokens"] == "8"
    assert completed.metadata["output_tokens"] == "0"
    assert completed.output_data == {"embeddings_count": 3}


def test_double_enable_is_idempotent(ctx):
    calls = []
    stub_sync(calls)
    assert openai_integration.enable() is True
    assert openai_integration.enable() is True

    make_call()

    assert len(calls) == 1
    assert len(ctx._emitter.events) == 2


def test_disable_restores_original(ctx):
    calls = []
    stub_sync(calls)
    openai_integration.enable()
    openai_integration.disable()

    make_call()

    assert len(calls) == 1
    assert ctx._emitter.events == []


def test_master_kill_switch(monkeypatch):
    monkeypatch.setenv("AGNT5_CAPTURE", "off")
    integrations.auto_enable()
    assert openai_integration._patched is False


def test_per_library_kill_switch(monkeypatch):
    monkeypatch.setenv("AGNT5_CAPTURE_OPENAI", "0")
    integrations.auto_enable()
    assert openai_integration._patched is False


def test_auto_enable_patches_when_library_present():
    integrations.auto_enable()
    assert openai_integration._patched is True


def test_auto_enable_never_raises(monkeypatch):
    monkeypatch.setattr(integrations, "enable_openai_capture", lambda: 1 / 0)
    integrations.auto_enable()


def test_content_capture_disabled(monkeypatch, ctx):
    monkeypatch.setenv("AGNT5_LLM_CAPTURE_CONTENT", "off")
    stub_sync([])
    openai_integration.enable()

    make_call()

    started, completed = ctx._emitter.events
    assert started.input_data is None
    assert completed.output_data is None
    # Metrics still flow even with content capture off.
    assert completed.metadata["input_tokens"] == "12"


def test_generator_inputs_are_not_consumed_by_capture(ctx):
    received_messages = []
    received_tools = []

    def create(self, *args, **kwargs):
        received_messages.extend(kwargs["messages"])
        received_tools.extend(kwargs["tools"])
        return fake_response()

    Completions.create = create
    openai_integration.enable()
    messages = ({"role": "user", "content": value} for value in ("one", "two"))
    tools = ({"type": "function", "name": value} for value in ("a", "b"))

    Completions.__new__(Completions).create(model="gpt-4o-mini", messages=messages, tools=tools)

    assert [message["content"] for message in received_messages] == ["one", "two"]
    assert [tool["name"] for tool in received_tools] == ["a", "b"]
    assert ctx._emitter.events[0].input_data["messages"] == received_messages
    assert ctx._emitter.events[0].input_data["tools_count"] == 2


def test_capture_snapshot_of_unbounded_iterator_is_bounded(ctx):
    received = []

    def create(self, *args, **kwargs):
        iterator = iter(kwargs["messages"])
        received.extend([next(iterator), next(iterator)])
        return fake_response()

    def messages():
        index = 0
        while True:
            yield {"role": "user", "content": str(index)}
            index += 1

    Completions.create = create
    openai_integration.enable()

    Completions.__new__(Completions).create(model="gpt-4o-mini", messages=messages())

    assert [message["content"] for message in received] == ["0", "1"]
    assert len(ctx._emitter.events[0].input_data["messages"]) == 128


def test_redacted_content_and_bounded_strings(monkeypatch, ctx):
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    monkeypatch.setenv("AGNT5_CAPTURE_CONTENT_MODE", "redacted")
    monkeypatch.setenv("AGNT5_CAPTURE_MAX_CONTENT_CHARS", "256")
    response = fake_response()
    response.choices[0].message.content = f"Bearer {secret} " + "x" * 400
    stub_sync([], response=response)
    openai_integration.enable()

    Completions.__new__(Completions).create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": secret}],
        max_tokens=64,
    )

    started, completed = ctx._emitter.events
    assert started.input_data["messages"][0]["content"] == "[REDACTED]"
    assert started.input_data["max_tokens"] == 64
    assert "Bearer [REDACTED]" in completed.output_data["output"]
    assert completed.output_data["output"].endswith("...[truncated]")
    assert len(completed.output_data["output"]) < 280


def test_lifecycle_subclass_fields_reach_wire_payload(ctx):
    stub_sync([])
    openai_integration.enable()

    make_call()

    started, completed = ctx._emitter.events
    assert started.to_dict()["provider"] == "openai"
    assert started.to_dict()["model"] == "openai/gpt-4o-mini"
    assert completed.to_dict()["input_tokens"] == 12
    assert completed.to_dict()["finish_reason"] == "stop"


def test_metadata_only_errors_do_not_leak_details(monkeypatch, ctx):
    monkeypatch.setenv("AGNT5_CAPTURE_CONTENT_MODE", "metadata-only")
    stub_sync([], error=RuntimeError("secret customer payload"))
    openai_integration.enable()

    with pytest.raises(RuntimeError, match="secret customer payload"):
        make_call()

    failed = ctx._emitter.events[1]
    assert failed.error_message == "[error details omitted by capture policy]"
    assert failed.error_traceback is None


def test_invalid_content_mode_fails_closed(monkeypatch, ctx):
    monkeypatch.setenv("AGNT5_CAPTURE_CONTENT_MODE", "ful")
    stub_sync([])
    openai_integration.enable()

    make_call()

    started, completed = ctx._emitter.events
    assert started.input_data is None
    assert completed.output_data is None


def test_unsupported_openai_version_is_not_patched(monkeypatch):
    monkeypatch.setattr(openai_integration, "supported_package_version", lambda *a, **k: False)

    assert openai_integration.enable() is False
    assert openai_integration._patched is False
