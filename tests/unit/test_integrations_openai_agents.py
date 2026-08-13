"""Unit tests for OpenAI Agents SDK capture (agnt5.integrations.openai_agents)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("agents")

from agents.tracing.span_data import (  # noqa: E402
    AgentSpanData,
    FunctionSpanData,
    GenerationSpanData,
    HandoffSpanData,
    ResponseSpanData,
)

import agnt5.integrations as integrations  # noqa: E402
from agnt5.context import Context, _current_context, set_current_context  # noqa: E402
from agnt5.integrations import openai_agents  # noqa: E402
from agnt5.integrations.openai_agents import CaptureProcessor  # noqa: E402


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
def _capture_enabled():
    previous_processor = openai_agents._processor
    openai_agents._enabled = True
    yield
    openai_agents._enabled = False
    openai_agents._processor = previous_processor
    integrations._auto_enabled = False


@pytest.fixture
def ctx():
    context = Context(run_id="run-1", correlation_id="root-cid", parent_correlation_id="run-1")
    context._emitter = RecordingEmitter()
    token = set_current_context(context)
    yield context
    _current_context.reset(token)


@pytest.fixture
def processor():
    return CaptureProcessor()


def make_span(span_data, span_id="span-1", parent_id=None, error=None, trace_id="trace-1"):
    return SimpleNamespace(
        trace_id=trace_id,
        span_id=span_id,
        parent_id=parent_id,
        span_data=span_data,
        error=error,
    )


def test_agent_span_lifecycle(ctx, processor):
    span = make_span(AgentSpanData(name="triage", tools=["search"]))
    processor.on_span_start(span)
    processor.on_span_end(span)

    started, completed = ctx._emitter.events
    assert started.event_type == "agent.started"
    assert completed.event_type == "agent.completed"
    assert started.name == "triage"
    assert started.tool_names == ["search"]
    assert started.parent_correlation_id == "root-cid"
    assert completed.correlation_id == started.correlation_id
    assert started.metadata["source"] == "openai_agents"
    assert started.metadata["capture_mode"] == "observed"


def test_generation_span_maps_to_lm_events(ctx, processor):
    agent_span = make_span(AgentSpanData(name="triage"), span_id="agent-1")
    processor.on_span_start(agent_span)
    agent_cid = ctx._emitter.events[0].correlation_id

    gen = make_span(
        GenerationSpanData(
            input=[{"role": "user", "content": "hi"}],
            output=[{"role": "assistant", "content": "hello"}],
            model="gpt-4o-mini",
            model_config={"temperature": 0.3},
            usage={"input_tokens": 7, "output_tokens": 9},
        ),
        span_id="gen-1",
        parent_id="agent-1",
    )
    processor.on_span_start(gen)
    processor.on_span_end(gen)

    lm_started, lm_completed = ctx._emitter.events[1:]
    assert lm_started.event_type == "lm.started"
    assert lm_completed.event_type == "lm.completed"
    assert lm_started.parent_correlation_id == agent_cid
    assert lm_completed.parent_correlation_id == agent_cid
    assert lm_started.name == "openai/gpt-4o-mini"
    assert lm_started.input_data["temperature"] == 0.3
    assert lm_completed.metadata["input_tokens"] == "7"
    assert lm_completed.metadata["output_tokens"] == "9"
    assert lm_completed.metadata["total_tokens"] == "16"
    assert lm_completed.metadata["provider"] == "openai"
    assert all(isinstance(v, str) for v in lm_completed.metadata.values())


def test_unknown_bare_model_is_not_priced_as_openai(ctx, processor):
    span = make_span(GenerationSpanData(model="vendor-model-v1", usage={"input_tokens": 1}))
    processor.on_span_start(span)
    processor.on_span_end(span)

    started, completed = ctx._emitter.events
    assert started.provider == "openai-compatible"
    assert completed.provider == "openai-compatible"
    assert completed.name == "openai-compatible/vendor-model-v1"


def test_response_span_reuses_response_extraction(ctx, processor):
    response = SimpleNamespace(
        model="gpt-4o-2024-08-06",
        status="completed",
        usage=SimpleNamespace(
            input_tokens=20,
            output_tokens=30,
            total_tokens=50,
            input_tokens_details=SimpleNamespace(cached_tokens=4),
        ),
        output_text="agents say hi",
        output=[],
    )
    span = make_span(ResponseSpanData(response=response))
    processor.on_span_start(span)
    processor.on_span_end(span)

    lm_started, lm_completed = ctx._emitter.events
    # Model unknown at start; resolved from the response at end.
    assert lm_started.name == "openai"
    assert lm_completed.name == "openai/gpt-4o-2024-08-06"
    assert lm_completed.metadata["total_tokens"] == "50"
    assert lm_completed.metadata["cached_tokens"] == "4"
    assert lm_completed.output_data["output"] == "agents say hi"


def test_response_span_failure_status_emits_failed(ctx, processor):
    response = SimpleNamespace(
        model="gpt-4o",
        status="failed",
        error=SimpleNamespace(message="response failed"),
        usage=None,
        output_text=None,
        output=[],
    )
    span = make_span(ResponseSpanData(response=response))
    processor.on_span_start(span)
    processor.on_span_end(span)

    started, failed = ctx._emitter.events
    assert failed.event_type == "lm.failed"
    assert failed.error_message == "response failed"
    assert failed.parent_correlation_id == started.parent_correlation_id


def test_function_span_maps_to_tool_call_events(ctx, processor):
    span = make_span(FunctionSpanData(name="get_weather", input='{"city": "SF"}', output="sunny"))
    processor.on_span_start(span)
    processor.on_span_end(span)

    started, completed = ctx._emitter.events
    assert started.event_type == "tool_call.started"
    assert completed.event_type == "tool_call.completed"
    assert started.tool_name == "get_weather"
    assert started.input_data == {"input": '{"city": "SF"}'}
    assert completed.output_data == {"result": "sunny"}
    assert completed.correlation_id == started.correlation_id
    assert started.to_dict()["tool_call_id"] == started.tool_call_id
    assert completed.to_dict()["tool_name"] == "get_weather"


def test_passthrough_span_keeps_parent_chain(ctx, processor):
    agent_span = make_span(AgentSpanData(name="triage"), span_id="agent-1")
    processor.on_span_start(agent_span)
    agent_cid = ctx._emitter.events[0].correlation_id

    handoff = make_span(
        HandoffSpanData(from_agent="triage", to_agent="math"),
        span_id="handoff-1",
        parent_id="agent-1",
    )
    processor.on_span_start(handoff)
    assert len(ctx._emitter.events) == 1  # no event for the handoff itself

    gen = make_span(GenerationSpanData(model="gpt-4o-mini"), span_id="gen-1", parent_id="handoff-1")
    processor.on_span_start(gen)
    # The generation nested under the handoff parents through to the agent.
    assert ctx._emitter.events[1].parent_correlation_id == agent_cid

    processor.on_span_end(handoff)
    assert len(ctx._emitter.events) == 2  # passthrough emits nothing on end


def test_error_span_emits_failed(ctx, processor):
    span = make_span(
        GenerationSpanData(model="gpt-4o-mini"),
        error={"message": "rate limited", "data": None},
    )
    processor.on_span_start(span)
    processor.on_span_end(span)

    started, failed = ctx._emitter.events
    assert failed.event_type == "lm.failed"
    assert failed.error_message == "rate limited"
    assert failed.correlation_id == started.correlation_id


def test_no_context_is_noop(processor):
    span = make_span(AgentSpanData(name="triage"))
    processor.on_span_start(span)
    processor.on_span_end(span)
    assert processor._spans == {}


def test_disabled_processor_is_inert(ctx, processor):
    openai_agents._enabled = False
    span = make_span(AgentSpanData(name="triage"))
    processor.on_span_start(span)
    processor.on_span_end(span)
    assert ctx._emitter.events == []


def test_trace_end_purges_state(ctx, processor):
    processor.on_span_start(make_span(AgentSpanData(name="a"), span_id="s1"))
    processor.on_span_start(make_span(AgentSpanData(name="b"), span_id="s2"))
    processor.on_trace_end(SimpleNamespace(trace_id="trace-1"))
    assert processor._spans == {}


def test_suppresses_client_capture_only_for_owned_model_spans(ctx, processor, monkeypatch):
    model_span = make_span(ResponseSpanData(), span_id="model")
    processor.on_span_start(model_span)
    monkeypatch.setattr(openai_agents, "_processor", processor)
    monkeypatch.setattr(openai_agents, "get_current_span", lambda: model_span)
    assert openai_agents.suppresses_client_capture() is True

    tool_span = make_span(FunctionSpanData(name="t", input=None, output=None), span_id="tool")
    processor.on_span_start(tool_span)
    monkeypatch.setattr(openai_agents, "get_current_span", lambda: tool_span)
    assert openai_agents.suppresses_client_capture() is False

    unowned_model_span = make_span(ResponseSpanData(), span_id="unowned")
    monkeypatch.setattr(openai_agents, "get_current_span", lambda: unowned_model_span)
    assert openai_agents.suppresses_client_capture() is False


def test_raw_client_capture_suppressed_inside_model_span(ctx, monkeypatch):
    from openai.resources.chat.completions import AsyncCompletions, Completions
    from openai.resources.embeddings import AsyncEmbeddings, Embeddings
    from openai.resources.responses import AsyncResponses, Responses

    from agnt5.integrations import openai as openai_integration

    patched_classes = (
        Completions,
        AsyncCompletions,
        Embeddings,
        AsyncEmbeddings,
        Responses,
        AsyncResponses,
    )
    real = {cls: cls.create for cls in patched_classes}
    calls = []

    def stub(self, *args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model="gpt-4o-mini", usage=None, choices=[])

    Completions.create = stub
    try:
        openai_integration.enable()
        processor = CaptureProcessor()
        model_span = make_span(ResponseSpanData(), span_id="model")
        processor.on_span_start(model_span)
        ctx._emitter.events.clear()
        monkeypatch.setattr(openai_agents, "_processor", processor)
        monkeypatch.setattr(openai_agents, "get_current_span", lambda: model_span)
        inst = Completions.__new__(Completions)
        inst.create(model="gpt-4o-mini", messages=[])
        assert len(calls) == 1
        assert ctx._emitter.events == []  # suppressed: the span capture owns it
    finally:
        openai_integration._patched = False
        openai_integration._originals.clear()
        openai_integration._wrappers.clear()
        for cls, method in real.items():
            cls.create = method


def test_enable_registers_only_one_processor_across_reenable(monkeypatch):
    registered = []
    monkeypatch.setattr(openai_agents, "_enabled", False)
    monkeypatch.setattr(openai_agents, "_processor", None)
    monkeypatch.setattr(openai_agents, "add_trace_processor", registered.append)

    assert openai_agents.enable() is True
    first = openai_agents._processor
    openai_agents.disable()
    assert openai_agents.enable() is True

    assert registered == [first]


def test_unsupported_agents_version_is_not_registered(monkeypatch):
    registered = []
    monkeypatch.setattr(openai_agents, "_enabled", False)
    monkeypatch.setattr(openai_agents, "_processor", None)
    monkeypatch.setattr(openai_agents, "add_trace_processor", registered.append)
    monkeypatch.setattr(openai_agents, "supported_package_version", lambda *a, **k: False)

    assert openai_agents.enable() is False
    assert registered == []


def test_processor_registration_failure_is_best_effort(monkeypatch):
    def fail_registration(processor):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(openai_agents, "_enabled", False)
    monkeypatch.setattr(openai_agents, "_processor", None)
    monkeypatch.setattr(openai_agents, "add_trace_processor", fail_registration)

    assert openai_agents.enable() is False
    assert openai_agents._processor is None


def test_processor_errors_never_escape(processor):
    class BrokenSpan:
        @property
        def trace_id(self):
            raise RuntimeError("broken tracing object")

    assert processor.on_span_start(BrokenSpan()) is None
