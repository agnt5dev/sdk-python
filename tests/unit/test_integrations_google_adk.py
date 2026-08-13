"""Unit tests for Google ADK capture (agnt5.integrations.google_adk)."""

from __future__ import annotations

import asyncio
import warnings
from types import SimpleNamespace

import pytest

pytest.importorskip("google.adk")

import agnt5.integrations as integrations  # noqa: E402
from agnt5.context import Context, _current_context, set_current_context  # noqa: E402
from agnt5.integrations import google_adk  # noqa: E402
from agnt5.integrations.google_adk import CapturePlugin  # noqa: E402


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
def _restore_adk():
    google_adk.disable()
    yield
    google_adk.disable()
    integrations._auto_enabled = False


@pytest.fixture
def ctx():
    context = Context(run_id="run-1", correlation_id="root-cid", parent_correlation_id="run-1")
    context._emitter = RecordingEmitter()
    token = set_current_context(context)
    yield context
    _current_context.reset(token)


@pytest.fixture
def plugin():
    google_adk._patched = True
    yield CapturePlugin()
    google_adk._patched = False


class FakeCallbackContext(SimpleNamespace):
    def __init__(self, invocation_id="inv-1", agent_name="researcher"):
        super().__init__(invocation_id=invocation_id, agent_name=agent_name)


class FakeToolContext(SimpleNamespace):
    def __init__(self, invocation_id="inv-1", agent_name="researcher", function_call_id="fc-1"):
        super().__init__(
            invocation_id=invocation_id,
            agent_name=agent_name,
            function_call_id=function_call_id,
        )


def fake_agent(name="researcher", model="gemini-2.0-flash"):
    return SimpleNamespace(name=name, model=model)


def fake_llm_request(model="gemini-2.0-flash"):
    return SimpleNamespace(
        model=model,
        config=SimpleNamespace(
            system_instruction="be thorough", temperature=0.1, max_output_tokens=256
        ),
        contents=[],
        tools_dict={"search": object()},
    )


def fake_llm_response(partial=None):
    return SimpleNamespace(
        model_version="gemini-2.0-flash-001",
        partial=partial,
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=20,
            total_token_count=30,
            cached_content_token_count=2,
        ),
        finish_reason=SimpleNamespace(name="STOP"),
        content=SimpleNamespace(parts=[SimpleNamespace(text="the answer", function_call=None)]),
    )


async def test_agent_lifecycle(ctx, plugin):
    cb = FakeCallbackContext()
    await plugin.before_agent_callback(agent=fake_agent(), callback_context=cb)
    await plugin.after_agent_callback(agent=fake_agent(), callback_context=cb)

    started, completed = ctx._emitter.events
    assert started.event_type == "agent.started"
    assert completed.event_type == "agent.completed"
    assert started.name == "researcher"
    assert started.agent_model == "gemini-2.0-flash"
    assert started.parent_correlation_id == "root-cid"
    assert completed.correlation_id == started.correlation_id
    assert completed.parent_correlation_id == started.parent_correlation_id
    assert started.metadata["source"] == "google_adk"
    assert started.metadata["capture_mode"] == "observed"
    assert started.to_dict()["agent_model"] == "gemini-2.0-flash"


async def test_nested_agent_parenting(ctx, plugin):
    outer = FakeCallbackContext(agent_name="coordinator")
    inner = FakeCallbackContext(agent_name="specialist")
    await plugin.before_agent_callback(agent=fake_agent("coordinator"), callback_context=outer)
    await plugin.before_agent_callback(agent=fake_agent("specialist"), callback_context=inner)

    outer_started, inner_started = ctx._emitter.events
    assert inner_started.parent_correlation_id == outer_started.correlation_id


async def test_parallel_sibling_agents_keep_their_shared_parent(ctx, plugin):
    outer = FakeCallbackContext(agent_name="coordinator")
    await plugin.before_agent_callback(agent=fake_agent("coordinator"), callback_context=outer)
    outer_cid = ctx._emitter.events[0].correlation_id

    async def run_child(name):
        callback = FakeCallbackContext(agent_name=name)
        await plugin.before_agent_callback(agent=fake_agent(name), callback_context=callback)
        await asyncio.sleep(0)
        await plugin.after_agent_callback(agent=fake_agent(name), callback_context=callback)

    await asyncio.gather(run_child("researcher"), run_child("writer"))
    await plugin.after_agent_callback(agent=fake_agent("coordinator"), callback_context=outer)

    child_events = [
        event for event in ctx._emitter.events if event.name in {"researcher", "writer"}
    ]
    assert len(child_events) == 4
    assert {event.parent_correlation_id for event in child_events} == {outer_cid}
    for name in ("researcher", "writer"):
        started = next(
            event
            for event in child_events
            if event.name == name and event.event_type.endswith("started")
        )
        completed = next(
            event
            for event in child_events
            if event.name == name and event.event_type.endswith("completed")
        )
        assert completed.correlation_id == started.correlation_id


async def test_model_lifecycle(ctx, plugin):
    agent_cb = FakeCallbackContext()
    await plugin.before_agent_callback(agent=fake_agent(), callback_context=agent_cb)
    agent_cid = ctx._emitter.events[0].correlation_id

    cb = FakeCallbackContext()
    await plugin.before_model_callback(callback_context=cb, llm_request=fake_llm_request())
    await plugin.after_model_callback(callback_context=cb, llm_response=fake_llm_response())

    lm_started, lm_completed = ctx._emitter.events[1:]
    assert lm_started.event_type == "lm.started"
    assert lm_completed.event_type == "lm.completed"
    assert lm_started.parent_correlation_id == agent_cid
    assert lm_started.name == "google/gemini-2.0-flash"
    assert lm_started.input_data["system_prompt"] == "be thorough"
    assert lm_started.input_data["tools_count"] == 1

    # Resolved model version preferred; provider google; all-string metadata.
    assert lm_completed.name == "google/gemini-2.0-flash-001"
    assert lm_completed.metadata["provider"] == "google"
    assert lm_completed.metadata["input_tokens"] == "10"
    assert lm_completed.metadata["output_tokens"] == "20"
    assert lm_completed.metadata["total_tokens"] == "30"
    assert lm_completed.metadata["cached_tokens"] == "2"
    assert lm_completed.finish_reason == "STOP"
    assert lm_completed.output_data["output"] == "the answer"
    assert all(isinstance(v, str) for v in lm_completed.metadata.values())
    wire = lm_completed.to_dict()
    assert wire["provider"] == "google"
    assert wire["finish_reason"] == "STOP"


async def test_partial_model_responses_do_not_close_span(ctx, plugin):
    cb = FakeCallbackContext()
    await plugin.before_model_callback(callback_context=cb, llm_request=fake_llm_request())
    await plugin.after_model_callback(
        callback_context=cb, llm_response=fake_llm_response(partial=True)
    )
    assert len(ctx._emitter.events) == 1  # started only

    await plugin.after_model_callback(callback_context=cb, llm_response=fake_llm_response())
    assert len(ctx._emitter.events) == 2
    assert ctx._emitter.events[1].event_type == "lm.completed"


async def test_model_error(ctx, plugin):
    cb = FakeCallbackContext()
    await plugin.before_model_callback(callback_context=cb, llm_request=fake_llm_request())
    await plugin.on_model_error_callback(
        callback_context=cb, llm_request=fake_llm_request(), error=RuntimeError("quota")
    )

    started, failed = ctx._emitter.events
    assert failed.event_type == "lm.failed"
    assert failed.correlation_id == started.correlation_id
    assert failed.error_message == "quota"


async def test_tool_lifecycle(ctx, plugin):
    agent_cb = FakeCallbackContext()
    await plugin.before_agent_callback(agent=fake_agent(), callback_context=agent_cb)
    agent_cid = ctx._emitter.events[0].correlation_id

    tool = SimpleNamespace(name="search")
    tool_cb = FakeToolContext()
    await plugin.before_tool_callback(tool=tool, tool_args={"q": "agnt5"}, tool_context=tool_cb)
    await plugin.after_tool_callback(
        tool=tool, tool_args={"q": "agnt5"}, tool_context=tool_cb, result={"hits": 3}
    )

    tool_started, tool_completed = ctx._emitter.events[1:]
    assert tool_started.event_type == "tool_call.started"
    assert tool_completed.event_type == "tool_call.completed"
    assert tool_started.parent_correlation_id == agent_cid
    assert tool_started.tool_name == "search"
    assert tool_started.tool_call_id == "fc-1"
    assert tool_started.input_data == {"q": "agnt5"}
    assert tool_completed.output_data == {"result": {"hits": 3}}
    assert tool_completed.correlation_id == tool_started.correlation_id


async def test_tool_error(ctx, plugin):
    tool = SimpleNamespace(name="search")
    tool_cb = FakeToolContext()
    await plugin.before_tool_callback(tool=tool, tool_args={}, tool_context=tool_cb)
    await plugin.on_tool_error_callback(
        tool=tool, tool_args={}, tool_context=tool_cb, error=ValueError("bad arg")
    )

    started, failed = ctx._emitter.events
    assert failed.event_type == "tool_call.failed"
    assert failed.error_code == "ValueError"
    assert failed.correlation_id == started.correlation_id


async def test_tool_without_function_call_id_still_closes(ctx, plugin):
    tool = SimpleNamespace(name="search")
    tool_cb = FakeToolContext(function_call_id=None)
    await plugin.before_tool_callback(tool=tool, tool_args={}, tool_context=tool_cb)
    await plugin.after_tool_callback(
        tool=tool, tool_args={}, tool_context=tool_cb, result={"ok": True}
    )

    started, completed = ctx._emitter.events
    assert started.tool_call_id
    assert completed.tool_call_id == started.tool_call_id
    assert completed.correlation_id == started.correlation_id


async def test_unpaired_after_is_skipped(ctx, plugin):
    # ADK skip semantics: after_* without before_* must not emit or crash.
    await plugin.after_agent_callback(agent=fake_agent(), callback_context=FakeCallbackContext())
    await plugin.after_model_callback(
        callback_context=FakeCallbackContext(), llm_response=fake_llm_response()
    )
    await plugin.after_tool_callback(
        tool=SimpleNamespace(name="t"),
        tool_args={},
        tool_context=FakeToolContext(),
        result={},
    )
    assert ctx._emitter.events == []


async def test_no_context_is_noop(plugin):
    cb = FakeCallbackContext()
    assert await plugin.before_agent_callback(agent=fake_agent(), callback_context=cb) is None
    assert (
        await plugin.before_model_callback(callback_context=cb, llm_request=fake_llm_request())
        is None
    )
    assert plugin._agent_spans == {}
    assert plugin._model_spans == {}


async def test_after_run_purges_state(ctx, plugin):
    await plugin.before_agent_callback(agent=fake_agent(), callback_context=FakeCallbackContext())
    await plugin.before_model_callback(
        callback_context=FakeCallbackContext(), llm_request=fake_llm_request()
    )
    await plugin.after_run_callback(invocation_context=SimpleNamespace(invocation_id="inv-1"))

    assert plugin._agent_spans == {}
    assert plugin._agent_stack.get() == ()
    assert plugin._model_spans == {}


async def test_content_capture_disabled(monkeypatch, ctx, plugin):
    monkeypatch.setenv("AGNT5_LLM_CAPTURE_CONTENT", "off")
    cb = FakeCallbackContext()
    await plugin.before_model_callback(callback_context=cb, llm_request=fake_llm_request())
    await plugin.after_model_callback(callback_context=cb, llm_response=fake_llm_response())

    started, completed = ctx._emitter.events
    assert started.input_data is None
    assert completed.output_data is None
    assert completed.metadata["input_tokens"] == "10"


async def test_litellm_provider_prefix_is_preserved(ctx, plugin):
    cb = FakeCallbackContext()
    await plugin.before_model_callback(
        callback_context=cb, llm_request=fake_llm_request(model="openai/gpt-4o-mini")
    )
    response = fake_llm_response()
    response.model_version = "gpt-4o-mini-2024-07-18"
    await plugin.after_model_callback(callback_context=cb, llm_response=response)

    started, completed = ctx._emitter.events
    assert started.provider == "openai"
    assert started.name == "openai/gpt-4o-mini"
    assert completed.provider == "openai"
    assert completed.name == "openai/gpt-4o-mini-2024-07-18"


async def test_callback_errors_never_escape(ctx, plugin):
    assert await plugin.before_agent_callback(agent=object(), callback_context=object()) is None


def test_runner_injection_via_plugins_kwarg():
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    assert google_adk.enable() is True
    runner = Runner(
        agent=BaseAgent(name="test_agent"),
        app_name="test-app",
        session_service=InMemorySessionService(),
    )
    assert any(isinstance(p, CapturePlugin) for p in runner.plugin_manager.plugins)


def test_runner_injection_is_idempotent():
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    google_adk.enable()
    mine = CapturePlugin()
    # The plugins kwarg is supported in ADK 1.x and deprecated in 2.x; users
    # on either API must not get a duplicate plugin.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        runner = Runner(
            agent=BaseAgent(name="test_agent"),
            app_name="test-app",
            session_service=InMemorySessionService(),
            plugins=[mine],
        )
    captures = [p for p in runner.plugin_manager.plugins if isinstance(p, CapturePlugin)]
    assert captures == [mine]


def test_runner_injection_is_idempotent_via_app():
    from google.adk.agents.base_agent import BaseAgent
    App = pytest.importorskip("google.adk.apps").App
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    google_adk.enable()
    mine = CapturePlugin()
    runner = Runner(
        app=App(name="test-app", root_agent=BaseAgent(name="test_agent"), plugins=[mine]),
        session_service=InMemorySessionService(),
    )
    captures = [p for p in runner.plugin_manager.plugins if isinstance(p, CapturePlugin)]
    assert captures == [mine]


def test_runner_injection_via_app():
    from google.adk.agents.base_agent import BaseAgent
    App = pytest.importorskip("google.adk.apps").App
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    google_adk.enable()
    runner = Runner(
        app=App(name="test-app", root_agent=BaseAgent(name="test_agent")),
        session_service=InMemorySessionService(),
    )
    assert any(isinstance(p, CapturePlugin) for p in runner.plugin_manager.plugins)


def test_disable_restores_runner_init():
    from google.adk.runners import Runner

    original = Runner.__init__
    google_adk.enable()
    assert Runner.__init__ is not original
    google_adk.disable()
    assert Runner.__init__ is original


def test_unsupported_adk_version_is_not_patched(monkeypatch):
    from google.adk.runners import Runner

    original = Runner.__init__
    google_adk._patched = False
    monkeypatch.setattr(google_adk, "supported_package_version", lambda *a, **k: False)

    assert google_adk.enable() is False
    assert Runner.__init__ is original


def test_supported_adk_version_band_starts_at_public_plugin_api(monkeypatch):
    observed = {}

    def supported(distribution, *, minimum, max_major_exclusive):
        observed.update(
            distribution=distribution,
            minimum=minimum,
            max_major_exclusive=max_major_exclusive,
        )
        return True

    monkeypatch.setattr(google_adk, "supported_package_version", supported)

    assert google_adk.enable() is True
    assert observed == {
        "distribution": "google-adk",
        "minimum": (1, 7, 0),
        "max_major_exclusive": 3,
    }
