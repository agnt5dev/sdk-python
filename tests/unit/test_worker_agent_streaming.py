"""Hosted Agent streaming executor tests."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from agnt5._serialization import serialize
from agnt5.agent.events import AgentCompleted, ToolCallCompleted, ToolCallStarted
from agnt5.lm.events import (
    LMContentBlockCompleted,
    LMContentBlockDelta,
    LMContentBlockStarted,
    LMFailed,
)
from agnt5.worker._executors import ExecutorMixin


class _RecordingWorker:
    def __init__(self) -> None:
        self.event_types: list[str] = []

    def emit_event_sync(self, *, event_type: str, **kwargs) -> None:
        self.event_types.append(event_type)

    async def emit_event_async(self, *, event_type: str, **kwargs) -> None:
        self.event_types.append(event_type)

    async def emit_event_batch_async(self, events) -> None:
        # (run_id, event_type, data, sequence, metadata, timestamp_ns)
        self.event_types.extend(event[1] for event in events)

    def queue_event(self, *, event_type: str, **kwargs) -> None:
        self.event_types.append(event_type)


class _DummyExecutor(ExecutorMixin):
    def __init__(self, worker: _RecordingWorker) -> None:
        self._entity_state_adapter = object()
        self._checkpoint_client = None
        self._rust_worker = worker
        self.service_name = "test"


class _BlockingFlushWorker(_RecordingWorker):
    def __init__(self, *, error=False):
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.loop_thread = threading.get_ident()
        self.entered = asyncio.Event()
        self.release = threading.Event()
        self.finished = False
        self.error = error

    def flush_workflow_checkpoints(self):
        assert threading.get_ident() != self.loop_thread, "native flush blocked the event loop"
        self.loop.call_soon_threadsafe(self.entered.set)
        assert self.release.wait(timeout=2), "test did not release the flush"
        self.finished = True
        if self.error:
            raise RuntimeError("flush failed")


class _StreamingAgent:
    name = "streaming_agent"

    def __init__(self) -> None:
        self.history = None

    async def stream(self, message, context, history=None):
        assert message == "hello"
        self.history = history
        yield LMContentBlockStarted(
            name="mock-model",
            correlation_id="lm-1",
            parent_correlation_id="iteration-1",
            block_type="text",
            index=0,
        )
        yield LMContentBlockDelta(
            name="mock-model",
            correlation_id="lm-1",
            parent_correlation_id="iteration-1",
            content="hello back",
            block_type="text",
            index=0,
        )
        yield LMContentBlockCompleted(
            name="mock-model",
            correlation_id="lm-1",
            parent_correlation_id="iteration-1",
            block_type="text",
            index=0,
        )
        yield LMContentBlockStarted(
            name="mock-model",
            correlation_id="thinking-1",
            parent_correlation_id="iteration-1",
            block_type="thinking",
            index=1,
        )
        yield LMContentBlockDelta(
            name="mock-model",
            correlation_id="thinking-1",
            parent_correlation_id="iteration-1",
            content="considering",
            block_type="thinking",
            index=1,
        )
        yield LMContentBlockCompleted(
            name="mock-model",
            correlation_id="thinking-1",
            parent_correlation_id="iteration-1",
            block_type="thinking",
            index=1,
        )
        tool_started = ToolCallStarted(
            name="search",
            correlation_id="tool-1",
            parent_correlation_id="iteration-1",
            tool_name="search",
            tool_call_id="call-1",
        )
        context.emit(tool_started)
        yield tool_started

        tool_completed = ToolCallCompleted(
            name="search",
            correlation_id="tool-1",
            parent_correlation_id="iteration-1",
            tool_name="search",
            tool_call_id="call-1",
            output_data={"result": "found"},
        )
        context.emit(tool_completed)
        yield tool_completed

        yield AgentCompleted(
            name=self.name,
            correlation_id="agent-1",
            parent_correlation_id="run-1",
            output_data={"output": "hello back", "tool_calls": []},
        )

    async def run(self, message, context):
        raise AssertionError("hosted Agent execution must use Agent.stream()")


class _FailingStreamingAgent:
    name = "failing_streaming_agent"

    async def stream(self, message, context):
        assert message == "hello"
        failure = LMFailed(
            name="mock-model",
            correlation_id="lm-1",
            parent_correlation_id="iteration-1",
            model="mock-model",
            provider="mock",
            error_code="RuntimeError",
            error_message="provider failed",
        )
        context.emit(failure)
        yield failure
        raise RuntimeError("provider failed")


def _request(*, messages=None):
    input_data = {"message": "hello"}
    if messages is not None:
        input_data["messages"] = messages
    return SimpleNamespace(
        invocation_id="run-streaming-agent",
        input_data=serialize(input_data),
        runtime_context=None,
        metadata={},
        session_id="",
        user_id="",
        attempt=0,
        is_streaming=True,
        component_name="streaming_agent",
    )


@pytest.mark.asyncio
async def test_execute_agent_forwards_stream_events_and_terminal_lifecycle():
    worker = _RecordingWorker()
    executor = _DummyExecutor(worker)
    agent = _StreamingAgent()
    history = [{"role": "user", "content": "prior turn"}]

    response = await executor._execute_agent(
        agent,
        b"",
        _request(messages=history),
    )

    assert response is None
    assert agent.history == history
    component_event_types = [
        event_type for event_type in worker.event_types if not event_type.startswith("log")
    ]
    assert component_event_types == [
        "run.started",
        "agent.started",
        "lm.message.start",
        "lm.message.delta",
        "lm.message.stop",
        "lm.thinking.start",
        "lm.thinking.delta",
        "lm.thinking.stop",
        "tool_call.started",
        "tool_call.completed",
        "agent.completed",
        "run.completed",
        "session.created",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_hosted_stream_flush_yields_and_settles_before_completion_or_cancellation(cancel):
    worker = _BlockingFlushWorker()
    task = asyncio.create_task(_DummyExecutor(worker)._execute_agent(_StreamingAgent(), b"", _request()))
    entered = asyncio.create_task(worker.entered.wait())
    try:
        done, _ = await asyncio.wait({task, entered}, timeout=2,
                                     return_when=asyncio.FIRST_COMPLETED)
        assert entered in done, "executor did not reach an off-thread flush"
        await asyncio.sleep(0)
        assert not task.done()
        assert "run.completed" not in worker.event_types
        if cancel:
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done(), "repeated cancellation overtook the pending flush"
        worker.release.set()
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await task
            assert "run.completed" not in worker.event_types
        else:
            await task
            assert "run.completed" in worker.event_types
        assert worker.finished
    finally:
        worker.release.set()
        entered.cancel()
        await asyncio.gather(entered, task, return_exceptions=True)


@pytest.mark.asyncio
async def test_hosted_stream_flush_failure_fails_run():
    worker = _BlockingFlushWorker(error=True)
    worker.release.set()

    await _DummyExecutor(worker)._execute_agent(_StreamingAgent(), b"", _request())

    assert worker.finished
    assert "run.failed" in worker.event_types
    assert "run.completed" not in worker.event_types


@pytest.mark.asyncio
async def test_cancelled_hosted_stream_observes_flush_failure_before_unwinding():
    worker = _BlockingFlushWorker(error=True)
    task = asyncio.create_task(_DummyExecutor(worker)._execute_agent(_StreamingAgent(), b"", _request()))
    try:
        await asyncio.wait_for(worker.entered.wait(), timeout=2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        worker.release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert worker.finished
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert str(caught.value.__cause__) == "flush failed"
        assert "run.completed" not in worker.event_types
    finally:
        worker.release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_execute_agent_does_not_duplicate_persisted_lm_failure():
    worker = _RecordingWorker()
    executor = _DummyExecutor(worker)

    response = await executor._execute_agent(
        _FailingStreamingAgent(),
        b"",
        _request(),
    )

    assert response is None
    component_event_types = [
        event_type for event_type in worker.event_types if not event_type.startswith("log")
    ]
    assert component_event_types == [
        "run.started",
        "agent.started",
        "lm.failed",
        "agent.failed",
        "run.failed",
    ]


class _EchoAgent:
    name = "echo_agent"

    def __init__(self) -> None:
        self.messages: list = []
        self.session_ids: list = []

    async def stream(self, message, context):
        self.messages.append(message)
        self.session_ids.append(context.session_id)
        yield AgentCompleted(
            name=self.name,
            correlation_id="agent-1",
            parent_correlation_id="run-1",
            output_data={"output": f"echo: {message}", "tool_calls": []},
        )


def _request_with(input_data):
    return SimpleNamespace(
        invocation_id="run-echo-agent",
        input_data=serialize(input_data),
        runtime_context=None,
        metadata={},
        session_id="",
        user_id="",
        attempt=0,
        is_streaming=True,
        component_name="echo_agent",
    )


def _component_events(worker: _RecordingWorker) -> list[str]:
    return [event_type for event_type in worker.event_types if not event_type.startswith("log")]


@pytest.mark.asyncio
async def test_execute_agent_accepts_mcp_tool_input():
    # A hosted MCP server forwards an agent tool's arguments unchanged as the
    # run input: {"input": ..., "session_id"?: ...} (AGENT_INPUT_SCHEMA).
    worker = _RecordingWorker()
    agent = _EchoAgent()

    response = await _DummyExecutor(worker)._execute_agent(
        agent, b"", _request_with({"input": "hello", "session_id": "mcp-session"})
    )

    assert response is None
    assert agent.messages == ["hello"]
    assert agent.session_ids == ["mcp-session"]
    events = _component_events(worker)
    assert "run.failed" not in events
    assert events[-2:] == ["run.completed", "session.created"]


@pytest.mark.asyncio
async def test_execute_agent_prefers_message_over_input():
    worker = _RecordingWorker()
    agent = _EchoAgent()

    await _DummyExecutor(worker)._execute_agent(
        agent, b"", _request_with({"message": "hello", "input": "ignored"})
    )

    assert agent.messages == ["hello"]
    assert "run.completed" in _component_events(worker)


@pytest.mark.asyncio
async def test_execute_agent_without_message_or_input_fails():
    worker = _RecordingWorker()
    agent = _EchoAgent()

    await _DummyExecutor(worker)._execute_agent(agent, b"", _request_with({"input": {"k": 1}}))

    assert agent.messages == []
    assert "run.failed" in _component_events(worker)
