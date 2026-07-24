"""Hosted Agent streaming executor tests."""

from types import SimpleNamespace

import pytest

from agnt5._serialization import serialize
from agnt5.agent.events import AgentCompleted, ToolCallCompleted, ToolCallStarted
from agnt5.lm.events import (
    LMContentBlockCompleted,
    LMContentBlockDelta,
    LMContentBlockStarted,
)
from agnt5.worker._executors import ExecutorMixin


class _RecordingWorker:
    def __init__(self) -> None:
        self.event_types: list[str] = []

    def emit_event_sync(self, *, event_type: str, **kwargs) -> None:
        self.event_types.append(event_type)

    def queue_event(self, *, event_type: str, **kwargs) -> None:
        self.event_types.append(event_type)


class _DummyExecutor(ExecutorMixin):
    def __init__(self, worker: _RecordingWorker) -> None:
        self._entity_state_adapter = object()
        self._checkpoint_client = None
        self._rust_worker = worker
        self.service_name = "test"


class _StreamingAgent:
    name = "streaming_agent"

    async def stream(self, message, context):
        assert message == "hello"
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


def _request():
    return SimpleNamespace(
        invocation_id="run-streaming-agent",
        input_data=serialize({"message": "hello"}),
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

    response = await executor._execute_agent(
        _StreamingAgent(),
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
        "lm.content_block.started",
        "lm.content_block.delta",
        "lm.content_block.completed",
        "tool_call.started",
        "tool_call.completed",
        "agent.completed",
        "run.completed",
        "session.created",
    ]
