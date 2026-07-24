"""Nested Workflow streaming tests."""

from unittest.mock import MagicMock

import pytest

from agnt5.agent.events import AgentCompleted, AgentStarted, ToolCallStarted
from agnt5.lm.events import LMContentBlockDelta
from agnt5.workflow import WorkflowContext


@pytest.mark.asyncio
async def test_nested_agent_stream_forwards_content_without_duplicate_lifecycle():
    context = object.__new__(WorkflowContext)
    context.emit = MagicMock()

    async def agent_events():
        yield AgentStarted(
            name="researcher",
            correlation_id="agent-1",
            parent_correlation_id="step-1",
        )
        yield LMContentBlockDelta(
            name="mock-model",
            correlation_id="lm-1",
            parent_correlation_id="iteration-1",
            content="streamed answer",
            block_type="text",
            index=0,
        )
        yield ToolCallStarted(
            name="search",
            correlation_id="tool-1",
            parent_correlation_id="iteration-1",
            tool_name="search",
            tool_call_id="call-1",
        )
        yield AgentCompleted(
            name="researcher",
            correlation_id="agent-1",
            parent_correlation_id="step-1",
            output_data={"output": "streamed answer", "tool_calls": []},
        )

    result = await context._consume_streaming_result(agent_events(), "research")

    assert result == "streamed answer"
    forwarded = [call.args[0] for call in context.emit.call_args_list]
    assert [event.event_type for event in forwarded] == ["lm.content_block.delta"]
