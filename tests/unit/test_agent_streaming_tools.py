"""Agent tool streaming contract tests."""

import pytest

from agnt5 import Agent, Context, tool
from agnt5.events import Event
from agnt5.lm import GenerateRequest, GenerateResponse, LanguageModel
from agnt5.lm.events import LMCompleted


class _StreamingToolModel(LanguageModel):
    supports_streaming_tools = True

    def __init__(self) -> None:
        self.generate_calls = 0
        self.stream_calls = 0

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.generate_calls += 1
        raise AssertionError("streaming-capable tool models must use stream()")

    async def stream(self, request: GenerateRequest):
        self.stream_calls += 1
        has_tool_result = any(message.tool_call_id for message in request.messages)
        if has_tool_result:
            output = {"text": "The result is 4.", "tool_calls": []}
        else:
            output = {
                "text": "",
                "tool_calls": [
                    {
                        "id": "call-double",
                        "name": "double",
                        "arguments": '{"value":2}',
                    }
                ],
            }
        yield LMCompleted(
            name="streaming-tools",
            correlation_id=f"lm-{self.stream_calls}",
            parent_correlation_id="iteration",
            model="streaming-tools",
            provider="test",
            output_data=output,
        )


@pytest.mark.asyncio
async def test_agent_uses_opted_in_tool_stream_across_iterations():
    @tool
    async def double(ctx: Context, value: int) -> int:
        """Double an integer."""
        return value * 2

    model = _StreamingToolModel()
    agent = Agent(
        name="streaming-tools-agent",
        model=model,
        instructions="Use the tool.",
        tools=[double],
        max_iterations=3,
    )

    events: list[Event] = [event async for event in agent.stream("Double 2")]
    completed = next(event for event in events if event.event_type == "agent.completed")

    assert model.generate_calls == 0
    assert model.stream_calls == 2
    assert completed.output_data["output"] == "The result is 4."
    assert completed.output_data["tool_calls"][0]["name"] == "double"
