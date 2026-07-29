"""Agent tool streaming contract tests."""

from unittest.mock import AsyncMock, MagicMock, patch

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


class _MockChunk:
    def __init__(self, *, text: str = "", tool_calls=None) -> None:
        self.chunk_type = "completed"
        self.text = text
        self.block_type = None
        self.index = None
        self.model = "gpt-4o-mini"
        self.finish_reason = "tool_calls" if tool_calls else "stop"
        self.usage = None
        self.tool_calls = tool_calls


class _MockAsyncStreamIterator:
    def __init__(self, chunks) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


@pytest.mark.asyncio
async def test_string_model_streams_tools_across_iterations():
    @tool
    async def double(ctx: Context, value: int) -> int:
        """Double an integer."""
        return value * 2

    streamed_responses = [
        _MockAsyncStreamIterator(
            [
                _MockChunk(
                    tool_calls=[
                        {
                            "id": "call-double",
                            "name": "double",
                            "arguments": '{"value":2}',
                        }
                    ]
                )
            ]
        ),
        _MockAsyncStreamIterator([_MockChunk(text="The result is 4.")]),
    ]

    with patch("agnt5.lm.client.RustLanguageModel") as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.generate = AsyncMock(
            side_effect=AssertionError("tool-capable string models must stream")
        )
        mock_instance.stream_iter = MagicMock(side_effect=streamed_responses)
        mock_rust_class.return_value = mock_instance

        agent = Agent(
            name="string-streaming-tools-agent",
            model="openai/gpt-4o-mini",
            instructions="Use the tool.",
            tools=[double],
            max_iterations=3,
        )
        events: list[Event] = [event async for event in agent.stream("Double 2")]

    completed = next(event for event in events if event.event_type == "agent.completed")
    assert mock_instance.generate.await_count == 0
    assert mock_instance.stream_iter.call_count == 2
    assert completed.output_data["output"] == "The result is 4."
    assert completed.output_data["tool_calls"][0]["name"] == "double"
