"""Tests for agent execution callbacks."""

from typing import AsyncGenerator, Optional

import pytest
from agnt5 import Agent, Context, override, tool
from agnt5.agent import AgentRegistry
from agnt5.events import Event
from agnt5.lm import GenerateRequest, GenerateResponse, LanguageModel, TokenUsage
from agnt5.lm.events import (
    LMCompleted,
    LMContentBlockCompleted,
    LMContentBlockDelta,
    LMContentBlockStarted,
)
from agnt5.tool import ToolRegistry


class MockLanguageModel(LanguageModel):
    def __init__(
        self,
        responses: Optional[list[GenerateResponse]] = None,
    ) -> None:
        self.responses = responses or [GenerateResponse(text="mock response")]
        self.call_count = 0
        self.requests: list[GenerateRequest] = []

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.requests.append(
            GenerateRequest(
                model=request.model,
                messages=list(request.messages),
                system_prompt=request.system_prompt,
                tools=list(request.tools),
                tool_choice=request.tool_choice,
                config=request.config,
                response_schema=request.response_schema,
            )
        )
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[Event, None]:
        self.requests.append(request)
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        yield LMContentBlockStarted(
            name="mock-model",
            correlation_id="lm",
            parent_correlation_id="agent",
            block_type="text",
            index=0,
        )
        yield LMContentBlockDelta(
            name="mock-model",
            correlation_id="lm",
            parent_correlation_id="agent",
            content=response.text,
            block_type="text",
            index=0,
        )
        yield LMContentBlockCompleted(
            name="mock-model",
            correlation_id="lm",
            parent_correlation_id="agent",
            block_type="text",
            index=0,
        )
        yield LMCompleted(
            name="mock-model",
            correlation_id="lm",
            parent_correlation_id="agent",
            model="mock-model",
            provider="mock",
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            output_data={"text": response.text},
        )


@pytest.fixture(autouse=True)
def clear_registries():
    ToolRegistry.clear()
    AgentRegistry.clear()
    yield
    ToolRegistry.clear()
    AgentRegistry.clear()


@pytest.mark.asyncio
async def test_before_agent_short_circuits_model_call() -> None:
    model = MockLanguageModel([GenerateResponse(text="provider response")])

    agent = Agent(
        name="callback-agent",
        model=model,
        instructions="Be helpful",
        before_agent_callback=lambda ctx: "blocked by callback",
    )

    result = await agent.run("hello")

    assert result.output == "blocked by callback"
    assert model.call_count == 0


@pytest.mark.asyncio
async def test_after_agent_replaces_output() -> None:
    model = MockLanguageModel([GenerateResponse(text="provider response")])

    agent = Agent(
        name="callback-agent",
        model=model,
        instructions="Be helpful",
        after_agent_callback=lambda ctx, result: "rewritten output",
    )

    result = await agent.run("hello")

    assert result.output == "rewritten output"
    assert model.call_count == 1


@pytest.mark.asyncio
async def test_before_model_short_circuits_provider_call() -> None:
    model = MockLanguageModel([GenerateResponse(text="provider response")])

    agent = Agent(
        name="callback-agent",
        model=model,
        instructions="Be helpful",
        before_model_callback=lambda ctx, request: GenerateResponse(
            text="cached response",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        ),
    )

    result = await agent.run("hello")

    assert result.output == "cached response"
    assert model.call_count == 0


@pytest.mark.asyncio
async def test_after_model_replaces_response() -> None:
    model = MockLanguageModel([GenerateResponse(text="provider response")])

    agent = Agent(
        name="callback-agent",
        model=model,
        instructions="Be helpful",
        after_model_callback=lambda ctx, request, response: GenerateResponse(
            text="rewritten model"
        ),
    )

    result = await agent.run("hello")

    assert result.output == "rewritten model"
    assert model.call_count == 1


@pytest.mark.asyncio
async def test_before_tool_short_circuits_tool_handler() -> None:
    tool_called = False
    model = MockLanguageModel(
        [
            GenerateResponse(
                text="using tool",
                tool_calls=[{"name": "lookup", "arguments": '{"query": "x"}'}],
            ),
            GenerateResponse(text="done"),
        ]
    )

    @tool
    async def lookup(ctx: Context, query: str) -> str:
        nonlocal tool_called
        tool_called = True
        return f"real {query}"

    agent = Agent(
        name="callback-agent",
        model=model,
        instructions="Use tools",
        tools=[lookup],
        before_tool_callback=lambda ctx: "cached tool result",
    )

    result = await agent.run("lookup x")

    assert result.output == "done"
    assert tool_called is False
    assert "cached tool result" in model.requests[1].messages[-1].content


@pytest.mark.asyncio
async def test_after_tool_rewrites_tool_result() -> None:
    model = MockLanguageModel(
        [
            GenerateResponse(
                text="using tool",
                tool_calls=[{"name": "lookup", "arguments": '{"query": "x"}'}],
            ),
            GenerateResponse(text="done"),
        ]
    )

    @tool
    async def lookup(ctx: Context, query: str) -> str:
        return f"real {query}"

    agent = Agent(
        name="callback-agent",
        model=model,
        instructions="Use tools",
        tools=[lookup],
        after_tool_callback=lambda ctx, result: override("rewritten tool result"),
    )

    result = await agent.run("lookup x")

    assert result.output == "done"
    assert "rewritten tool result" in model.requests[1].messages[-1].content
