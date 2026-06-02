"""Tests for Agent provider-hosted built-in tool support.

Verifies that:
- `built_in_tools=` reaches the LM request config.
- Server-side built-in tool calls (e.g. `web_search_preview`) are recorded in
  the trace but not dispatched locally.
- Mixed built-in + user tool calls in one turn route correctly.
"""

import logging

import pytest

from agnt5 import Agent, Context, tool
from agnt5.agent import AgentRegistry
from agnt5.lm import (
    BuiltInTool,
    GenerateRequest,
    GenerateResponse,
    LanguageModel,
    MessageRole,
    TokenUsage,
)
from agnt5.lm.events import (
    LMCompleted,
    LMContentBlockCompleted,
    LMContentBlockDelta,
    LMContentBlockStarted,
)
from agnt5.tool import ToolRegistry


class MockLanguageModel(LanguageModel):
    """Mock LLM mirroring the pattern in tests/unit/test_agent.py."""

    def __init__(self, responses=None, tool_calls=None):
        self.responses = responses or ["Mock response"]
        self.tool_calls_list = tool_calls or []
        self.call_count = 0
        self.requests = []

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.requests.append(request)
        response_text = self.responses[min(self.call_count, len(self.responses) - 1)]
        tool_calls = None
        if self.call_count < len(self.tool_calls_list):
            tool_calls = self.tool_calls_list[self.call_count]
        self.call_count += 1
        return GenerateResponse(
            text=response_text,
            tool_calls=tool_calls,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

    async def stream(self, request: GenerateRequest):
        # Tests below force the generate() path by setting built_in_tools or tools.
        response_text = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        yield LMContentBlockStarted(name="mock-model", correlation_id="c", parent_correlation_id="p", block_type="text", index=0)
        yield LMContentBlockDelta(name="mock-model", correlation_id="c", parent_correlation_id="p", content=response_text, block_type="text", index=0)
        yield LMContentBlockCompleted(name="mock-model", correlation_id="c", parent_correlation_id="p", block_type="text", index=0)
        yield LMCompleted(name="mock-model", correlation_id="c", parent_correlation_id="p", model="mock-model", provider="mock", input_tokens=10, output_tokens=20, total_tokens=30, output_data={"text": response_text})


@pytest.fixture(autouse=True)
def _clear_registries():
    ToolRegistry.clear()
    AgentRegistry.clear()
    yield
    ToolRegistry.clear()
    AgentRegistry.clear()


@pytest.mark.asyncio
async def test_built_in_tools_forwarded_to_request_config():
    """Agent passes built_in_tools through to GenerationConfig on each LM call."""
    mock_lm = MockLanguageModel(
        responses=["Search complete: the sky is blue."],
        tool_calls=[
            [{"id": "ws_1", "name": "web_search_preview", "arguments": '{"query": "sky color"}'}],
        ],
    )

    agent = Agent(
        name="builtin_agent",
        model=mock_lm,
        instructions="Use web search when helpful.",
        built_in_tools=[BuiltInTool.WEB_SEARCH],
    )

    result = await agent.run("Why is the sky blue?")

    assert mock_lm.requests, "agent should have made at least one LM request"
    sent = mock_lm.requests[0].config.built_in_tools
    assert sent == [BuiltInTool.WEB_SEARCH]
    assert "sky is blue" in result.output


@pytest.mark.asyncio
async def test_built_in_tool_configuration_and_usage_are_logged(caplog):
    """Agent logs configured built-ins and provider-side usage."""
    caplog.set_level(logging.INFO, logger="agnt5.agent.logged_builtin")
    mock_lm = MockLanguageModel(
        responses=["Final answer based on search results."],
        tool_calls=[
            [{"id": "ws_1", "name": "web_search_preview", "arguments": "{}"}],
        ],
    )

    agent = Agent(
        name="logged_builtin",
        model=mock_lm,
        instructions="Use web search.",
        built_in_tools=[BuiltInTool.WEB_SEARCH],
    )

    await agent.run("Look something up.")

    assert "Built-in tools enabled: ['web_search_preview']" in caplog.text
    assert "Built-in tools used by model: ['web_search_preview']" in caplog.text


@pytest.mark.asyncio
async def test_built_in_tool_without_surfaced_call_is_logged(caplog):
    """Agent logs when configured built-ins have no surfaced provider call."""
    caplog.set_level(logging.INFO, logger="agnt5.agent.logged_builtin_unused")
    mock_lm = MockLanguageModel(
        responses=["Final answer without search."],
        tool_calls=[None],
    )

    agent = Agent(
        name="logged_builtin_unused",
        model=mock_lm,
        instructions="Use web search when helpful.",
        built_in_tools=[BuiltInTool.WEB_SEARCH],
    )

    await agent.run("Answer directly.")

    assert "Built-in tools enabled: ['web_search_preview']" in caplog.text
    assert "no provider-side tool calls were surfaced this iteration" in caplog.text


@pytest.mark.asyncio
async def test_server_side_built_in_call_is_not_dispatched_locally():
    """A web_search_preview call returned by the LM is recorded but not executed."""
    mock_lm = MockLanguageModel(
        responses=["Final answer based on search results."],
        tool_calls=[
            [{"id": "ws_1", "name": "web_search_preview", "arguments": '{"query": "topic"}'}],
        ],
    )

    agent = Agent(
        name="builtin_only",
        model=mock_lm,
        instructions="Use the built-in search.",
        built_in_tools=[BuiltInTool.WEB_SEARCH],
    )

    result = await agent.run("Look something up.")

    # Single LM turn — agent does NOT loop again because no user tool needs dispatch.
    assert mock_lm.call_count == 1
    # Built-in call recorded with the marker.
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "web_search_preview"
    assert result.tool_calls[0].get("built_in") is True
    # No "Tool not found" sneaking into the output.
    assert "not found" not in result.output.lower()
    assert "Final answer" in result.output


@pytest.mark.asyncio
async def test_mixed_built_in_and_user_tool_calls_in_one_turn():
    """When a turn returns both a built-in call and a user tool call, only the
    user tool is dispatched locally; both are recorded; the loop continues."""

    @tool
    async def save_note(ctx: Context, text: str) -> str:
        """Save a note."""
        return f"saved:{text}"

    mock_lm = MockLanguageModel(
        responses=[
            "I'll search and save.",  # turn 1 — issues both calls
            "Done.",                    # turn 2 — final answer after user tool result
        ],
        tool_calls=[
            [
                {"id": "ws_1", "name": "web_search_preview", "arguments": '{"query": "x"}'},
                {"id": "save_1", "name": "save_note", "arguments": '{"text": "hello"}'},
            ],
            None,
        ],
    )

    agent = Agent(
        name="mixed_agent",
        model=mock_lm,
        instructions="Search and save.",
        tools=[save_note],
        built_in_tools=[BuiltInTool.WEB_SEARCH],
    )

    result = await agent.run("Search and save a note.")

    # Loop ran twice: turn 1 (mixed calls), turn 2 (final answer).
    assert mock_lm.call_count == 2
    # Both calls recorded; the built-in carries the marker.
    names = [tc["name"] for tc in result.tool_calls]
    assert "web_search_preview" in names
    assert "save_note" in names
    built_in_entries = [tc for tc in result.tool_calls if tc.get("built_in")]
    assert len(built_in_entries) == 1
    assert built_in_entries[0]["name"] == "web_search_preview"

    second_request_assistant = [
        msg for msg in mock_lm.requests[1].messages
        if msg.role == MessageRole.ASSISTANT and msg.content == "I'll search and save."
    ][0]
    assert second_request_assistant.tool_calls == [
        {"id": "save_1", "name": "save_note", "arguments": '{"text": "hello"}'},
    ]
    assert "Done" in result.output


@pytest.mark.asyncio
async def test_anthropic_style_tool_use_name_is_recognized_as_built_in():
    """Anthropic returns tool_use blocks with name="web_search" (not the
    OpenAI alias). The Agent must still treat these as server-side and skip
    local dispatch."""
    mock_lm = MockLanguageModel(
        responses=["Final answer with grounded info."],
        tool_calls=[
            [{"id": "ws_1", "name": "web_search", "arguments": '{"query": "topic"}'}],
        ],
    )

    agent = Agent(
        name="anthropic_builtin",
        model=mock_lm,
        instructions="Use the built-in search.",
        built_in_tools=[BuiltInTool.WEB_SEARCH],
    )

    result = await agent.run("Look something up.")

    assert mock_lm.call_count == 1
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "web_search"
    assert result.tool_calls[0].get("built_in") is True
    assert "Final answer" in result.output


@pytest.mark.asyncio
async def test_web_fetch_built_in_recognized_as_server_side():
    """Anthropic's web_fetch built-in surfaces as a tool_use block named
    `web_fetch`. Agent must filter it the same way as web_search."""
    mock_lm = MockLanguageModel(
        responses=["Final answer using fetched content."],
        tool_calls=[
            [{"id": "wf_1", "name": "web_fetch", "arguments": '{"url": "https://example.com"}'}],
        ],
    )

    agent = Agent(
        name="fetch_builtin",
        model=mock_lm,
        instructions="Fetch URLs when asked.",
        built_in_tools=[BuiltInTool.WEB_FETCH],
    )

    result = await agent.run("Read https://example.com")

    assert mock_lm.call_count == 1
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "web_fetch"
    assert result.tool_calls[0].get("built_in") is True
    sent = mock_lm.requests[0].config.built_in_tools
    assert sent == [BuiltInTool.WEB_FETCH]


@pytest.mark.asyncio
async def test_no_built_in_tools_means_no_config_change():
    """Agents without built_in_tools leave GenerationConfig.built_in_tools empty."""

    @tool
    async def echo(ctx: Context, msg: str) -> str:
        return msg

    mock_lm = MockLanguageModel(
        responses=["ok"],
        tool_calls=[None],
    )

    agent = Agent(
        name="no_builtin",
        model=mock_lm,
        instructions="Reply ok.",
        tools=[echo],
    )

    await agent.run("hi")

    assert mock_lm.requests
    assert mock_lm.requests[0].config.built_in_tools == []
