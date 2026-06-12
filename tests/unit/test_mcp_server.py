"""Tests for the developer-facing MCPServer API."""

import io

import httpx
import pytest

from agnt5 import Agent, Context, MCPServer, Prompt, Resource, tool, workflow
from agnt5.agent import AgentRegistry
from agnt5.lm import GenerateRequest, GenerateResponse, LanguageModel, TokenUsage
from agnt5.lm.events import (
    LMCompleted,
    LMContentBlockCompleted,
    LMContentBlockDelta,
    LMContentBlockStarted,
)
from agnt5.tool import ToolRegistry


class MockLanguageModel(LanguageModel):
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        return GenerateResponse(
            text="Agent answer",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def stream(self, request: GenerateRequest):
        # The Agent loop falls back to streaming when no tools are configured.
        # Emit the same text the generate() path returns so both paths agree.
        text = "Agent answer"
        yield LMContentBlockStarted(
            name="mock-model", correlation_id="c", parent_correlation_id="p",
            block_type="text", index=0,
        )
        yield LMContentBlockDelta(
            name="mock-model", correlation_id="c", parent_correlation_id="p",
            content=text, block_type="text", index=0,
        )
        yield LMContentBlockCompleted(
            name="mock-model", correlation_id="c", parent_correlation_id="p",
            block_type="text", index=0,
        )
        yield LMCompleted(
            name="mock-model", correlation_id="c", parent_correlation_id="p",
            model="mock-model", provider="mock",
            input_tokens=1, output_tokens=1, total_tokens=2,
            output_data={"text": text},
        )


@pytest.fixture(autouse=True)
def clear_registries():
    ToolRegistry.clear()
    AgentRegistry.clear()
    yield
    ToolRegistry.clear()
    AgentRegistry.clear()


@tool
async def echo(ctx: Context, message: str) -> str:
    """Echo a message."""
    return f"echo:{message}"


@workflow()
async def summarize_topic(ctx, input: dict) -> dict:
    return {"topic": input["topic"], "summary": "done"}


@pytest.mark.asyncio
async def test_mcp_server_lists_and_calls_registered_primitives():
    agent = Agent(
        name="research_agent",
        model=MockLanguageModel(),
        instructions="Be helpful",
    )

    server = MCPServer(
        id="test-mcp",
        name="Test MCP",
        version="1.0.0",
        tools={"echo": echo},
        agents={"research_agent": agent},
        workflows={"summarize_topic": summarize_topic},
    )

    tools_response = await server.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    )
    tools = tools_response["result"]["tools"]
    tool_names = {tool_def["name"] for tool_def in tools}

    assert "echo" in tool_names
    assert "research_agent" in tool_names
    assert "summarize_topic" in tool_names

    echo_response = await server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"message": "hello"}},
        }
    )
    assert echo_response["result"]["content"][0]["text"] == "echo:hello"

    # Workflow arguments are spread as kwargs into the handler keyed by
    # parameter name — `summarize_topic(ctx, input: dict)` expects an `input`
    # kwarg, matching how `_invoke_tool` spreads arguments into tool handlers.
    workflow_response = await server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "summarize_topic",
                "arguments": {"input": {"topic": "mcp"}},
            },
        }
    )
    workflow_text = workflow_response["result"]["content"][0]["text"]
    assert '"summary"' in workflow_text and '"done"' in workflow_text
    assert '"topic"' in workflow_text and '"mcp"' in workflow_text

    agent_response = await server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "research_agent", "arguments": {"input": "hi"}},
        }
    )
    agent_text = agent_response["result"]["content"][0]["text"]
    assert '"output"' in agent_text and '"Agent answer"' in agent_text


@pytest.mark.asyncio
async def test_mcp_server_prompts_and_resources():
    async def prompt_handler(topic: str) -> dict:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": {"type": "text", "text": f"Research {topic}"},
                }
            ]
        }

    async def read_handbook() -> str:
        return "# Handbook"

    server = MCPServer(
        id="test-mcp",
        name="Test MCP",
        version="1.0.0",
        prompts={
            "research_brief": Prompt(
                name="research_brief",
                description="Build a research brief",
                arguments_schema={
                    "type": "object",
                    "properties": {"topic": {"type": "string"}},
                    "required": ["topic"],
                },
                handler=prompt_handler,
            )
        },
        resources={
            "docs://handbook": Resource.text(
                uri="docs://handbook",
                name="Handbook",
                mime_type="text/markdown",
                read=read_handbook,
            )
        },
    )

    prompts_response = await server.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "prompts/list", "params": {}}
    )
    assert prompts_response["result"]["prompts"][0]["name"] == "research_brief"

    prompt_response = await server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "prompts/get",
            "params": {"name": "research_brief", "arguments": {"topic": "AGNT5"}},
        }
    )
    assert prompt_response["result"]["messages"][0]["content"]["text"] == "Research AGNT5"

    resources_response = await server.dispatch(
        {"jsonrpc": "2.0", "id": 3, "method": "resources/list", "params": {}}
    )
    assert resources_response["result"]["resources"][0]["uri"] == "docs://handbook"

    resource_response = await server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "resources/read",
            "params": {"uri": "docs://handbook"},
        }
    )
    assert resource_response["result"]["contents"][0]["text"] == "# Handbook"


def test_mcp_server_stdio_helpers_use_jsonl_framing():
    payload = b'{"jsonrpc":"2.0","id":1,"result":{"ok":true}}'
    input_stream = io.BytesIO(payload + b"\r\n")
    output_stream = io.BytesIO()

    assert MCPServer._read_message(input_stream) == payload

    MCPServer._write_message(output_stream, payload)
    output = output_stream.getvalue()
    assert output == payload + b"\n"
    assert b"Content-Length" not in output


@pytest.mark.asyncio
async def test_mcp_server_run_http_serves_json_rpc():
    server = MCPServer(id="test-mcp", name="Test MCP", version="1.0.0")
    handle = await server._start_http_server(host="127.0.0.1", port=0, path="/mcp")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://{handle.host}:{handle.port}/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                headers={"Accept": "application/json, text/event-stream"},
            )

        assert response.status_code == 200
        assert response.json()["result"]["serverInfo"]["name"] == "Test MCP"
        assert response.json()["result"]["protocolVersion"] == "2025-11-25"
    finally:
        await handle.close()


@pytest.mark.asyncio
async def test_mcp_server_run_http_rejects_cross_origin_requests():
    server = MCPServer(id="test-mcp", name="Test MCP", version="1.0.0")
    handle = await server._start_http_server(host="127.0.0.1", port=0, path="/mcp")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://{handle.host}:{handle.port}/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}},
                headers={"Origin": "https://evil.example"},
            )

        assert response.status_code == 403
    finally:
        await handle.close()
