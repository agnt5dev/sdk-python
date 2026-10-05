"""Tests for the developer-facing MCPServer API."""

import io
import json
import sys

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


def _serve_stdio(server: MCPServer, monkeypatch, *messages) -> list[dict]:
    """Run the stdio loop over the given lines and return the replies."""
    lines = []
    for message in messages:
        lines.append(message if isinstance(message, str) else json.dumps(message))
    stdin = io.TextIOWrapper(io.BytesIO(("\n".join(lines) + "\n").encode("utf-8")))
    stdout = io.TextIOWrapper(io.BytesIO())
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)
    server._serve_stdio_sync()
    output = stdout.buffer.getvalue().decode("utf-8")
    return [json.loads(line) for line in output.splitlines() if line]


def test_stdio_does_not_reply_to_notifications(monkeypatch):
    server = MCPServer(id="test-mcp", tools={"echo": echo})
    replies = _serve_stdio(
        server,
        monkeypatch,
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1}},
        {"jsonrpc": "2.0", "method": "no/such/notification"},
        {"jsonrpc": "2.0", "id": 2, "method": "ping"},
    )
    assert [reply["id"] for reply in replies] == [1, 2]
    assert replies[0]["result"]["serverInfo"]["name"] == "test-mcp"
    assert replies[1]["result"] == {}


def test_stdio_ignores_responses_from_the_client(monkeypatch):
    server = MCPServer(id="test-mcp")
    replies = _serve_stdio(
        server,
        monkeypatch,
        {"jsonrpc": "2.0", "id": "srv-1", "result": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "ping"},
    )
    assert [reply["id"] for reply in replies] == [3]


@pytest.mark.asyncio
async def test_unknown_method_is_method_not_found():
    server = MCPServer(id="test-mcp")
    response = await server.dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/nope"})
    assert response["id"] == 1
    assert response["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_unknown_tool_is_invalid_params():
    server = MCPServer(id="test-mcp", tools={"echo": echo})
    response = await server.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "nope", "arguments": {}}}
    )
    assert response["error"]["code"] == -32602
    assert "nope" in response["error"]["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params",
    [
        {"name": "echo", "arguments": "hello"},
        {"name": "echo", "arguments": ["hello"]},
        {"arguments": {"message": "hello"}},
        {"name": 7, "arguments": {}},
    ],
)
async def test_malformed_tool_call_is_invalid_params(params):
    server = MCPServer(id="test-mcp", tools={"echo": echo})
    response = await server.dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params})
    assert response["error"]["code"] == -32602


@pytest.mark.asyncio
async def test_params_that_are_not_an_object_are_invalid_params():
    server = MCPServer(id="test-mcp")
    response = await server.dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": [1]})
    assert response["error"]["code"] == -32602


@pytest.mark.asyncio
async def test_a_failing_tool_is_a_tool_error_not_a_protocol_error():
    @tool
    async def explode(ctx: Context) -> str:
        """Always fails."""
        raise RuntimeError("boom")

    server = MCPServer(id="test-mcp", tools={"explode": explode})
    response = await server.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "explode", "arguments": {}}}
    )
    assert "error" not in response
    assert response["result"]["isError"] is True
    assert "boom" in response["result"]["content"][0]["text"]


@pytest.mark.asyncio
async def test_unknown_prompt_and_resource_errors():
    server = MCPServer(id="test-mcp")
    prompt = await server.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "prompts/get", "params": {"name": "nope"}}
    )
    assert prompt["error"]["code"] == -32602
    resource = await server.dispatch(
        {"jsonrpc": "2.0", "id": 2, "method": "resources/read", "params": {"uri": "docs://nope"}}
    )
    assert resource["error"]["code"] == -32002


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_",
    [
        {"id": 1, "method": "ping"},
        {"jsonrpc": "1.0", "id": 1, "method": "ping"},
        {"jsonrpc": 2.0, "id": 1, "method": "ping"},
    ],
)
async def test_request_without_jsonrpc_2_is_invalid_request(request_):
    server = MCPServer(id="test-mcp")
    response = await server.dispatch(request_)
    assert response["id"] == 1
    assert response["error"]["code"] == -32600


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_",
    [
        [{"jsonrpc": "2.0", "id": 1, "method": "ping"}],
        "ping",
        {"jsonrpc": "2.0", "id": None, "method": "ping"},
        {"jsonrpc": "2.0", "id": True, "method": "ping"},
        {"jsonrpc": "2.0", "id": 1},
        {"jsonrpc": "2.0", "id": 1, "method": 5},
    ],
)
async def test_malformed_requests_are_invalid_request(request_):
    server = MCPServer(id="test-mcp")
    response = await server.dispatch(request_)
    assert response["error"]["code"] == -32600


def test_stdio_answers_unparseable_lines_with_parse_error(monkeypatch):
    server = MCPServer(id="test-mcp")
    replies = _serve_stdio(
        server,
        monkeypatch,
        "{not json",
        {"jsonrpc": "2.0", "id": 1, "method": "ping"},
    )
    assert replies[0] == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": -32700, "message": replies[0]["error"]["message"]},
    }
    assert replies[1]["id"] == 1 and replies[1]["result"] == {}


def test_stdio_error_codes_end_to_end(monkeypatch):
    server = MCPServer(id="test-mcp", tools={"echo": echo})
    replies = _serve_stdio(
        server,
        monkeypatch,
        {"jsonrpc": "2.0", "id": 1, "method": "nope"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "nope"}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "echo", "arguments": 1}},
        {"id": 4, "method": "ping"},
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "echo", "arguments": {"message": "hi"}}},
    )
    assert [(reply["id"], reply.get("error", {}).get("code")) for reply in replies] == [
        (1, -32601),
        (2, -32602),
        (3, -32602),
        (4, -32600),
        (5, None),
    ]
    assert replies[4]["result"]["content"][0]["text"] == "echo:hi"


def test_the_post_only_http_transport_is_gone():
    server = MCPServer(id="test-mcp")
    assert not hasattr(server, "run_http")
    assert not hasattr(server, "_start_http_server")
