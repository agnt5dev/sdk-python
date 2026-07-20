"""
Tests for MCP (Model Context Protocol) module.

Tests cover:
- MCP type definitions
- ServerConfig creation from dict
- MCPClient initialization and configuration
- Transport type selection
"""

import pytest

from agnt5.mcp import (
    CallToolResult,
    MCPClient,
    MCPError,
    McpTool,
    McpToolWithServer,
    ServerCapabilities,
    ServerConfig,
    ServerInfo,
    SseConfig,
    StdioConfig,
    ToolContent,
    TransportType,
)

# =============================================================================
# TYPE TESTS
# =============================================================================


def test_server_capabilities_defaults():
    """Test ServerCapabilities default values."""
    caps = ServerCapabilities()
    assert caps.has_tools is False
    assert caps.has_resources is False
    assert caps.has_prompts is False


def test_server_capabilities_with_values():
    """Test ServerCapabilities with explicit values."""
    caps = ServerCapabilities(has_tools=True, has_resources=True, has_prompts=False)
    assert caps.has_tools is True
    assert caps.has_resources is True
    assert caps.has_prompts is False


def test_server_info():
    """Test ServerInfo creation."""
    info = ServerInfo(name="test-server", version="1.0.0")
    assert info.name == "test-server"
    assert info.version == "1.0.0"


def test_mcp_tool():
    """Test McpTool creation."""
    tool = McpTool(
        name="search",
        description="Search for information",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    assert tool.name == "search"
    assert tool.description == "Search for information"
    assert tool.input_schema is not None


def test_mcp_tool_minimal():
    """Test McpTool with minimal fields."""
    tool = McpTool(name="simple")
    assert tool.name == "simple"
    assert tool.description is None
    assert tool.input_schema is None


def test_mcp_tool_with_server():
    """Test McpToolWithServer creation."""
    tool = McpTool(name="search", description="Search")
    tool_with_server = McpToolWithServer(server="wikipedia", tool=tool)
    assert tool_with_server.server == "wikipedia"
    assert tool_with_server.tool.name == "search"


def test_tool_content_text():
    """Test ToolContent with text content."""
    content = ToolContent(type="text", text="Hello, world!")
    assert content.is_text is True
    assert content.is_image is False
    assert content.text == "Hello, world!"


def test_tool_content_image():
    """Test ToolContent with image content."""
    content = ToolContent(type="image", data="base64data", mime_type="image/png")
    assert content.is_text is False
    assert content.is_image is True
    assert content.data == "base64data"
    assert content.mime_type == "image/png"


def test_call_tool_result():
    """Test CallToolResult creation and text extraction."""
    result = CallToolResult(
        content=[
            ToolContent(type="text", text="First line"),
            ToolContent(type="text", text="Second line"),
        ],
        is_error=False,
    )
    assert result.is_error is False
    assert result.get_text() == "First line"
    assert result.get_all_text() == "First line\nSecond line"


def test_call_tool_result_empty():
    """Test CallToolResult with no content."""
    result = CallToolResult(content=[], is_error=False)
    assert result.get_text() is None
    assert result.get_all_text() == ""


def test_call_tool_result_error():
    """Test CallToolResult with error."""
    result = CallToolResult(
        content=[ToolContent(type="text", text="Error message")],
        is_error=True,
    )
    assert result.is_error is True
    assert result.get_text() == "Error message"


# =============================================================================
# CONFIG TESTS
# =============================================================================


def test_stdio_config():
    """Test StdioConfig creation."""
    config = StdioConfig(
        command="npx",
        args=["-y", "wikipedia-mcp"],
        env={"NODE_ENV": "production"},
        cwd="/tmp",
    )
    assert config.command == "npx"
    assert config.args == ["-y", "wikipedia-mcp"]
    assert config.env == {"NODE_ENV": "production"}
    assert config.cwd == "/tmp"


def test_stdio_config_defaults():
    """Test StdioConfig default values."""
    config = StdioConfig(command="python")
    assert config.command == "python"
    assert config.args == []
    assert config.env == {}
    assert config.cwd is None


def test_sse_config():
    """Test SseConfig creation."""
    config = SseConfig(
        url="https://example.com/mcp",
        headers={"Authorization": "Bearer token"},
    )
    assert config.url == "https://example.com/mcp"
    assert config.headers == {"Authorization": "Bearer token"}


def test_sse_config_with_api_key():
    """Test SseConfig with_api_key method."""
    config = SseConfig(url="https://example.com/mcp")
    config = config.with_api_key("my-api-key")
    assert config.headers == {"X-API-KEY": "my-api-key"}


def test_server_config_from_stdio():
    """Test ServerConfig.from_stdio factory method."""
    config = ServerConfig.from_stdio(
        command="npx",
        args=["-y", "test-mcp"],
        env={"DEBUG": "1"},
    )
    assert config.transport_type == TransportType.STDIO
    assert config.stdio is not None
    assert config.stdio.command == "npx"
    assert config.sse is None


def test_server_config_from_sse():
    """Test ServerConfig.from_sse factory method."""
    config = ServerConfig.from_sse(
        url="https://example.com/mcp",
        headers={"X-API-KEY": "key"},
    )
    assert config.transport_type == TransportType.SSE
    assert config.sse is not None
    assert config.sse.url == "https://example.com/mcp"
    assert config.stdio is None


def test_server_config_from_dict_stdio():
    """Test ServerConfig.from_dict with stdio configuration."""
    config = ServerConfig.from_dict({
        "command": "npx",
        "args": ["-y", "wikipedia-mcp"],
        "env": {"DEBUG": "1"},
    })
    assert config.transport_type == TransportType.STDIO
    assert config.stdio is not None
    assert config.stdio.command == "npx"


def test_server_config_from_dict_sse():
    """Test ServerConfig.from_dict with SSE configuration."""
    config = ServerConfig.from_dict({
        "url": "https://example.com/mcp",
        "headers": {"X-API-KEY": "key"},
    })
    assert config.transport_type == TransportType.SSE
    assert config.sse is not None
    assert config.sse.url == "https://example.com/mcp"


def test_server_config_from_dict_invalid():
    """Test ServerConfig.from_dict with invalid configuration."""
    with pytest.raises(ValueError, match="must have 'command' or 'url'"):
        ServerConfig.from_dict({"invalid": "config"})


# =============================================================================
# CLIENT TESTS
# =============================================================================


def test_mcp_client_creation():
    """Test MCPClient creation."""
    client = MCPClient(id="test-client")
    assert client.id == "test-client"
    assert client.connected_servers() == []


def test_mcp_client_with_servers():
    """Test MCPClient creation with servers."""
    client = MCPClient(
        id="test-client",
        servers={
            "wikipedia": {"command": "npx", "args": ["-y", "wikipedia-mcp"]},
            "remote": {"url": "https://example.com/mcp"},
        },
    )
    assert client.id == "test-client"
    # Servers are configured but not connected yet
    assert client.connected_servers() == []


def test_mcp_client_add_server():
    """Test adding server configuration."""
    client = MCPClient(id="test-client")

    # Add via dict
    client.add_server("wiki", {"command": "npx", "args": ["-y", "wikipedia-mcp"]})

    # Add via ServerConfig
    config = ServerConfig.from_sse(url="https://example.com/mcp")
    client.add_server("remote", config)


def test_mcp_client_add_stdio_server():
    """Test add_stdio_server convenience method."""
    client = MCPClient(id="test-client")
    client.add_stdio_server(
        name="wiki",
        command="npx",
        args=["-y", "wikipedia-mcp"],
        env={"DEBUG": "1"},
    )


def test_mcp_client_add_sse_server():
    """Test add_sse_server convenience method."""
    client = MCPClient(id="test-client")
    client.add_sse_server(
        name="remote",
        url="https://example.com/mcp",
        api_key="my-api-key",
    )


def test_mcp_client_is_connected_before_connect():
    """Test is_connected returns False before connection."""
    client = MCPClient(id="test-client")
    client.add_stdio_server("wiki", "npx", ["-y", "wikipedia-mcp"])
    assert client.is_connected("wiki") is False
    assert client.is_connected("unknown") is False


def test_mcp_error():
    """Test MCPError exception."""
    error = MCPError("Connection failed")
    assert str(error) == "Connection failed"
    assert isinstance(error, Exception)


# =============================================================================
# ASYNC TESTS (require running event loop)
# =============================================================================


@pytest.mark.asyncio
async def test_mcp_client_list_tools_empty():
    """Test list_tools returns empty list when not connected."""
    client = MCPClient(id="test-client")
    tools = await client.list_tools()
    assert tools == []


@pytest.mark.asyncio
async def test_mcp_client_call_tool_not_connected():
    """Test call_tool raises error when server not connected."""
    client = MCPClient(id="test-client")
    with pytest.raises(MCPError, match="Server not connected"):
        await client.call_tool("wiki", "search", {"query": "test"})


@pytest.mark.asyncio
async def test_mcp_client_list_server_tools_not_connected():
    """Test list_server_tools raises error when server not connected."""
    client = MCPClient(id="test-client")
    with pytest.raises(MCPError, match="Server not connected"):
        await client.list_server_tools("wiki")


@pytest.mark.asyncio
async def test_mcp_client_call_tool_auto_not_found():
    """Test call_tool_auto raises error when tool not found."""
    client = MCPClient(id="test-client")
    with pytest.raises(MCPError, match="Tool not found"):
        await client.call_tool_auto("unknown_tool", {})


@pytest.mark.asyncio
async def test_mcp_client_context_manager():
    """Test MCPClient context manager (without actual connection)."""
    # This tests the async context manager protocol
    # Actual connection would fail without MCP server
    client = MCPClient(id="test-client")
    # We can at least verify the protocol is implemented
    assert hasattr(client, "__aenter__")
    assert hasattr(client, "__aexit__")


@pytest.mark.asyncio
async def test_mcp_client_get_tools_empty():
    """Test get_tools returns empty list when not connected."""
    client = MCPClient(id="test-client")
    tools = client.get_tools()
    assert tools == []
