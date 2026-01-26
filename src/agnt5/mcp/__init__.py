"""MCP (Model Context Protocol) support for AGNT5.

This module provides MCP client functionality for connecting to external
MCP servers and using their tools within AGNT5 agents.

Example:
    ```python
    from agnt5.mcp import MCPClient

    # Create client with server configurations
    async with MCPClient(
        id="research-client",
        servers={
            "wikipedia": {"command": "npx", "args": ["-y", "wikipedia-mcp"]},
            "weather": {"url": "https://smithery.ai/.../weather-mcp"},
        }
    ) as client:
        # List available tools
        tools = await client.list_tools()

        # Call a tool
        result = await client.call_tool("wikipedia", "search", {"query": "Rust"})
        print(result.get_text())
    ```

For integration with AGNT5 agents:
    ```python
    from agnt5 import Agent
    from agnt5.mcp import MCPClient

    # Create MCP client
    mcp = MCPClient(
        id="agent-tools",
        servers={"wikipedia": {"command": "npx", "args": ["-y", "wikipedia-mcp"]}}
    )
    await mcp.connect()

    # Get tools as AGNT5 Tool objects
    tools = await mcp.list_tools()

    # Use with agent (tools are automatically converted)
    agent = Agent(name="research", model="openai/gpt-4o", tools=tools)
    result = await agent.run("What is the population of France?")
    ```
"""

from .client import MCPClient, MCPError
from .types import (
    CallToolResult,
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

__all__ = [
    # Client
    "MCPClient",
    "MCPError",
    # Types
    "CallToolResult",
    "McpTool",
    "McpToolWithServer",
    "ServerCapabilities",
    "ServerConfig",
    "ServerInfo",
    "SseConfig",
    "StdioConfig",
    "ToolContent",
    "TransportType",
]
