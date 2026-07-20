"""
Example: AGNT5 MCP Client

This example demonstrates connecting to external MCP servers:
- MCPClient for managing MCP server connections
- Stdio transport for local subprocess servers (npx, uvx)
- SSE transport for remote HTTP servers
- Integration with AGNT5 Agent's tool system

MCP (Model Context Protocol) allows agents to use tools from external servers:
- Community MCP servers (Wikipedia, GitHub, file system, etc.)
- Smithery and Composio hosted servers
- AGNT5 Gateway MCP endpoints

Each component can be:
- Run standalone: python ex_17_mcp_client.py
- Used in workflows and agents

Prerequisites:
- Set OPENAI_API_KEY for Agent examples
- Install MCP servers: npx wikipedia-mcp, etc.
"""

import asyncio
import os

from agnt5 import Agent, Context
from agnt5.mcp import MCPClient


# =============================================================================
# MCP CLIENT EXAMPLES
# =============================================================================


async def example_stdio_mcp():
    """
    Connect to a local MCP server via stdio transport.

    This example connects to the Wikipedia MCP server which provides
    search and page retrieval capabilities.

    Note: Requires npx and the wikipedia-mcp package.
    Run: npx -y wikipedia-mcp to test the server first.
    """
    print("\n--- MCP Stdio Transport (Wikipedia) ---")

    # Create client with stdio server configuration
    async with MCPClient(
        id="wiki-client",
        servers={
            "wikipedia": {
                "command": "npx",
                "args": ["-y", "wikipedia-mcp"],
            }
        },
    ) as client:
        # List available tools
        tools = await client.list_tools()
        print(f"Found {len(tools)} tools:")
        for t in tools:
            print(f"  - {t.server}/{t.tool.name}: {t.tool.description or 'No description'}")

        # Call a tool directly
        if any(t.tool.name == "search" for t in tools):
            result = await client.call_tool("wikipedia", "search", {"query": "Python programming"})
            print(f"\nSearch result: {result.get_text()[:200]}...")


async def example_sse_mcp():
    """
    Connect to a remote MCP server via SSE transport.

    This example connects to the AGNT5 Gateway MCP endpoint which
    exposes registered tools, workflows, and agents.

    Note: Requires AGNT5 Gateway running with MCP enabled.
    """
    print("\n--- MCP SSE Transport (AGNT5 Gateway) ---")

    # Get API key from environment
    api_key = os.getenv("AGNT5_API_KEY", "demo-api-key")
    gateway_url = os.getenv("AGNT5_GATEWAY_URL", "http://localhost:34183")

    # Create client with SSE server configuration
    client = MCPClient(
        id="gateway-client",
    )
    client.add_sse_server(
        name="agnt5",
        url=f"{gateway_url}/v1/mcp/sse",
        api_key=api_key,
    )

    try:
        await client.connect()

        # List available tools
        tools = await client.list_tools()
        print(f"Found {len(tools)} tools from AGNT5 Gateway:")
        for t in tools:
            print(f"  - {t.tool.name}: {t.tool.description or 'No description'}")

    except Exception as e:
        print(f"Could not connect to Gateway: {e}")
        print("Make sure the AGNT5 Gateway is running with MCP enabled.")
    finally:
        await client.disconnect()


async def example_multiple_servers():
    """
    Connect to multiple MCP servers simultaneously.

    This demonstrates the client's ability to manage connections
    to multiple servers and aggregate their tools.
    """
    print("\n--- Multiple MCP Servers ---")

    # Create client with multiple servers
    client = MCPClient(
        id="multi-client",
        servers={
            # Local stdio server
            "wikipedia": {
                "command": "npx",
                "args": ["-y", "wikipedia-mcp"],
            },
            # Would also work with remote servers:
            # "weather": {
            #     "url": "https://smithery.ai/.../weather-mcp",
            #     "headers": {"X-API-KEY": "your-key"},
            # },
        },
    )

    try:
        await client.connect()

        # List all tools from all servers
        tools = await client.list_tools()
        print(f"Total tools from all servers: {len(tools)}")

        # Group by server
        servers = {}
        for t in tools:
            if t.server not in servers:
                servers[t.server] = []
            servers[t.server].append(t.tool.name)

        for server, tool_names in servers.items():
            print(f"\n  {server}:")
            for name in tool_names:
                print(f"    - {name}")

        # call_tool_auto finds the right server automatically
        if any(t.tool.name == "search" for t in tools):
            result = await client.call_tool_auto("search", {"query": "Rust programming"})
            print(f"\nAuto tool call result: {result.get_text()[:100]}...")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.disconnect()


async def example_agent_with_mcp_tools():
    """
    Use MCP tools with an AGNT5 Agent.

    This demonstrates the integration between MCPClient and Agent:
    - Connect to MCP servers
    - Get tools as AGNT5 Tool objects
    - Create an agent with MCP tools
    - Agent can now use external MCP tools

    Note: Requires OPENAI_API_KEY for the agent.
    """
    print("\n--- Agent with MCP Tools ---")

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Skipping agent example.")
        return

    # Create MCP client and connect
    mcp = MCPClient(
        id="agent-mcp",
        servers={
            "wikipedia": {
                "command": "npx",
                "args": ["-y", "wikipedia-mcp"],
            }
        },
    )

    try:
        await mcp.connect()

        # Convert MCP tools to AGNT5 Tool objects
        tools = mcp.get_tools()
        print(f"Converted {len(tools)} MCP tools for agent use")

        # Create agent with MCP tools
        agent = Agent(
            name="research_agent",
            model="openai/gpt-4o-mini",
            instructions="""You are a helpful research assistant.
Use the available tools to search and retrieve information.
Provide concise, accurate answers based on the tool results.""",
            tools=tools,
            max_iterations=5,
        )

        # Run the agent
        result = await agent.run("What is the Python programming language?")
        print(f"\nAgent response:\n{result.output[:500]}...")
        print(f"\nTool calls made: {len(result.tool_calls)}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await mcp.disconnect()


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================


async def main():
    """Run examples standalone."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("=" * 60)
    print("AGNT5 MCP Client Example")
    print("=" * 60)

    # Note: These examples require MCP servers to be available.
    # Uncomment the examples you want to run.

    # Example 1: Stdio transport (Wikipedia MCP)
    # Requires: npx and internet access
    # await example_stdio_mcp()

    # Example 2: SSE transport (AGNT5 Gateway)
    # Requires: AGNT5 Gateway running with MCP enabled
    # await example_sse_mcp()

    # Example 3: Multiple servers
    # await example_multiple_servers()

    # Example 4: Agent with MCP tools
    # Requires: OPENAI_API_KEY and npx
    # await example_agent_with_mcp_tools()

    # Show the API without requiring external services.
    print("\nMCP Client API Example:")
    print("""
    from agnt5.mcp import MCPClient

    # Create and connect to MCP servers
    async with MCPClient(
        id="my-client",
        servers={
            "wikipedia": {"command": "npx", "args": ["-y", "wikipedia-mcp"]},
            "remote": {"url": "https://mcp-server.example.com/sse"},
        }
    ) as client:
        # List tools from all servers
        tools = await client.list_tools()

        # Call a tool directly
        result = await client.call_tool("wikipedia", "search", {"query": "Python"})

        # Or use with an Agent
        agent = Agent(name="researcher", model="openai/gpt-4o-mini", tools=client.get_tools())
        response = await agent.run("Tell me about Python")
    """)

    print("=" * 60)
    print("Done! Uncomment examples above to test with real MCP servers.")


if __name__ == "__main__":
    asyncio.run(main())
