"""MCP Client implementation for Python SDK.

This module provides an MCPClient for connecting to external MCP servers
and using their tools within AGNT5 agents.

Example:
    ```python
    from agnt5.mcp import MCPClient

    # Create client with server configurations
    client = MCPClient(
        id="research-client",
        servers={
            "wikipedia": {"command": "npx", "args": ["-y", "wikipedia-mcp"]},
            "agnt5": {"url": "http://localhost:34183/v1/mcp/sse", "headers": {"X-API-KEY": "..."}},
        }
    )

    # Connect to all servers
    await client.connect()

    # List available tools
    tools = await client.list_tools()

    # Call a tool
    result = await client.call_tool("wikipedia", "search", {"query": "Rust"})

    # Disconnect when done
    await client.disconnect()
    ```
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from agnt5.tool import Tool

from .types import (
    CallToolResult,
    McpTool,
    McpToolWithServer,
    ServerCapabilities,
    ServerConfig,
    ServerInfo,
    ToolContent,
    TransportType,
)

logger = logging.getLogger(__name__)


class MCPError(Exception):
    """MCP-specific error."""

    pass


class MCPClient:
    """MCP Client for connecting to external MCP servers.

    The client manages connections to one or more MCP servers and provides
    a unified interface for discovering and calling tools.

    Args:
        id: Client identifier.
        servers: Dictionary mapping server names to configurations.
                 Each config can be:
                 - {"command": "npx", "args": ["-y", "wikipedia-mcp"]} for stdio
                 - {"url": "https://...", "headers": {...}} for SSE
    """

    def __init__(
        self,
        id: str,
        servers: Optional[dict[str, dict[str, Any]]] = None,
    ):
        self._id = id
        self._server_configs: dict[str, ServerConfig] = {}
        self._connections: dict[str, _ServerConnection] = {}

        # Parse server configurations
        if servers:
            for name, config in servers.items():
                self._server_configs[name] = ServerConfig.from_dict(config)

    @property
    def id(self) -> str:
        """Get client identifier."""
        return self._id

    def add_server(self, name: str, config: dict[str, Any] | ServerConfig) -> None:
        """Add a server configuration.

        Args:
            name: Server name.
            config: Server configuration (dict or ServerConfig).
        """
        if isinstance(config, dict):
            config = ServerConfig.from_dict(config)
        self._server_configs[name] = config

    def add_stdio_server(
        self,
        name: str,
        command: str,
        args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> None:
        """Add a stdio server (convenience method).

        Args:
            name: Server name.
            command: Command to run (e.g., "npx", "uvx").
            args: Command arguments.
            env: Environment variables.
            cwd: Working directory.
        """
        self._server_configs[name] = ServerConfig.from_stdio(
            command=command,
            args=args,
            env=env,
            cwd=cwd,
        )

    def add_sse_server(
        self,
        name: str,
        url: str,
        headers: Optional[dict[str, str]] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Add an SSE server (convenience method).

        Args:
            name: Server name.
            url: Server URL.
            headers: HTTP headers.
            api_key: API key (added as X-API-KEY header).
        """
        headers = headers or {}
        if api_key:
            headers["X-API-KEY"] = api_key
        self._server_configs[name] = ServerConfig.from_sse(url=url, headers=headers)

    async def connect(self) -> None:
        """Connect to all configured servers."""
        for name, config in self._server_configs.items():
            await self._connect_server(name, config)

    async def _connect_server(self, name: str, config: ServerConfig) -> None:
        """Connect to a specific server."""
        logger.info(f"Connecting to MCP server: {name}")

        try:
            if config.transport_type == TransportType.STDIO:
                transport = await _StdioTransport.create(config.stdio)
            elif config.transport_type == TransportType.SSE:
                transport = await _SseTransport.create(config.sse)
            else:
                raise MCPError(f"Unknown transport type: {config.transport_type}")

            # Initialize connection
            init_result = await transport.request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "agnt5-sdk", "version": "0.1.0"},
                },
            )

            server_info = ServerInfo(
                name=init_result.get("serverInfo", {}).get("name", "unknown"),
                version=init_result.get("serverInfo", {}).get("version", "unknown"),
            )

            caps = init_result.get("capabilities", {})
            capabilities = ServerCapabilities(
                has_tools="tools" in caps,
                has_resources="resources" in caps,
                has_prompts="prompts" in caps,
            )

            logger.info(
                f"Connected to MCP server {server_info.name} "
                f"(version: {server_info.version})"
            )

            # Send initialized notification
            await transport.notify("initialized", {})

            # List available tools
            tools = []
            if capabilities.has_tools:
                tools_result = await transport.request("tools/list", {})
                for t in tools_result.get("tools", []):
                    tools.append(
                        McpTool(
                            name=t["name"],
                            description=t.get("description"),
                            input_schema=t.get("inputSchema"),
                        )
                    )

            logger.info(f"MCP server {name} has {len(tools)} tools available")

            # Store connection
            self._connections[name] = _ServerConnection(
                transport=transport,
                server_info=server_info,
                capabilities=capabilities,
                tools=tools,
            )

        except Exception as e:
            logger.error(f"Failed to connect to MCP server {name}: {e}")
            raise MCPError(f"Failed to connect to {name}: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from all servers."""
        for name, conn in list(self._connections.items()):
            logger.info(f"Disconnecting from MCP server: {name}")
            await conn.transport.close()
            del self._connections[name]

    async def list_tools(self) -> list[McpToolWithServer]:
        """List all available tools from all connected servers.

        Returns:
            List of tools with server information.
        """
        all_tools = []
        for name, conn in self._connections.items():
            for tool in conn.tools:
                all_tools.append(McpToolWithServer(server=name, tool=tool))
        return all_tools

    async def list_server_tools(self, server: str) -> list[McpTool]:
        """List tools from a specific server.

        Args:
            server: Server name.

        Returns:
            List of tools.

        Raises:
            MCPError: If server is not connected.
        """
        conn = self._connections.get(server)
        if not conn:
            raise MCPError(f"Server not connected: {server}")
        return conn.tools.copy()

    def get_tools(self) -> list["Tool"]:
        """Get MCP tools as AGNT5 Tool objects.

        Returns a list of Tool objects that can be passed directly to an Agent.
        Each tool wraps the MCP call_tool method.

        Example:
            ```python
            mcp = MCPClient(...)
            await mcp.connect()

            # Get tools as AGNT5 Tool objects
            tools = mcp.get_tools()

            # Use with agent
            agent = Agent(name="research", model="openai/gpt-4o", tools=tools)
            result = await agent.run("Search for Python")
            ```

        Returns:
            List of AGNT5 Tool objects.
        """
        from agnt5.tool import Tool
        from agnt5.context import Context

        tools = []
        for server_name, conn in self._connections.items():
            for mcp_tool in conn.tools:
                # Create a closure to capture server_name and tool_name
                def make_handler(srv: str, name: str):
                    async def handler(ctx: Context, **kwargs) -> str:
                        result = await self.call_tool(srv, name, kwargs)
                        if result.is_error:
                            raise MCPError(
                                f"MCP tool error: {result.get_text() or 'Unknown error'}"
                            )
                        return result.get_text() or ""

                    return handler

                tool = Tool(
                    name=mcp_tool.name,
                    description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
                    handler=make_handler(server_name, mcp_tool.name),
                    input_schema=mcp_tool.input_schema or {"type": "object", "properties": {}},
                )
                tools.append(tool)

        return tools

    async def call_tool(
        self,
        server: str,
        tool_name: str,
        arguments: Optional[dict[str, Any]] = None,
    ) -> CallToolResult:
        """Call a tool on a specific server.

        Args:
            server: Server name.
            tool_name: Tool name.
            arguments: Tool arguments.

        Returns:
            Tool execution result.

        Raises:
            MCPError: If server is not connected or tool not found.
        """
        conn = self._connections.get(server)
        if not conn:
            raise MCPError(f"Server not connected: {server}")

        # Verify tool exists
        if not any(t.name == tool_name for t in conn.tools):
            raise MCPError(f"Tool not found: {tool_name}")

        result = await conn.transport.request(
            "tools/call",
            {"name": tool_name, "arguments": arguments or {}},
        )

        # Parse result
        content = []
        for c in result.get("content", []):
            content.append(
                ToolContent(
                    type=c.get("type", "text"),
                    text=c.get("text"),
                    data=c.get("data"),
                    mime_type=c.get("mimeType"),
                )
            )

        return CallToolResult(
            content=content,
            is_error=result.get("isError", False),
        )

    async def call_tool_auto(
        self,
        tool_name: str,
        arguments: Optional[dict[str, Any]] = None,
    ) -> CallToolResult:
        """Call a tool by finding it across all servers.

        Args:
            tool_name: Tool name.
            arguments: Tool arguments.

        Returns:
            Tool execution result.

        Raises:
            MCPError: If tool not found in any server.
        """
        # Find server with this tool
        for name, conn in self._connections.items():
            if any(t.name == tool_name for t in conn.tools):
                return await self.call_tool(name, tool_name, arguments)

        raise MCPError(f"Tool not found: {tool_name}")

    def is_connected(self, server: str) -> bool:
        """Check if a server is connected.

        Args:
            server: Server name.

        Returns:
            True if connected.
        """
        conn = self._connections.get(server)
        return conn is not None and conn.transport.is_connected

    def connected_servers(self) -> list[str]:
        """Get list of connected server names."""
        return list(self._connections.keys())

    async def __aenter__(self) -> "MCPClient":
        """Async context manager entry - connects to all servers."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - disconnects from all servers."""
        await self.disconnect()


class _ServerConnection:
    """Internal class representing a connection to an MCP server."""

    def __init__(
        self,
        transport: "_Transport",
        server_info: ServerInfo,
        capabilities: ServerCapabilities,
        tools: list[McpTool],
    ):
        self.transport = transport
        self.server_info = server_info
        self.capabilities = capabilities
        self.tools = tools


class _Transport:
    """Base transport interface."""

    @property
    def is_connected(self) -> bool:
        raise NotImplementedError

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError


class _StdioTransport(_Transport):
    """Stdio transport for subprocess communication."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        self._process = process
        self._reader = reader
        self._writer = writer
        self._request_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._connected = True
        self._read_task: Optional[asyncio.Task] = None

    @classmethod
    async def create(cls, config) -> "_StdioTransport":
        """Create a new stdio transport."""
        from .types import StdioConfig

        if not isinstance(config, StdioConfig):
            raise MCPError("Invalid config for stdio transport")

        # Start process
        process = await asyncio.create_subprocess_exec(
            config.command,
            *config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**dict(__import__("os").environ), **config.env} if config.env else None,
            cwd=config.cwd,
        )

        if process.stdout is None or process.stdin is None:
            raise MCPError("Failed to get process streams")

        transport = cls(
            process=process,
            reader=process.stdout,
            writer=process.stdin,
        )

        # Start background reader
        transport._read_task = asyncio.create_task(transport._read_loop())

        return transport

    @property
    def is_connected(self) -> bool:
        return self._connected and self._process.returncode is None

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise MCPError("Transport not connected")

        self._request_id += 1
        req_id = self._request_id

        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": req_id,
        }

        # Create future for response
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future

        try:
            await self._send(request)
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise MCPError("Request timed out")

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        if not self.is_connected:
            raise MCPError("Transport not connected")

        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        await self._send(request)

    async def close(self) -> None:
        self._connected = False
        if self._read_task:
            self._read_task.cancel()
        if self._writer:
            self._writer.close()
        if self._process:
            self._process.terminate()
            await self._process.wait()

    async def _send(self, data: dict[str, Any]) -> None:
        """Send a message with Content-Length framing."""
        message = json.dumps(data).encode("utf-8")
        header = f"Content-Length: {len(message)}\r\n\r\n"
        self._writer.write(header.encode("utf-8"))
        self._writer.write(message)
        await self._writer.drain()

    async def _read_loop(self) -> None:
        """Background task to read responses."""
        try:
            while self._connected:
                message = await self._read_message()
                if message is None:
                    break

                data = json.loads(message)
                req_id = data.get("id")

                if req_id and req_id in self._pending:
                    future = self._pending.pop(req_id)
                    if "error" in data:
                        future.set_exception(
                            MCPError(data["error"].get("message", "Unknown error"))
                        )
                    else:
                        future.set_result(data.get("result", {}))

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Read loop error: {e}")
        finally:
            self._connected = False
            # Fail all pending requests
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(MCPError("Connection closed"))
            self._pending.clear()

    async def _read_message(self) -> Optional[bytes]:
        """Read a Content-Length framed message."""
        # Read headers
        content_length = None
        while True:
            line = await self._reader.readline()
            if not line:
                return None  # EOF

            line = line.decode("utf-8").strip()
            if not line:
                break  # End of headers

            if line.lower().startswith("content-length:"):
                content_length = int(line.split(":")[1].strip())

        if content_length is None:
            raise MCPError("Missing Content-Length header")

        # Read body
        body = await self._reader.readexactly(content_length)
        return body


class _SseTransport(_Transport):
    """SSE transport for HTTP-based servers."""

    def __init__(self, url: str, headers: dict[str, str]):
        self._url = url
        self._headers = headers
        self._session: Optional[Any] = None  # httpx.AsyncClient
        self._session_id: Optional[str] = None
        self._connected = False
        self._request_id = 0

    @classmethod
    async def create(cls, config) -> "_SseTransport":
        """Create a new SSE transport."""
        from .types import SseConfig

        if not isinstance(config, SseConfig):
            raise MCPError("Invalid config for SSE transport")

        transport = cls(url=config.url, headers=config.headers)
        await transport._connect()
        return transport

    async def _connect(self) -> None:
        """Connect to SSE stream."""
        try:
            import httpx
        except ImportError:
            raise MCPError("httpx is required for SSE transport: pip install httpx")

        self._session = httpx.AsyncClient(headers=self._headers, timeout=60.0)

        # Make initial SSE connection to get session ID
        try:
            async with self._session.stream("GET", self._url) as response:
                if response.status_code != 200:
                    raise MCPError(f"SSE connection failed: {response.status_code}")

                # Read first few events to get session ID
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "session" in data:
                            self._session_id = data["session"]
                            break
                        # Also check for capabilities message
                        if "capabilities" in data:
                            break

        except Exception as e:
            # For stateless endpoints, connection might just work
            logger.debug(f"SSE stream setup: {e}")

        self._connected = True

    @property
    def is_connected(self) -> bool:
        return self._connected and self._session is not None

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise MCPError("Transport not connected")

        self._request_id += 1
        req_id = self._request_id

        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": req_id,
        }

        # Build URL with session
        url = self._url
        if self._session_id:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}session={self._session_id}"

        try:
            response = await self._session.post(url, json=request)
            if response.status_code != 200:
                raise MCPError(f"Request failed: {response.status_code}")

            data = response.json()
            if "error" in data:
                raise MCPError(data["error"].get("message", "Unknown error"))

            return data.get("result", {})

        except Exception as e:
            if "MCPError" in str(type(e)):
                raise
            raise MCPError(f"Request failed: {e}") from e

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        if not self.is_connected:
            raise MCPError("Transport not connected")

        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        url = self._url
        if self._session_id:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}session={self._session_id}"

        await self._session.post(url, json=request)

    async def close(self) -> None:
        self._connected = False
        if self._session:
            await self._session.aclose()
            self._session = None
