"""Rust-backed MCP client facade for the Python SDK."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from .._compat import _rust_available
from .types import (
    CallToolResult,
    McpTool,
    McpToolWithServer,
    ServerConfig,
    ToolContent,
    TransportType,
)

if TYPE_CHECKING:
    from agnt5.tool import Tool

logger = logging.getLogger(__name__)

if _rust_available:
    from agnt5._core import MCPClientCore as RustMCPClientCore
else:
    RustMCPClientCore = None  # type: ignore


class MCPError(Exception):
    """MCP-specific error."""

    pass


class MCPClient:
    """Python facade over the Rust sdk-core MCP client."""

    def __init__(
        self,
        id: str,
        servers: Optional[dict[str, dict[str, Any]]] = None,
    ):
        if RustMCPClientCore is None:
            raise ImportError(
                "Rust extension not available. Rebuild with: cd sdk/sdk-python && maturin develop"
            )

        self._id = id
        self._server_configs: dict[str, ServerConfig] = {}
        self._connected_servers: set[str] = set()
        self._tool_cache: list[McpToolWithServer] = []
        self._core = RustMCPClientCore(id)

        if servers:
            for name, config in servers.items():
                self.add_server(name, config)

    @property
    def id(self) -> str:
        return self._id

    def add_server(self, name: str, config: dict[str, Any] | ServerConfig) -> None:
        config_dict = config if isinstance(config, dict) else _server_config_to_dict(config)
        self._server_configs[name] = ServerConfig.from_dict(config_dict)
        self._core.add_server(name, config_dict)

    def add_stdio_server(
        self,
        name: str,
        command: str,
        args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> None:
        self._server_configs[name] = ServerConfig.from_stdio(
            command=command,
            args=args,
            env=env,
            cwd=cwd,
        )
        self._core.add_stdio_server(name, command, args, env, cwd)

    def add_sse_server(
        self,
        name: str,
        url: str,
        headers: Optional[dict[str, str]] = None,
        api_key: Optional[str] = None,
    ) -> None:
        headers = dict(headers or {})
        if api_key:
            headers["X-API-KEY"] = api_key
        self._server_configs[name] = ServerConfig.from_sse(url=url, headers=headers)
        self._core.add_sse_server(name, url, headers, api_key)

    def add_streamable_http_server(
        self,
        name: str,
        url: str,
        headers: Optional[dict[str, str]] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Add a Streamable HTTP MCP server (2025-11-25 spec).

        Use this for current-spec remote MCP servers like
        `https://mcp.deepwiki.com/mcp`. For older HTTP+SSE servers, use
        `add_sse_server` instead.
        """
        headers = dict(headers or {})
        if api_key:
            headers["X-API-KEY"] = api_key
        self._server_configs[name] = ServerConfig.from_streamable_http(url=url, headers=headers)
        self._core.add_streamable_http_server(name, url, headers, api_key)

    async def connect(self) -> None:
        try:
            await self._core.connect()
            self._connected_servers = set(self._server_configs.keys())
            self._tool_cache = await self.list_tools()
        except Exception as e:
            logger.error("Failed to connect MCP servers: %s", e)
            raise MCPError(str(e)) from e

    async def disconnect(self) -> None:
        try:
            await self._core.disconnect()
        except Exception as e:
            raise MCPError(str(e)) from e
        finally:
            self._connected_servers.clear()
            self._tool_cache = []

    async def list_tools(self) -> list[McpToolWithServer]:
        try:
            tools = await self._core.list_tools()
        except Exception as e:
            raise MCPError(str(e)) from e

        result = [
            McpToolWithServer(
                server=item["server"],
                tool=_tool_from_dict(item["tool"]),
            )
            for item in tools
        ]
        self._tool_cache = result
        return result

    async def list_server_tools(self, server: str) -> list[McpTool]:
        try:
            tools = await self._core.list_server_tools(server)
        except Exception as e:
            raise MCPError(str(e)) from e
        return [_tool_from_dict(tool) for tool in tools]

    def get_tools(self) -> list["Tool"]:
        from agnt5.context import Context
        from agnt5.tool import Tool

        tools: list[Tool] = []
        for tool_with_server in self._tool_cache:
            server_name = tool_with_server.server
            mcp_tool = tool_with_server.tool

            def make_handler(srv: str, name: str):
                async def handler(ctx: Context, **kwargs) -> str:
                    result = await self.call_tool(srv, name, kwargs)
                    if result.is_error:
                        raise MCPError(
                            f"MCP tool error: {result.get_text() or 'Unknown error'}"
                        )
                    return result.get_text() or ""

                return handler

            tools.append(
                Tool(
                    name=mcp_tool.name,
                    description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
                    handler=make_handler(server_name, mcp_tool.name),
                    input_schema=mcp_tool.input_schema
                    or {"type": "object", "properties": {}},
                )
            )

        return tools

    async def call_tool(
        self,
        server: str,
        tool_name: str,
        arguments: Optional[dict[str, Any]] = None,
    ) -> CallToolResult:
        try:
            result = await self._core.call_tool(server, tool_name, arguments or {})
        except Exception as e:
            raise MCPError(str(e)) from e
        return _call_tool_result_from_dict(result)

    async def call_tool_auto(
        self,
        tool_name: str,
        arguments: Optional[dict[str, Any]] = None,
    ) -> CallToolResult:
        try:
            result = await self._core.call_tool_auto(tool_name, arguments or {})
        except Exception as e:
            raise MCPError(str(e)) from e
        return _call_tool_result_from_dict(result)

    def is_connected(self, server: str) -> bool:
        return server in self._connected_servers

    def connected_servers(self) -> list[str]:
        return sorted(self._connected_servers)

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()


def _tool_from_dict(data: dict[str, Any]) -> McpTool:
    return McpTool(
        name=data["name"],
        description=data.get("description"),
        input_schema=data.get("input_schema"),
    )


def _call_tool_result_from_dict(data: dict[str, Any]) -> CallToolResult:
    return CallToolResult(
        content=[
            ToolContent(
                type=item.get("type", "text"),
                text=item.get("text"),
                data=item.get("data"),
                mime_type=item.get("mime_type") or item.get("mimeType"),
            )
            for item in data.get("content", [])
        ],
        is_error=data.get("is_error", data.get("isError", False)),
    )


def _server_config_to_dict(config: ServerConfig) -> dict[str, Any]:
    if config.transport_type == TransportType.STDIO and config.stdio is not None:
        return {
            "command": config.stdio.command,
            "args": config.stdio.args,
            "env": config.stdio.env,
            "cwd": config.stdio.cwd,
        }
    if config.transport_type == TransportType.SSE and config.sse is not None:
        return {
            "url": config.sse.url,
            "headers": config.sse.headers,
            "transport": "sse",
        }
    if (
        config.transport_type == TransportType.STREAMABLE_HTTP
        and config.streamable_http is not None
    ):
        return {
            "url": config.streamable_http.url,
            "headers": config.streamable_http.headers,
            "transport": "streamable_http",
        }
    raise ValueError("Invalid server config")
