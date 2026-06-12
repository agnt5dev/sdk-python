"""MCP type definitions for Python SDK."""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional
from enum import Enum


@dataclass
class ServerCapabilities:
    """MCP server capabilities."""

    has_tools: bool = False
    has_resources: bool = False
    has_prompts: bool = False


@dataclass
class ServerInfo:
    """MCP server information."""

    name: str
    version: str


@dataclass
class McpTool:
    """MCP tool definition."""

    name: str
    description: Optional[str] = None
    input_schema: Optional[dict[str, Any]] = None


@dataclass
class McpToolWithServer:
    """MCP tool with server information."""

    server: str
    tool: McpTool


@dataclass
class ToolContent:
    """Tool result content."""

    type: str
    text: Optional[str] = None
    data: Optional[str] = None
    mime_type: Optional[str] = None

    @property
    def is_text(self) -> bool:
        """Check if content is text."""
        return self.type == "text"

    @property
    def is_image(self) -> bool:
        """Check if content is an image."""
        return self.type == "image"


@dataclass
class CallToolResult:
    """Result from calling an MCP tool."""

    content: list[ToolContent]
    is_error: bool = False

    def get_text(self) -> Optional[str]:
        """Get the first text content, if any."""
        for c in self.content:
            if c.is_text and c.text:
                return c.text
        return None

    def get_all_text(self) -> str:
        """Get all text content concatenated."""
        return "\n".join(c.text for c in self.content if c.is_text and c.text)


@dataclass
class StdioConfig:
    """Configuration for stdio transport."""

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None


@dataclass
class SseConfig:
    """Configuration for legacy HTTP+SSE transport (MCP 2024-11-05 spec)."""

    url: str
    headers: dict[str, str] = field(default_factory=dict)

    def with_api_key(self, api_key: str) -> "SseConfig":
        """Add API key header."""
        self.headers["X-API-KEY"] = api_key
        return self


@dataclass
class StreamableHttpConfig:
    """Configuration for Streamable HTTP transport (MCP 2025-11-25 spec).

    Single-endpoint POST/SSE transport replacing the older HTTP+SSE design.
    Use this for current-spec MCP servers (e.g. mcp.deepwiki.com/mcp).
    """

    url: str
    headers: dict[str, str] = field(default_factory=dict)

    def with_api_key(self, api_key: str) -> "StreamableHttpConfig":
        """Add API key header."""
        self.headers["X-API-KEY"] = api_key
        return self


class TransportType(Enum):
    """Transport type for MCP connection."""

    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable_http"


@dataclass
class ServerConfig:
    """Server configuration."""

    transport_type: TransportType
    stdio: Optional[StdioConfig] = None
    sse: Optional[SseConfig] = None
    streamable_http: Optional[StreamableHttpConfig] = None

    @classmethod
    def from_stdio(
        cls,
        command: str,
        args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> "ServerConfig":
        """Create stdio server config."""
        return cls(
            transport_type=TransportType.STDIO,
            stdio=StdioConfig(
                command=command,
                args=args or [],
                env=env or {},
                cwd=cwd,
            ),
        )

    @classmethod
    def from_sse(
        cls,
        url: str,
        headers: Optional[dict[str, str]] = None,
    ) -> "ServerConfig":
        """Create legacy HTTP+SSE server config."""
        return cls(
            transport_type=TransportType.SSE,
            sse=SseConfig(url=url, headers=headers or {}),
        )

    @classmethod
    def from_streamable_http(
        cls,
        url: str,
        headers: Optional[dict[str, str]] = None,
    ) -> "ServerConfig":
        """Create Streamable HTTP server config (MCP 2025-11-25)."""
        return cls(
            transport_type=TransportType.STREAMABLE_HTTP,
            streamable_http=StreamableHttpConfig(url=url, headers=headers or {}),
        )

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ServerConfig":
        """Create server config from dictionary.

        Supports three formats:
        1. {"command": "npx", "args": ["-y", "wikipedia-mcp"]}  -> stdio
        2. {"url": "https://...", "headers": {...}}  -> SSE (default)
        3. {"url": "https://...", "transport": "streamable_http"}  -> Streamable HTTP
        """
        if "command" in config:
            return cls.from_stdio(
                command=config["command"],
                args=config.get("args", []),
                env=config.get("env", {}),
                cwd=config.get("cwd"),
            )
        if "url" in config:
            transport = (config.get("transport") or "sse").lower()
            headers = config.get("headers", {})
            if transport in ("streamable_http", "streamable-http", "http"):
                return cls.from_streamable_http(url=config["url"], headers=headers)
            if transport == "sse":
                return cls.from_sse(url=config["url"], headers=headers)
            raise ValueError(
                f"Unknown transport: {transport!r} (expected 'sse' or 'streamable_http')"
            )
        raise ValueError("Invalid server config: must have 'command' or 'url'")


PromptHandler = Callable[..., Awaitable[Any]]
ResourceReadHandler = Callable[..., Awaitable[Any]]


@dataclass
class Prompt:
    """Prompt exposed by an MCP server."""

    name: str
    description: Optional[str] = None
    arguments_schema: Optional[dict[str, Any]] = None
    handler: Optional[PromptHandler] = None


@dataclass
class Resource:
    """Resource exposed by an MCP server."""

    uri: str
    name: str
    read: ResourceReadHandler
    description: Optional[str] = None
    mime_type: Optional[str] = None

    @classmethod
    def text(
        cls,
        *,
        uri: str,
        name: str,
        read: ResourceReadHandler,
        description: Optional[str] = None,
        mime_type: str = "text/plain",
    ) -> "Resource":
        """Create a text resource."""
        return cls(
            uri=uri,
            name=name,
            read=read,
            description=description,
            mime_type=mime_type,
        )
