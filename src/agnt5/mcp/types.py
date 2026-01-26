"""MCP type definitions for Python SDK."""

from dataclasses import dataclass, field
from typing import Any, Optional
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
    """Configuration for SSE transport."""

    url: str
    headers: dict[str, str] = field(default_factory=dict)

    def with_api_key(self, api_key: str) -> "SseConfig":
        """Add API key header."""
        self.headers["X-API-KEY"] = api_key
        return self


class TransportType(Enum):
    """Transport type for MCP connection."""

    STDIO = "stdio"
    SSE = "sse"


@dataclass
class ServerConfig:
    """Server configuration."""

    transport_type: TransportType
    stdio: Optional[StdioConfig] = None
    sse: Optional[SseConfig] = None

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
        """Create SSE server config."""
        return cls(
            transport_type=TransportType.SSE,
            sse=SseConfig(url=url, headers=headers or {}),
        )

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ServerConfig":
        """Create server config from dictionary.

        Supports two formats:
        1. {"command": "npx", "args": ["-y", "wikipedia-mcp"]}  -> stdio
        2. {"url": "https://...", "headers": {...}}  -> SSE
        """
        if "command" in config:
            return cls.from_stdio(
                command=config["command"],
                args=config.get("args", []),
                env=config.get("env", {}),
                cwd=config.get("cwd"),
            )
        elif "url" in config:
            return cls.from_sse(
                url=config["url"],
                headers=config.get("headers", {}),
            )
        else:
            raise ValueError("Invalid server config: must have 'command' or 'url'")
