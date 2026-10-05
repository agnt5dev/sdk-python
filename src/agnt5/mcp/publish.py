"""Publishing MCP servers with a deployment (AGNT5-1569).

An ``MCPServer`` that names functions, workflows or agents through
``add_function`` / ``add_workflow`` / ``add_agent`` is registered by the
worker as a component of type ``mcp``. Its definition follows the platform
contract (docs/architecture/mcp-server-definition.md in the AGNT5 repo,
``schema_version`` 1): the hosted MCP server then calls those components as
durable runs.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .server import MCPServer

SCHEMA_VERSION = 1

MODES = ("sync", "auto", "background")
VISIBILITIES = ("model", "app")
ANNOTATION_HINTS = ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint")
RESERVED_TOOL_NAMES = frozenset({"get_run", "cancel_run"})

#: The default ``view``: the AGNT5 run card that MCP Apps hosts (ChatGPT,
#: Claude, Cursor, VS Code) show for ``auto`` and ``background`` tools while
#: their run goes on. Pass ``view=None`` to turn it off for a tool.
RUN_VIEW = "run"
#: How a definition says a tool has no view.
_NO_VIEW = "none"

_TOOL_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_SERVER_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")

#: Input every published agent takes: the message, and an optional session to
#: continue.
AGENT_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "input": {"type": "string", "description": "The message for the agent."},
        "session_id": {"type": "string", "description": "Continue an earlier conversation."},
    },
    "required": ["input"],
}


@dataclass
class PublishedTool:
    """One tool of a published server and the component it calls."""

    name: str
    component_type: str  # function | workflow | agent
    component_name: str
    input_schema: dict[str, Any]
    output_schema: Optional[dict[str, Any]] = None
    title: Optional[str] = None
    description: Optional[str] = None
    mode: Optional[str] = None
    visibility: Optional[list[str]] = None
    annotations: dict[str, Any] = field(default_factory=dict)
    #: ``RUN_VIEW`` (the default, left out of the definition) or None (off).
    view: Optional[str] = RUN_VIEW

    def to_definition(self) -> dict[str, Any]:
        tool: dict[str, Any] = {
            "name": self.name,
            "component": {"type": self.component_type, "name": self.component_name},
            "input_schema": self.input_schema,
            # ChatGPT requires explicit hints; a tool that doesn't say it only
            # reads is treated as one that writes.
            "annotations": {"readOnlyHint": False, **self.annotations},
        }
        for key, value in (
            ("title", self.title),
            ("description", self.description),
            ("mode", self.mode),
            ("visibility", self.visibility),
            ("output_schema", self.output_schema),
        ):
            if value:
                tool[key] = value
        if self.view is None:
            tool["view"] = _NO_VIEW
        return tool


def check_tool_options(
    name: str,
    *,
    mode: Optional[str],
    visibility: Optional[list[str]],
    annotations: Optional[dict[str, Any]],
    view: Optional[str] = RUN_VIEW,
) -> dict[str, Any]:
    """Validate ``add_*`` options where the developer wrote them, so a typo
    fails at import time rather than as a rejected deployment."""
    if view is not None and view != RUN_VIEW:
        raise ValueError(f"view must be {RUN_VIEW!r} (the AGNT5 run card) or None (no view), not {view!r}")
    if not _TOOL_NAME.match(name):
        raise ValueError(f"MCP tool name {name!r} must be 1 to 128 letters, digits, '_', '-' or '.'")
    if name in RESERVED_TOOL_NAMES:
        raise ValueError(f"MCP tool name {name!r} is reserved for the built-in run tools")
    if mode is not None and mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, not {mode!r}")
    if visibility is not None:
        if not visibility or any(v not in VISIBILITIES for v in visibility) or len(set(visibility)) != len(visibility):
            raise ValueError(f"visibility must be a non-empty list of {', '.join(VISIBILITIES)}")
    clean: dict[str, Any] = {}
    for key, value in (annotations or {}).items():
        if key == "title":
            if not isinstance(value, str):
                raise ValueError("annotations['title'] must be a string")
        elif key in ANNOTATION_HINTS:
            if not isinstance(value, bool):
                raise ValueError(f"annotations[{key!r}] must be True or False")
        else:
            raise ValueError(f"unknown annotation {key!r}; use {', '.join(ANNOTATION_HINTS)} or title")
        clean[key] = value
    return clean


def object_schema(schema: Any) -> Optional[dict[str, Any]]:
    """MCP tool schemas must describe an object; anything else is left out."""
    if isinstance(schema, dict) and schema.get("type") == "object":
        return schema
    return None


def first_line(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    for line in text.strip().splitlines():
        if line.strip():
            return line.strip()
    return None


def valid_server_name(name: str) -> bool:
    """A published server's name is a URL segment: /mcp/{project}/{env}/{name}."""
    return bool(_SERVER_NAME.match(name))


_lock = threading.Lock()
_SERVERS: dict[str, "MCPServer"] = {}


class MCPServerRegistry:
    """Servers defined in the worker's code, published at startup."""

    @staticmethod
    def register(server: "MCPServer") -> None:
        with _lock:
            _SERVERS[server.info.id] = server

    @staticmethod
    def discard(name: str) -> None:
        with _lock:
            _SERVERS.pop(name, None)

    @staticmethod
    def get(name: str) -> Optional["MCPServer"]:
        return _SERVERS.get(name)

    @staticmethod
    def all() -> dict[str, "MCPServer"]:
        with _lock:
            return dict(_SERVERS)

    @staticmethod
    def clear() -> None:
        with _lock:
            _SERVERS.clear()
