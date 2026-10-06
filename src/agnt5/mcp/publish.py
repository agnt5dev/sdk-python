"""Publishing MCP servers with a deployment (AGNT5-1569).

An ``MCPServer`` that names functions, workflows or agents through
``add_function`` / ``add_workflow`` / ``add_agent`` is registered by the
worker as a component of type ``mcp``. Its definition follows the platform
contract (docs/architecture/mcp-server-definition.md in the AGNT5 repo,
``schema_version`` 1): the hosted MCP server then calls those components as
durable runs.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Union

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

#: The most one view's HTML may weigh, in bytes.
MAX_VIEW_BYTES = 2 * 1024 * 1024
#: The most the views of the servers a worker publishes may take of its
#: registration together, measured as they travel there (``registration_size``):
#: AGNT5 accepts a registration up to 4 MB.
MAX_VIEWS_BYTES = 3 * 1024 * 1024
_RESERVED_VIEW_NAMES = frozenset({RUN_VIEW, _NO_VIEW})

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
    #: ``RUN_VIEW`` (the default, left out of the definition), None (off), or
    #: the name of one of the server's own views (``add_view``).
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
        elif self.view != RUN_VIEW:
            tool["view"] = self.view
        return tool


@dataclass(frozen=True)
class MCPView:
    """An MCP App view a server ships: one self-contained HTML file that
    clients rendering MCP Apps (ChatGPT, Claude, Cursor, VS Code) show for
    the results of the tools that name it. Made by ``MCPServer.add_view``;
    pass it (or its name) as a tool's ``view``.

    The bundle travels with the worker's registration; AGNT5 stores it by
    ``sha256`` and serves it at ``ui://{server}/{name}/{sha256[:16]}``.
    """

    name: str
    sha256: str
    size: int
    #: The server that ships it.
    server: str
    html: str = field(repr=False)
    #: What the bundle takes of the worker's registration (``registration_size``).
    registration_bytes: int = 0

    def to_definition(self) -> dict[str, Any]:
        return {"name": self.name, "sha256": self.sha256, "size": self.size, "html": self.html}


def registration_size(html: str) -> int:
    """How many bytes a view bundle takes in the worker's registration.

    The definition carrying it travels as a JSON string inside a JSON
    webhook, so the HTML is JSON-escaped twice: ``"`` and ``\\`` take 4 bytes,
    ``\\b \\f \\n \\r \\t`` take 3, other control characters 7, everything else
    its UTF-8 length. The platform measures the same way
    (``ViewRegistrationBytes``), whatever JSON encoder either side uses.
    """
    size = len(html.encode("utf-8"))
    for ch in html:
        if ch in ('"', "\\"):
            size += 3
        elif ch in "\b\f\n\r\t":
            size += 2
        elif ch < "\x20":
            size += 6
    return size


def check_views_budget(servers: "list[MCPServer]") -> None:
    """Refuse a worker whose published servers' views together pass
    ``MAX_VIEWS_BYTES`` of its registration: the registration would be
    rejected whole, taking every component with it."""
    total = 0
    sizes = []
    for server in servers:
        for view in server.views.values():
            total += view.registration_bytes
            sizes.append(f"{server.info.id}/{view.name} {view.registration_bytes}")
    if total > MAX_VIEWS_BYTES:
        raise ValueError(
            f"this worker's MCP views take {total} bytes of its registration together; the limit is "
            f"{MAX_VIEWS_BYTES} ({', '.join(sizes)})"
        )


def load_view(
    server: str,
    name: str,
    html: Optional[str],
    path: Union[str, "os.PathLike[str]", None],
) -> MCPView:
    """Check a view where the developer added it, so a missing build or an
    oversized bundle fails at import time rather than as a rejected
    deployment."""
    if not isinstance(name, str) or not _SERVER_NAME.fullmatch(name):
        raise ValueError(f"view name {name!r} must be lowercase letters, digits, '-' and '_' (up to 63 characters)")
    if name in _RESERVED_VIEW_NAMES:
        raise ValueError(f"view name {name!r} is reserved ({RUN_VIEW!r} is the AGNT5 run card, {_NO_VIEW!r} means no view)")
    if (html is None) == (path is None):
        raise ValueError(f"add_view({name!r}, ...) takes one of html= or path=")
    if path is not None:
        try:
            # Too big is refused before anything is read.
            on_disk = os.stat(path).st_size
            if on_disk > MAX_VIEW_BYTES:
                raise ValueError(f"view {name!r} is {on_disk} bytes; the limit is {MAX_VIEW_BYTES}")
            with open(path, encoding="utf-8") as f:
                html = f.read()
        except FileNotFoundError:
            raise ValueError(
                f"view {name!r}: {os.fspath(path)} does not exist. Build it first: one self-contained "
                "HTML file (for example Vite with vite-plugin-singlefile)"
            ) from None
        except UnicodeDecodeError:
            raise ValueError(f"view {name!r}: {os.fspath(path)} is not UTF-8 text") from None
    if not isinstance(html, str) or not html.strip():
        raise ValueError(f"view {name!r} has no HTML")
    try:
        data = html.encode("utf-8")
    except UnicodeEncodeError:
        # Lone surrogates: not text UTF-8 (or the registration) can carry.
        raise ValueError(f"view {name!r}: its HTML contains unpaired UTF-16 surrogates") from None
    if len(data) > MAX_VIEW_BYTES:
        raise ValueError(f"view {name!r} is {len(data)} bytes; the limit is {MAX_VIEW_BYTES}")
    return MCPView(
        name=name,
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
        server=server,
        html=html,
        registration_bytes=registration_size(html),
    )


def check_tool_options(
    name: str,
    *,
    mode: Optional[str],
    visibility: Optional[list[str]],
    annotations: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Validate ``add_*`` options where the developer wrote them, so a typo
    fails at import time rather than as a rejected deployment."""
    if not _TOOL_NAME.fullmatch(name):
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
    return bool(_SERVER_NAME.fullmatch(name))


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
