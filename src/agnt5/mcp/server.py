"""Developer-facing MCP server support for the Python SDK."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from .._ids import generate_cid
from .._serialization import serialize_to_str
from ..agent import Agent
from ..function import FunctionContext
from ..tool import Tool
from .publish import (
    AGENT_INPUT_SCHEMA,
    MAX_VIEWS_BYTES,
    RUN_VIEW,
    SCHEMA_VERSION,
    MCPServerRegistry,
    MCPView,
    PublishedTool,
    check_tool_options,
    first_line,
    load_view,
    object_schema,
)
from .types import Prompt, Resource

#: A tool's ``view``: ``RUN_VIEW`` (the default), None, or one of the
#: server's own views, by handle or name.
ToolView = Union[MCPView, str, None]

# JSON-RPC 2.0 and MCP error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
RESOURCE_NOT_FOUND = -32002


class MCPServerError(Exception):
    """MCP server-specific error, answered with its JSON-RPC error ``code``."""

    def __init__(self, message: str, *, code: int = INTERNAL_ERROR) -> None:
        super().__init__(message)
        self.code = code


def _error_response(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _is_valid_id(value: Any) -> bool:
    # MCP request ids are strings or integers, never null.
    return isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool))


@dataclass
class _ServerInfo:
    id: str
    name: str
    version: str
    instructions: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MCPServer:
    """An MCP server built from AGNT5 functions, workflows and agents.

    Published with the deployment: tools added with ``add_function``,
    ``add_workflow`` or ``add_agent`` are served at
    ``https://api.agnt5.com/mcp/{project}/{env}/{name}``, each call running as
    a durable AGNT5 run::

        support = MCPServer("support", instructions="Order and ticket tools.")
        support.add_function("lookup_order", lookup_order, annotations={"readOnlyHint": True})
        support.add_workflow("triage_ticket", triage_ticket)  # mode="auto"

    The server's name is part of its URL: lowercase letters, digits, ``-``
    and ``_``. ``run_stdio()`` still serves it locally for development.

    A server can ship its own MCP App views (``add_view``) and show one for
    a tool's results instead of the built-in run card::

        board = support.add_view("order", path=VIEWS / "order.html")
        support.add_function("lookup_order", lookup_order, view=board)
    """

    def __init__(
        self,
        id: str,
        name: Optional[str] = None,
        version: Optional[str] = None,
        *,
        title: Optional[str] = None,
        tools: Optional[dict[str, Tool]] = None,
        agents: Optional[dict[str, Agent]] = None,
        workflows: Optional[dict[str, Any]] = None,
        prompts: Optional[dict[str, Prompt]] = None,
        resources: Optional[dict[str, Resource]] = None,
        instructions: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.info = _ServerInfo(
            id=id,
            name=name or id,
            version=version or "0.1.0",
            instructions=instructions,
            metadata=metadata or {},
        )
        self.title = title
        self._tools: dict[str, Tool] = dict(tools or {})
        self._agents: dict[str, Agent] = dict(agents or {})
        self._workflows: dict[str, Any] = dict(workflows or {})
        self._prompts: dict[str, Prompt] = dict(prompts or {})
        self._resources: dict[str, Resource] = dict(resources or {})
        self._published: dict[str, PublishedTool] = {}
        self._views: dict[str, MCPView] = {}
        MCPServerRegistry.register(self)

    def add_tool(self, name: str, tool: Tool) -> None:
        """Serve a tool over ``run_stdio`` only; it is not published."""
        self._tools[name] = tool

    def add_view(
        self,
        name: str,
        html: Optional[str] = None,
        *,
        path: Union[str, "os.PathLike[str]", None] = None,
    ) -> MCPView:
        """Ship an MCP App view with this server and return its handle; pass
        it (or ``name``) as a tool's ``view`` to show it for that tool's
        results in clients that render MCP Apps (ChatGPT, Claude, Cursor,
        VS Code).

        A view is one self-contained HTML file: give its text as ``html`` or
        the built file as ``path`` (for example Vite with
        ``vite-plugin-singlefile``). It is read now, so a missing build fails
        at import. Up to 2 MB per view. Views travel in the worker's
        registration, so the views of all the servers it publishes may take
        up to 3 MB of it, measured JSON-escaped as they travel
        (``registration_size``); a worker past that refuses to register.
        ``name`` follows the server-name rule, and ``run`` and ``none`` are
        reserved.

        The view gets the tool's result over the MCP Apps bridge: its
        ``structuredContent`` (the output, when it is an object) and its
        text. A call that hands off (``auto`` or ``background``) gives it the
        run handle instead, in ``_meta["com.agnt5/run"]``; the view can poll
        the built-in ``get_run`` tool through the host. Clients that don't
        render MCP Apps read the text.
        """
        view = load_view(self.info.id, name, html, path)
        if name in self._views:
            raise ValueError(f"MCP server {self.info.id!r} already has a view named {name!r}")
        # The server's own views; the worker checks those of every server it
        # publishes together when it registers.
        taken = sum(v.registration_bytes for v in self._views.values()) + view.registration_bytes
        if taken > MAX_VIEWS_BYTES:
            raise ValueError(
                f"view {name!r} would make MCP server {self.info.id!r}'s views take {taken} bytes of the "
                f"registration; the limit is {MAX_VIEWS_BYTES}"
            )
        self._views[name] = view
        return view

    def _resolve_view(self, view: ToolView) -> Optional[str]:
        """A tool's ``view`` as published: RUN_VIEW, None, or a view name."""
        if view is None or view == RUN_VIEW:
            return view
        if isinstance(view, MCPView):
            if self._views.get(view.name) is not view:
                raise ValueError(
                    f"view {view.name!r} belongs to MCP server {view.server!r}; "
                    f"add it to {self.info.id!r} with add_view"
                )
            return view.name
        if isinstance(view, str) and view in self._views:
            return view
        raise ValueError(
            f"view must be {RUN_VIEW!r} (the AGNT5 run card), None (no view) or a view added with "
            f"add_view, not {view!r}"
        )

    def add_function(
        self,
        name: str,
        function: Any,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        mode: Optional[str] = None,
        visibility: Optional[list[str]] = None,
        annotations: Optional[dict[str, Any]] = None,
        view: ToolView = RUN_VIEW,
    ) -> None:
        """Publish an ``@function`` as a tool. Functions wait for their result
        (``mode="sync"``) unless told otherwise; with ``mode="auto"`` or
        ``"background"`` they get the run card too (see ``add_workflow``).
        ``view=`` one of the server's own views (``add_view``) shows it for
        every call, sync ones too."""
        config = getattr(function, "_agnt5_config", None)
        if config is None or getattr(config, "handler", None) is None:
            raise TypeError(f"add_function({name!r}, ...) needs a function decorated with @function")
        self._publish(
            name, "function", config.name,
            input_schema=config.input_schema,
            output_schema=config.output_schema,
            default_description=(config.metadata or {}).get("description") or first_line(config.handler.__doc__),
            title=title, description=description, mode=mode, visibility=visibility, annotations=annotations,
            view=view,
        )

    def add_workflow(
        self,
        name: str,
        workflow: Any,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        mode: Optional[str] = None,
        visibility: Optional[list[str]] = None,
        annotations: Optional[dict[str, Any]] = None,
        view: ToolView = RUN_VIEW,
    ) -> None:
        """Publish a ``@workflow`` as a tool. By default a call waits up to the
        server's call budget, then hands back a run handle (``mode="auto"``).

        Clients that render MCP Apps (ChatGPT, Claude, Cursor, VS Code) show
        such calls as an AGNT5 run card with live status, steps and output;
        ``view=None`` turns the card off for this tool, and ``view=`` one of
        the server's own views (``add_view``) shows that instead."""
        self._workflows[name] = workflow
        config = getattr(workflow, "_agnt5_config", None)
        if config is None:
            # A plain callable still serves over run_stdio; only @workflow
            # handlers are registered components the platform can run.
            return
        self._publish(
            name, "workflow", config.name,
            input_schema=config.input_schema,
            output_schema=config.output_schema,
            default_description=(config.metadata or {}).get("description") or first_line(config.handler.__doc__),
            title=title, description=description, mode=mode, visibility=visibility, annotations=annotations,
            view=view,
        )

    def add_agent(
        self,
        name: str,
        agent: Agent,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        mode: Optional[str] = None,
        visibility: Optional[list[str]] = None,
        annotations: Optional[dict[str, Any]] = None,
        view: ToolView = RUN_VIEW,
    ) -> None:
        """Publish an agent as a tool that takes a message (and optionally a
        session to continue). Like workflows, its calls show the run card
        in MCP Apps clients unless ``view=None`` or one of the server's own
        views."""
        self._agents[name] = agent
        self._publish(
            name, "agent", agent.name,
            input_schema=AGENT_INPUT_SCHEMA,
            output_schema=None,
            default_description=first_line(getattr(agent, "description", None))
            or first_line(getattr(agent, "instructions", None)),
            title=title, description=description, mode=mode, visibility=visibility, annotations=annotations,
            view=view,
        )

    def _publish(
        self,
        name: str,
        component_type: str,
        component_name: str,
        *,
        input_schema: Any,
        output_schema: Any,
        default_description: Optional[str],
        title: Optional[str],
        description: Optional[str],
        mode: Optional[str],
        visibility: Optional[list[str]],
        annotations: Optional[dict[str, Any]],
        view: ToolView,
    ) -> None:
        resolved = self._resolve_view(view)
        clean = check_tool_options(name, mode=mode, visibility=visibility, annotations=annotations)
        if name in self._published:
            raise ValueError(f"MCP server {self.info.id!r} already has a tool named {name!r}")
        self._published[name] = PublishedTool(
            name=name,
            component_type=component_type,
            component_name=component_name,
            input_schema=object_schema(input_schema) or {"type": "object", "properties": {}},
            output_schema=object_schema(output_schema),
            title=title,
            description=description or default_description,
            mode=mode,
            visibility=list(visibility) if visibility else None,
            annotations=clean,
            view=resolved,
        )

    @property
    def published(self) -> bool:
        """Whether the worker publishes this server with the deployment."""
        return bool(self._published)

    def definition(self) -> dict[str, Any]:
        """The definition the worker registers (platform contract, version 1)."""
        definition: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "name": self.info.id,
            "tools": [tool.to_definition() for tool in self._published.values()],
        }
        if self.title:
            definition["title"] = self.title
        if self.info.instructions:
            definition["instructions"] = self.info.instructions
        if self._views:
            definition["views"] = [view.to_definition() for view in self._views.values()]
        return definition

    @property
    def views(self) -> dict[str, MCPView]:
        """The views this server ships (``add_view``), by name."""
        return dict(self._views)

    def add_prompt(self, name: str, prompt: Prompt) -> None:
        self._prompts[name] = prompt

    def add_resource(self, name: str, resource: Resource) -> None:
        self._resources[name] = resource

    async def run_stdio(self) -> None:
        """Serve MCP JSON-RPC over stdio using newline-delimited JSON."""
        await asyncio.to_thread(self._serve_stdio_sync)

    async def dispatch(self, request: Any) -> Optional[dict[str, Any]]:
        """Handle one JSON-RPC message and return the response to send.

        Returns ``None`` for a notification (a message without an ``id``) and
        for a response from the client: neither gets a reply. Exposed for
        tests and embeddings.
        """
        if not isinstance(request, dict):
            return _error_response(None, INVALID_REQUEST, "Invalid Request: expected a JSON object")
        has_id = "id" in request
        req_id = request.get("id")
        if has_id and not _is_valid_id(req_id):
            return _error_response(None, INVALID_REQUEST, "Invalid Request: id must be a string or an integer")
        if request.get("jsonrpc") != "2.0":
            return _error_response(req_id, INVALID_REQUEST, 'Invalid Request: "jsonrpc" must be "2.0"')
        method = request.get("method")
        if method is None and ("result" in request or "error" in request):
            # A response to a request this server never sends.
            return None
        if not isinstance(method, str) or not method:
            return _error_response(req_id, INVALID_REQUEST, 'Invalid Request: "method" must be a string')
        if not has_id:
            # Notifications (notifications/initialized, notifications/cancelled,
            # ...) are never answered, not even with an error.
            return None
        params = request.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return _error_response(req_id, INVALID_PARAMS, 'Invalid params: "params" must be an object')
        try:
            result = await self._handle_request(method, params)
        except MCPServerError as exc:
            return _error_response(req_id, exc.code, str(exc))
        except Exception as exc:
            return _error_response(req_id, INTERNAL_ERROR, str(exc) or type(exc).__name__)
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    async def _handle_line(self, line: bytes) -> Optional[dict[str, Any]]:
        """Parse one stdio line and dispatch it."""
        try:
            message = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return _error_response(None, PARSE_ERROR, f"Parse error: {exc}")
        return await self.dispatch(message)

    def _serve_stdio_sync(self) -> None:
        input_stream = sys.stdin.buffer
        output_stream = sys.stdout.buffer
        while True:
            raw = self._read_message(input_stream)
            if raw is None:
                return
            if not raw.strip():
                continue
            response = asyncio.run(self._handle_line(raw))
            if response is None:
                continue
            payload = json.dumps(response).encode("utf-8")
            self._write_message(output_stream, payload)

    @staticmethod
    def _read_message(stream: Any) -> Optional[bytes]:
        line = stream.readline()
        if not line:
            return None
        return line.rstrip(b"\r\n")

    @staticmethod
    def _write_message(stream: Any, payload: bytes) -> None:
        stream.write(payload + b"\n")
        stream.flush()

    async def _handle_request(self, method: str, params: dict[str, Any]) -> Any:
        if method == "initialize":
            return {
                "protocolVersion": "2025-11-25",
                "serverInfo": {
                    "name": self.info.name,
                    "version": self.info.version,
                },
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"listChanged": False},
                    "prompts": {"listChanged": False},
                },
            }

        if method in ("tools/list", "tools.list"):
            return {"tools": self._list_tools()}

        if method in ("tools/call", "tools.call"):
            name = params.get("name")
            if not isinstance(name, str) or not name:
                raise MCPServerError('Invalid params: "name" must be a tool name', code=INVALID_PARAMS)
            arguments = params.get("arguments")
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                raise MCPServerError('Invalid params: "arguments" must be an object', code=INVALID_PARAMS)
            return await self._call_tool(name, arguments)

        if method in ("prompts/list", "prompts.list"):
            return {"prompts": self._list_prompts()}

        if method in ("prompts/get", "prompts.get"):
            return await self._get_prompt(
                params.get("name", ""),
                params.get("arguments") or {},
            )

        if method in ("resources/list", "resources.list"):
            return {"resources": self._list_resources()}

        if method in ("resources/read", "resources.read"):
            return await self._read_resource(params.get("uri", ""))

        if method == "ping":
            return {}

        raise MCPServerError(f"Method not found: {method}", code=METHOD_NOT_FOUND)

    def _list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for name, tool in self._tools.items():
            tools.append(
                {
                    "name": name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                }
            )
        for name in self._agents:
            tools.append(
                {
                    "name": name,
                    "description": f"AGNT5 agent: {name}",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string", "description": "User input for the agent"},
                            "session_id": {"type": "string"},
                            "max_iterations": {"type": "integer"},
                        },
                        "required": ["input"],
                    },
                }
            )
        for name, workflow in self._workflows.items():
            config = getattr(workflow, "_agnt5_config", None)
            tools.append(
                {
                    "name": name,
                    "description": f"AGNT5 workflow: {name}",
                    "inputSchema": getattr(config, "input_schema", None)
                    or {
                        "type": "object",
                        "properties": {},
                    },
                }
            )
        return tools

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name in self._tools:
            call = self._invoke_tool(self._tools[name], arguments)
        elif name in self._agents:
            call = self._invoke_agent(self._agents[name], arguments)
        elif name in self._workflows:
            call = self._invoke_workflow(self._workflows[name], arguments)
        else:
            raise MCPServerError(f"Unknown tool: {name}", code=INVALID_PARAMS)
        try:
            result = await call
        except Exception as exc:
            # A tool execution error, not a protocol one, so the model can
            # read it and correct the call (as the hosted server does).
            return self._wrap_text_result(f"{name} failed: {exc}", is_error=True)
        return self._wrap_text_result(result)

    def _list_prompts(self) -> list[dict[str, Any]]:
        prompts: list[dict[str, Any]] = []
        for name, prompt in self._prompts.items():
            arguments_schema = prompt.arguments_schema or {}
            properties = arguments_schema.get("properties", {})
            required = set(arguments_schema.get("required", []))
            arguments = []
            for arg_name, schema in properties.items():
                arguments.append(
                    {
                        "name": arg_name,
                        "description": schema.get("description"),
                        "required": arg_name in required,
                    }
                )
            prompts.append(
                {
                    "name": name,
                    "description": prompt.description,
                    "arguments": arguments,
                }
            )
        return prompts

    async def _get_prompt(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        prompt = self._prompts.get(name)
        if prompt is None or prompt.handler is None:
            raise MCPServerError(f"Unknown prompt: {name}", code=INVALID_PARAMS)
        result = await prompt.handler(**arguments)
        if isinstance(result, str):
            messages = [{"role": "user", "content": {"type": "text", "text": result}}]
        elif isinstance(result, list):
            messages = result
        elif isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
        else:
            messages = [
                {
                    "role": "user",
                    "content": {"type": "text", "text": serialize_to_str(result)},
                }
            ]
        return {
            "description": prompt.description,
            "messages": messages,
        }

    def _list_resources(self) -> list[dict[str, Any]]:
        resources: list[dict[str, Any]] = []
        for resource in self._resources.values():
            resources.append(
                {
                    "uri": resource.uri,
                    "name": resource.name,
                    "description": resource.description,
                    "mimeType": resource.mime_type,
                }
            )
        return resources

    async def _read_resource(self, uri: str) -> dict[str, Any]:
        resource = next((r for r in self._resources.values() if r.uri == uri), None)
        if resource is None:
            raise MCPServerError(f"Resource not found: {uri}", code=RESOURCE_NOT_FOUND)
        result = await resource.read()
        if isinstance(result, bytes):
            text = result.decode("utf-8")
        else:
            text = result if isinstance(result, str) else serialize_to_str(result)
        return {
            "contents": [
                {
                    "uri": resource.uri,
                    "mimeType": resource.mime_type or "text/plain",
                    "text": text,
                }
            ]
        }

    async def _invoke_tool(self, tool: Tool, arguments: dict[str, Any]) -> Any:
        ctx = self._create_context(tool.name)
        return await tool.invoke(ctx, **arguments)

    async def _invoke_agent(self, agent: Agent, arguments: dict[str, Any]) -> Any:
        prompt = arguments.get("input")
        if not isinstance(prompt, str) or not prompt:
            raise MCPServerError("agent tools require a non-empty 'input' string")
        result = await agent.run(prompt, context=self._create_context(agent.name))
        return {
            "output": result.output,
            "tool_calls": result.tool_calls,
        }

    async def _invoke_workflow(self, workflow: Callable[..., Any], arguments: dict[str, Any]) -> Any:
        config = getattr(workflow, "_agnt5_config", None)
        if config is None:
            raise MCPServerError("workflow is missing _agnt5_config metadata")
        # Spread arguments as kwargs. The workflow wrapper auto-creates a
        # WorkflowContext when no positional Context is supplied, then forwards
        # kwargs to the handler. Callers must key `arguments` by the workflow's
        # parameter names (e.g. {"input": {...}}), matching how `_invoke_tool`
        # spreads arguments into a tool handler.
        return await workflow(**arguments)

    @staticmethod
    def _wrap_text_result(result: Any, *, is_error: bool = False) -> dict[str, Any]:
        if isinstance(result, str):
            text = result
        else:
            text = serialize_to_str(result)
        return {
            "content": [
                {
                    "type": "text",
                    "text": text,
                }
            ],
            "isError": is_error,
        }

    @staticmethod
    def _create_context(component_name: str) -> FunctionContext:
        run_id = f"mcp-{secrets.token_hex(8)}"
        return FunctionContext(
            run_id=run_id,
            correlation_id=generate_cid(),
            parent_correlation_id=component_name,
        )
