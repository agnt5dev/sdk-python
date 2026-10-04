"""Developer-facing MCP server support for the Python SDK."""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

from .._ids import generate_cid
from .._serialization import serialize_to_str
from ..agent import Agent
from ..function import FunctionContext
from ..tool import Tool
from .publish import (
    AGENT_INPUT_SCHEMA,
    SCHEMA_VERSION,
    MCPServerRegistry,
    PublishedTool,
    check_tool_options,
    first_line,
    object_schema,
)
from .types import Prompt, Resource


class MCPServerError(Exception):
    """MCP server-specific error."""


@dataclass
class _ServerInfo:
    id: str
    name: str
    version: str
    instructions: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _NetworkServerHandle:
    host: str
    port: int
    server: asyncio.Server

    async def close(self) -> None:
        self.server.close()
        await self.server.wait_closed()


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
        MCPServerRegistry.register(self)

    def add_tool(self, name: str, tool: Tool) -> None:
        """Serve a tool over ``run_stdio`` only; it is not published."""
        self._tools[name] = tool

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
    ) -> None:
        """Publish an ``@function`` as a tool. Functions wait for their result
        (``mode="sync"``) unless told otherwise."""
        config = getattr(function, "_agnt5_config", None)
        if config is None or getattr(config, "handler", None) is None:
            raise TypeError(f"add_function({name!r}, ...) needs a function decorated with @function")
        self._publish(
            name, "function", config.name,
            input_schema=config.input_schema,
            output_schema=config.output_schema,
            default_description=(config.metadata or {}).get("description") or first_line(config.handler.__doc__),
            title=title, description=description, mode=mode, visibility=visibility, annotations=annotations,
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
    ) -> None:
        """Publish a ``@workflow`` as a tool. By default a call waits up to the
        server's call budget, then hands back a run handle (``mode="auto"``)."""
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
    ) -> None:
        """Publish an agent as a tool that takes a message (and optionally a
        session to continue)."""
        self._agents[name] = agent
        self._publish(
            name, "agent", agent.name,
            input_schema=AGENT_INPUT_SCHEMA,
            output_schema=None,
            default_description=first_line(getattr(agent, "description", None))
            or first_line(getattr(agent, "instructions", None)),
            title=title, description=description, mode=mode, visibility=visibility, annotations=annotations,
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
    ) -> None:
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
        return definition

    def add_prompt(self, name: str, prompt: Prompt) -> None:
        self._prompts[name] = prompt

    def add_resource(self, name: str, resource: Resource) -> None:
        self._resources[name] = resource

    async def run_stdio(self) -> None:
        """Serve MCP JSON-RPC over stdio using newline-delimited JSON."""
        await asyncio.to_thread(self._serve_stdio_sync)

    async def run_http(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        path: str = "/mcp",
    ) -> None:
        handle = await self._start_http_server(host, port, path)
        await handle.server.serve_forever()

    async def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a JSON-RPC request. Exposed for tests and embeddings."""
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}
        try:
            result = await self._handle_request(method, params)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": result,
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32603,
                    "message": str(exc),
                },
            }

    def _serve_stdio_sync(self) -> None:
        input_stream = sys.stdin.buffer
        output_stream = sys.stdout.buffer
        while True:
            raw = self._read_message(input_stream)
            if raw is None:
                return
            request = json.loads(raw.decode("utf-8"))
            response = asyncio.run(self.dispatch(request))
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

    async def _start_http_server(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        path: str = "/mcp",
    ) -> _NetworkServerHandle:
        normalized_path = self._normalize_path(path)

        async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await self._handle_http_connection(reader, writer, normalized_path)

        server = await asyncio.start_server(handler, host, port)
        bound_port = server.sockets[0].getsockname()[1]
        return _NetworkServerHandle(host=host, port=bound_port, server=server)

    async def _handle_http_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        path: str,
    ) -> None:
        try:
            method, target, headers, body = await self._read_http_request(reader)
            if not self._is_allowed_origin(headers.get("origin"), headers.get("host")):
                await self._write_http_response(writer, 403, b"forbidden origin", "text/plain")
                return

            url = urlsplit(target)
            if url.path != path:
                await self._write_http_response(writer, 404, b"not found", "text/plain")
                return
            if method == "GET":
                await self._write_http_response(writer, 405, b"GET stream is not supported", "text/plain")
                return
            if method != "POST":
                await self._write_http_response(writer, 405, b"method not allowed", "text/plain")
                return

            request = json.loads(body.decode("utf-8"))
            if "id" not in request:
                await self.dispatch(request)
                await self._write_http_response(writer, 202, b"", "text/plain")
                return

            response = await self.dispatch(request)
            await self._write_json_response(writer, 200, response)
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _read_http_request(
        self,
        reader: asyncio.StreamReader,
    ) -> tuple[str, str, dict[str, str], bytes]:
        header_bytes = await reader.readuntil(b"\r\n\r\n")
        header_text = header_bytes.decode("iso-8859-1")
        lines = header_text.split("\r\n")
        method, target, _version = lines[0].split(" ", 2)
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()

        content_length = int(headers.get("content-length", "0") or "0")
        body = await reader.readexactly(content_length) if content_length else b""
        return method.upper(), target, headers, body

    async def _write_json_response(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        payload: dict[str, Any],
    ) -> None:
        await self._write_http_response(
            writer,
            status,
            json.dumps(payload).encode("utf-8"),
            "application/json",
        )

    async def _write_http_response(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        body: bytes,
        content_type: str,
    ) -> None:
        reason = {
            200: "OK",
            202: "Accepted",
            403: "Forbidden",
            404: "Not Found",
            405: "Method Not Allowed",
            500: "Internal Server Error",
        }.get(status, "OK")
        header = (
            f"HTTP/1.1 {status} {reason}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n"
            "\r\n"
        )
        writer.write(header.encode("utf-8") + body)
        await writer.drain()

    @staticmethod
    def _normalize_path(path: str) -> str:
        return path if path.startswith("/") else f"/{path}"

    @staticmethod
    def _is_allowed_origin(origin: Optional[str], host: Optional[str]) -> bool:
        if not origin:
            return True
        if not host:
            return False
        try:
            return urlsplit(origin).netloc == host
        except Exception:
            return False

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

        if method in ("notifications/initialized", "initialized"):
            return {"ok": True}

        if method in ("tools/list", "tools.list"):
            return {"tools": self._list_tools()}

        if method in ("tools/call", "tools.call"):
            return await self._call_tool(
                params.get("name", ""),
                params.get("arguments") or {},
            )

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
            return {"pong": True}

        raise MCPServerError(f"method not found: {method}")

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
            result = await self._invoke_tool(self._tools[name], arguments)
            return self._wrap_text_result(result)
        if name in self._agents:
            result = await self._invoke_agent(self._agents[name], arguments)
            return self._wrap_text_result(result)
        if name in self._workflows:
            result = await self._invoke_workflow(self._workflows[name], arguments)
            return self._wrap_text_result(result)
        raise MCPServerError(f"unknown tool: {name}")

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
            raise MCPServerError(f"unknown prompt: {name}")
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
            raise MCPServerError(f"unknown resource: {uri}")
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
    def _wrap_text_result(result: Any) -> dict[str, Any]:
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
            "isError": False,
        }

    @staticmethod
    def _create_context(component_name: str) -> FunctionContext:
        run_id = f"mcp-{secrets.token_hex(8)}"
        return FunctionContext(
            run_id=run_id,
            correlation_id=generate_cid(),
            parent_correlation_id=component_name,
        )
