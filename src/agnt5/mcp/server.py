"""Developer-facing MCP server support for the Python SDK."""

from __future__ import annotations

import asyncio
import json
import secrets
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .._ids import generate_cid
from .._serialization import serialize_to_str
from ..agent import Agent
from ..function import FunctionContext
from ..tool import Tool
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


class MCPServer:
    """Expose AGNT5 primitives as an MCP server.

    This is a stdio-first v1 intended for local developer workflows.
    """

    def __init__(
        self,
        id: str,
        name: str,
        version: str,
        *,
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
            name=name,
            version=version,
            instructions=instructions,
            metadata=metadata or {},
        )
        self._tools: dict[str, Tool] = dict(tools or {})
        self._agents: dict[str, Agent] = dict(agents or {})
        self._workflows: dict[str, Any] = dict(workflows or {})
        self._prompts: dict[str, Prompt] = dict(prompts or {})
        self._resources: dict[str, Resource] = dict(resources or {})

    def add_tool(self, name: str, tool: Tool) -> None:
        self._tools[name] = tool

    def add_agent(self, name: str, agent: Agent) -> None:
        self._agents[name] = agent

    def add_workflow(self, name: str, workflow: Any) -> None:
        self._workflows[name] = workflow

    def add_prompt(self, name: str, prompt: Prompt) -> None:
        self._prompts[name] = prompt

    def add_resource(self, name: str, resource: Resource) -> None:
        self._resources[name] = resource

    async def run_stdio(self) -> None:
        """Serve MCP JSON-RPC over stdio using Content-Length framing."""
        await asyncio.to_thread(self._serve_stdio_sync)

    async def run_sse(self, host: str = "127.0.0.1", port: int = 0) -> None:
        raise NotImplementedError("MCPServer.run_sse() is not implemented yet")

    async def run_http(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        path: str = "/mcp",
    ) -> None:
        raise NotImplementedError("MCPServer.run_http() is not implemented yet")

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
        content_length: Optional[int] = None
        while True:
            line = stream.readline()
            if not line:
                return None
            if line in (b"\r\n", b"\n"):
                break
            if line.lower().startswith(b"content-length:"):
                content_length = int(line.split(b":", 1)[1].strip())

        if content_length is None:
            raise MCPServerError("Missing Content-Length header")
        body = stream.read(content_length)
        if not body:
            return None
        return body

    @staticmethod
    def _write_message(stream: Any, payload: bytes) -> None:
        header = f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8")
        stream.write(header)
        stream.write(payload)
        stream.flush()

    async def _handle_request(self, method: str, params: dict[str, Any]) -> Any:
        if method == "initialize":
            return {
                "protocolVersion": "2024-11-05",
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

        if method == "initialized":
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
        result = await agent.run_sync(prompt, context=self._create_context(agent.name))
        return {
            "output": result.output,
            "tool_calls": result.tool_calls,
        }

    async def _invoke_workflow(self, workflow: Callable[..., Any], arguments: dict[str, Any]) -> Any:
        config = getattr(workflow, "_agnt5_config", None)
        if config is None:
            raise MCPServerError("workflow is missing _agnt5_config metadata")
        return await workflow(arguments)

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
