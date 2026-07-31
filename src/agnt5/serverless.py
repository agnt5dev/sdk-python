"""Workerless HTTP adapter for Python ASGI applications."""

from __future__ import annotations

import hashlib
import hmac
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlencode

import httpx

from ._ids import generate_cid
from ._serialization import deserialize, serialize
from .agent import AgentContext, AgentRegistry
from .context import runtime_context_from_metadata, set_current_context
from .function import FunctionContext, FunctionRegistry
from .tool import ToolRegistry
from .workflow import WorkflowRegistry

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "workerless.v1"
DEFAULT_MANIFEST_PATH = "/.well-known/agnt5"
INVOKE_PATH = "/agnt5/invoke"
SIGNATURE_VERSION = "workerless-hmac-sha256.v1"
SIGNATURE_MAX_SKEW_MS = 5 * 60 * 1000
INPUT_REF_KIND = "agnt5.object_store.signed_url.v1"
OUTPUT_UPLOAD_KIND = "agnt5.object_store.signed_url.v1"
OUTPUT_REF_KIND = "agnt5.object_store.ref.v1"
MAX_INPUT_REF_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_REF_BYTES = 64 * 1024 * 1024

HeaderMap = Mapping[str, str]
SigningSecretResolver = str | Callable[[], str | None] | Callable[[HeaderMap], str | None]
EnabledResolver = bool | Callable[[], bool] | Callable[[HeaderMap], bool]


@dataclass(frozen=True)
class WorkerlessComponent:
    name: str
    component_type: str
    config: Any


class WorkerlessSuspension(BaseException):
    def __init__(
        self,
        *,
        reason: str,
        checkpoint: dict[str, Any],
        ready_at_ms: int | None = None,
        timer_key: str | None = None,
        signal_name: str | None = None,
        waiting_step: str | None = None,
        deadline_ms: int | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.checkpoint = checkpoint
        self.ready_at_ms = ready_at_ms
        self.timer_key = timer_key
        self.signal_name = signal_name
        self.waiting_step = waiting_step
        self.deadline_ms = deadline_ms


class WorkerlessWaitingForUserInput(BaseException):
    def __init__(
        self,
        *,
        question: str,
        checkpoint: dict[str, Any],
        pause_index: int,
        input_type: str = "text",
        options: list[dict[str, Any]] | None = None,
        allow_custom: bool = False,
        skippable: bool = False,
        step_name: str | None = None,
    ) -> None:
        super().__init__(question)
        self.question = question
        self.checkpoint = checkpoint
        self.pause_index = pause_index
        self.input_type = input_type
        self.options = options
        self.allow_custom = allow_custom
        self.skippable = skippable
        self.step_name = step_name


class WorkerlessContext:
    """Request-scoped context for Python workerless workflows."""

    signal: None = None

    def __init__(
        self,
        *,
        invocation_id: str,
        run_id: str,
        attempt: int,
        component_name: str,
        checkpoints: Optional[dict[str, Any]] = None,
        deadline_ms: Optional[int] = None,
        yield_before_timeout_ms: int = 1000,
        metadata: Optional[dict[str, str]] = None,
    ) -> None:
        self.invocation_id = invocation_id
        self.run_id = run_id
        self.attempt = attempt
        self.component_name = component_name
        self.metadata = dict(metadata or {})
        self.runtime = runtime_context_from_metadata(self.metadata)
        self._runtime_context = None
        self._trace_metadata = self.metadata
        self.logger = logging.getLogger(f"agnt5.workerless.{component_name}")
        self._state: dict[str, Any] = {}
        self._checkpoints = dict(checkpoints or {})
        self._events: list[dict[str, Any]] = []
        self._deadline_ms = deadline_ms
        self._yield_before_timeout_ms = yield_before_timeout_ms
        self._pause_index = 0
        self._user_responses = _user_responses_from_metadata(self.metadata)
        self._signal_responses = _signal_responses_from_metadata(self.metadata)
        self.correlation_id = generate_cid()
        self.parent_correlation_id = ""
        self._correlation_id = self.correlation_id
        self._parent_correlation_id = self.parent_correlation_id

    async def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    async def set(self, key: str, value: Any) -> None:
        self._state[key] = value

    async def delete(self, key: str) -> bool:
        return self._state.pop(key, None) is not None

    async def step(self, name: str, func: Callable[[], Any] | Awaitable[Any]) -> Any:
        checkpoint_key = f"step:{name}"
        if checkpoint_key in self._checkpoints:
            return self._checkpoints[checkpoint_key]

        if inspect.isawaitable(func):
            result = await func
        elif callable(func):
            result = func()
            if inspect.isawaitable(result):
                result = await result
        else:
            raise TypeError("ctx.step() expects an awaitable or callable")

        self._checkpoints[checkpoint_key] = result
        return result

    async def yield_if_needed(self, reason: str = "budget") -> None:
        if self._deadline_ms is None:
            return
        if int(time.time() * 1000) + self._yield_before_timeout_ms < self._deadline_ms:
            return
        raise WorkerlessSuspension(
            reason=reason,
            checkpoint=self.checkpoint_snapshot(),
            deadline_ms=self._deadline_ms,
        )

    async def sleep(self, seconds: float, name: str | None = None) -> None:
        if seconds < 0:
            raise ValueError("sleep seconds must be non-negative")
        if seconds == 0:
            return

        timer_key = name or f"sleep_{int(seconds * 1000)}ms"
        started_at_ms = await self.step(timer_key, lambda: int(time.time() * 1000))
        ready_at_ms = int(started_at_ms) + int(seconds * 1000)
        if int(time.time() * 1000) >= ready_at_ms:
            return
        raise WorkerlessSuspension(
            reason="timer",
            checkpoint=self.checkpoint_snapshot(),
            ready_at_ms=ready_at_ms,
            timer_key=timer_key,
        )

    async def wait_for_user(
        self,
        question: str,
        *,
        input_type: str = "text",
        options: list[dict[str, Any]] | None = None,
        allow_custom: bool = False,
        skippable: bool = False,
    ) -> str | None:
        pause_index = self._pause_index
        self._pause_index += 1
        if pause_index in self._user_responses:
            return self._user_responses[pause_index]

        step_name = f"wait_for_user_{pause_index}"
        raise WorkerlessWaitingForUserInput(
            question=question,
            checkpoint=self.checkpoint_snapshot(),
            pause_index=pause_index,
            input_type=input_type,
            options=options,
            allow_custom=allow_custom,
            skippable=skippable,
            step_name=step_name,
        )

    async def wait_for_signal(self, signal_name: str, name: str | None = None) -> Any:
        waiting_step = name or signal_name
        response_key = f"{signal_name}:{waiting_step}"
        if response_key in self._signal_responses:
            return self._signal_responses[response_key]
        raise WorkerlessSuspension(
            reason="signal",
            checkpoint=self.checkpoint_snapshot(),
            signal_name=signal_name,
            waiting_step=waiting_step,
        )

    async def emit(
        self,
        event: str | dict[str, Any] | Any,
        data: Any = None,
        *,
        metadata: dict[str, str] | None = None,
        correlation_id: str | None = None,
        parent_event_id: str | None = None,
        step_key: str | None = None,
        data_type: str | None = None,
    ) -> None:
        self._append_event(
            event,
            data,
            metadata=metadata,
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
            step_key=step_key,
            data_type=data_type,
        )

    def _append_event(
        self,
        event: str | dict[str, Any] | Any,
        data: Any = None,
        *,
        metadata: dict[str, str] | None = None,
        correlation_id: str | None = None,
        parent_event_id: str | None = None,
        step_key: str | None = None,
        data_type: str | None = None,
    ) -> None:
        if isinstance(event, str):
            payload: dict[str, Any] = {"event_type": event, "data": data}
        elif isinstance(event, dict):
            payload = dict(event)
        elif hasattr(event, "to_response_fields"):
            fields = event.to_response_fields()
            payload = {
                "event_type": fields.get("event_type", getattr(event, "event_type", "")),
                "data": _decode_event_output(fields.get("output_data")),
            }
        else:
            payload = {
                "event_type": getattr(event, "event_type", type(event).__name__),
                "data": getattr(event, "output_data", data),
            }
        payload.setdefault("timestamp_ns", time.time_ns())
        if metadata:
            payload["metadata"] = metadata
        if correlation_id:
            payload["correlation_id"] = correlation_id
        if parent_event_id:
            payload["parent_event_id"] = parent_event_id
        if step_key:
            payload["step_key"] = step_key
        if data_type:
            payload["data_type"] = data_type
        self._events.append(payload)

    def checkpoint_snapshot(self) -> dict[str, Any]:
        return dict(self._checkpoints)

    def set_checkpoint(self, key: str, value: Any) -> None:
        self._checkpoints[key] = value

    def events_snapshot(self) -> list[dict[str, Any]]:
        return list(self._events)


class _WorkerlessFunctionContext(FunctionContext):
    """Normal SDK function/tool context that returns emitted events in the HTTP response."""

    def __init__(self, parent: WorkerlessContext) -> None:
        super().__init__(
            run_id=parent.run_id,
            correlation_id=parent.correlation_id,
            parent_correlation_id=parent.parent_correlation_id,
            attempt=parent.attempt,
            runtime_context=None,
            trace_metadata=parent.metadata,
        )
        self._workerless_parent = parent

    def emit(self, event: Any) -> Any:
        self._workerless_parent._append_event(event)
        return event

    async def emit_async(self, event: Any) -> Any:
        self._workerless_parent._append_event(event)
        return event


class _WorkerlessAgentContext(AgentContext):
    """Agent context that keeps conversation state inside the invoke checkpoint."""

    def __init__(self, parent: WorkerlessContext, agent_name: str, session_id: str) -> None:
        super().__init__(
            run_id=parent.run_id,
            agent_name=agent_name,
            session_id=session_id,
            parent_context=parent,  # type: ignore[arg-type]
            attempt=parent.attempt,
            runtime_context=None,
            correlation_id=parent.correlation_id,
            parent_correlation_id=parent.parent_correlation_id,
            trace_metadata=parent.metadata,
        )
        self._memo = None
        self.saved_messages: list[Any] = []

    async def get_conversation_history(self) -> list[Any]:
        return []

    async def save_conversation_history(self, messages: list[Any]) -> None:
        self.saved_messages = list(messages)

    def emit(self, event: Any) -> Any:
        return event


class ServerlessApp:
    """ASGI app implementing the AGNT5 workerless HTTP protocol."""

    def __init__(
        self,
        *,
        service_name: str | None = None,
        service_version: str | None = None,
        functions: list[Any] | None = None,
        workflows: list[Any] | None = None,
        tools: list[Any] | None = None,
        agents: list[Any] | None = None,
        signing_secret: SigningSecretResolver | None = None,
        enabled: EnabledResolver = True,
    ) -> None:
        self.service_name = service_name
        self.service_version = service_version
        self.signing_secret = signing_secret
        self.enabled = enabled
        self.components = _collect_components(
            functions=functions,
            workflows=workflows,
            tools=tools,
            agents=agents,
        )
        self._manifest = _build_manifest(service_name, service_version, self.components)

    def manifest(self) -> dict[str, Any]:
        return dict(self._manifest)

    async def __call__(self, scope: dict[str, Any], receive: Callable[[], Any], send: Callable[[dict[str, Any]], Any]) -> None:
        if scope.get("type") != "http":
            raise RuntimeError("agnt5.serverless only supports ASGI HTTP scopes")

        body = await _read_asgi_body(receive)
        status, payload, headers = await self.handle_http(
            method=str(scope.get("method", "")),
            path=str(scope.get("path", "")),
            headers=_headers_from_scope(scope),
            body=body,
            url=_url_from_scope(scope),
        )
        await _send_json(send, status, payload, headers)

    async def handle_starlette_request(self, request: Any) -> Any:
        try:
            from starlette.responses import Response  # pyright: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover - exercised only without Starlette installed.
            raise ImportError("Install FastAPI or Starlette to use handle_starlette_request().") from exc

        status, payload, headers = await self.handle_http(
            method=request.method,
            path=request.url.path,
            headers={key.lower(): value for key, value in request.headers.items()},
            body=await request.body(),
            url=str(request.url),
        )
        return Response(
            content=serialize(payload),
            status_code=status,
            media_type="application/json",
            headers=headers,
        )

    def mount_fastapi(self, app: Any) -> None:
        if not hasattr(app, "add_route"):
            raise TypeError("mount_fastapi() expects a FastAPI or Starlette application")

        async def manifest_route(request: Any) -> Any:
            return await self.handle_starlette_request(request)

        async def invoke_route(request: Any) -> Any:
            return await self.handle_starlette_request(request)

        app.add_route(
            DEFAULT_MANIFEST_PATH,
            manifest_route,
            methods=["GET"],
            include_in_schema=False,
        )
        app.add_route(
            INVOKE_PATH,
            invoke_route,
            methods=["POST"],
            include_in_schema=False,
        )

    async def handle_http(
        self,
        *,
        method: str,
        path: str,
        headers: HeaderMap,
        body: bytes,
        url: str = "",
    ) -> tuple[int, dict[str, Any], dict[str, str]]:
        normalized_method = method.upper()
        if normalized_method in {"GET", "POST"} and path in {DEFAULT_MANIFEST_PATH, INVOKE_PATH}:
            enabled = await _resolve_enabled(self.enabled, headers)
            if not enabled:
                return _failed("WORKERLESS_DISABLED", "AGNT5 serverless endpoint is disabled", 503)

        if normalized_method == "GET" and path == DEFAULT_MANIFEST_PATH:
            return 200, self.manifest(), {}
        if normalized_method == "POST" and path == INVOKE_PATH:
            return await self._handle_invoke(headers=headers, body=body, url=url)
        return 404, {"error": "not_found", "message": "AGNT5 workerless route not found"}, {}

    async def _handle_invoke(
        self,
        *,
        headers: HeaderMap,
        body: bytes,
        url: str,
    ) -> tuple[int, dict[str, Any], dict[str, str]]:
        signature_failure = await _verify_signature(headers, body, self.signing_secret)
        if signature_failure:
            return signature_failure

        try:
            payload = deserialize(body) if body else {}
        except Exception as exc:
            return _failed("WORKERLESS_INVALID_REQUEST", f"request body must be JSON: {exc}", 400)
        if not isinstance(payload, dict):
            return _failed("WORKERLESS_INVALID_REQUEST", "request body must be a JSON object", 400)

        protocol_version = payload.get("protocol_version")
        if protocol_version and protocol_version != PROTOCOL_VERSION:
            return _failed("WORKERLESS_PROTOCOL_MISMATCH", f"unsupported protocol_version {protocol_version}", 400)

        component_type = _normalize_component_type(payload.get("component_type"))
        component_name = str(payload.get("component_name") or "").strip()
        if not component_type or not component_name:
            return _failed("WORKERLESS_COMPONENT_REQUIRED", "component_type and component_name are required", 400)

        component = next(
            (
                candidate
                for candidate in self.components
                if candidate.component_type == component_type and candidate.name == component_name
            ),
            None,
        )
        if component is None:
            return _failed(
                "WORKERLESS_COMPONENT_NOT_FOUND",
                f"{component_type} component {component_name} was not found",
                404,
            )

        input_status, input_value = await _resolve_input(payload)
        if input_status:
            return input_status

        run_id = str(payload.get("run_id") or headers.get("x-agnt5-run-id") or f"workerless-{int(time.time() * 1000)}")
        metadata = _string_map(payload.get("metadata"))
        ctx = WorkerlessContext(
            invocation_id=f"workerless-{run_id}",
            run_id=run_id,
            attempt=_safe_int(payload.get("attempt"), 0),
            component_name=component_name,
            checkpoints=_checkpoint_storage_from_payload(payload.get("checkpoint")),
            deadline_ms=_optional_int((payload.get("budget") or {}).get("deadline_ms")),
            yield_before_timeout_ms=_safe_int((payload.get("budget") or {}).get("yield_before_timeout_ms"), 1000),
            metadata=metadata,
        )

        token = set_current_context(ctx)  # type: ignore[arg-type]
        try:
            output = await _invoke_component(component, ctx, input_value)
            response_status, response_body = await _completed_response(
                output,
                _checkpoint_payload_from_storage(ctx.checkpoint_snapshot()),
                payload.get("output_upload"),
            )
            if response_status:
                return response_status
            response_body = _with_events(response_body, ctx.events_snapshot())
            return 200, response_body, {}
        except WorkerlessSuspension as exc:
            body_payload: dict[str, Any] = {
                "status": "suspended",
                "reason": exc.reason,
                "checkpoint": _checkpoint_payload_from_storage(exc.checkpoint),
            }
            if exc.ready_at_ms is not None:
                body_payload["ready_at_ms"] = exc.ready_at_ms
            if exc.timer_key:
                body_payload["timer_key"] = exc.timer_key
            if exc.signal_name:
                body_payload["signal_name"] = exc.signal_name
            if exc.waiting_step:
                body_payload["waiting_step"] = exc.waiting_step
            if exc.deadline_ms is not None:
                body_payload["budget"] = {"deadline_ms": exc.deadline_ms}
            return 200, _with_events(body_payload, ctx.events_snapshot()), {}
        except WorkerlessWaitingForUserInput as exc:
            body_payload = {
                "status": "suspended",
                "reason": "user_input_required",
                "checkpoint": _checkpoint_payload_from_storage(exc.checkpoint),
                "question": exc.question,
                "input_type": exc.input_type,
                "pause_index": exc.pause_index,
                "allow_custom": exc.allow_custom,
                "skippable": exc.skippable,
            }
            if exc.options is not None:
                body_payload["options"] = exc.options
            if exc.step_name:
                body_payload["waiting_step"] = exc.step_name
            return 200, _with_events(body_payload, ctx.events_snapshot()), {}
        except Exception as exc:
            logger.exception("workerless Python handler failed")
            return _failed(
                "WORKERLESS_HANDLER_FAILED",
                f"{type(exc).__name__}: {exc}",
                500,
                retryable=True,
            )
        finally:
            from .context import _current_context

            _current_context.reset(token)


def serve(
    *,
    service_name: str | None = None,
    service_version: str | None = None,
    functions: list[Any] | None = None,
    workflows: list[Any] | None = None,
    tools: list[Any] | None = None,
    agents: list[Any] | None = None,
    signing_secret: SigningSecretResolver | None = None,
    enabled: EnabledResolver = True,
) -> ServerlessApp:
    return ServerlessApp(
        service_name=service_name,
        service_version=service_version,
        functions=functions,
        workflows=workflows,
        tools=tools,
        agents=agents,
        signing_secret=signing_secret,
        enabled=enabled,
    )


def _collect_components(
    *,
    functions: list[Any] | None,
    workflows: list[Any] | None,
    tools: list[Any] | None,
    agents: list[Any] | None,
) -> list[WorkerlessComponent]:
    collected: list[WorkerlessComponent] = []

    function_configs = [_component_config(item) for item in functions] if functions is not None else list(FunctionRegistry.all().values())
    workflow_configs = [_component_config(item) for item in workflows] if workflows is not None else list(WorkflowRegistry.all().values())
    tool_configs = list(tools) if tools is not None else list(ToolRegistry.all().values())
    agent_configs = list(agents) if agents is not None else list(AgentRegistry.all().values())

    for config in function_configs:
        if config is not None:
            collected.append(WorkerlessComponent(name=config.name, component_type="function", config=config))
    for config in workflow_configs:
        if config is not None:
            collected.append(WorkerlessComponent(name=config.name, component_type="workflow", config=config))
    for config in tool_configs:
        if config is not None:
            collected.append(WorkerlessComponent(name=config.name, component_type="tool", config=config))
    for config in agent_configs:
        if config is not None:
            collected.append(WorkerlessComponent(name=config.name, component_type="agent", config=config))
    return collected


def _component_config(item: Any) -> Any:
    if item is None:
        return None
    if hasattr(item, "_agnt5_config"):
        return item._agnt5_config
    if hasattr(item, "name") and hasattr(item, "handler"):
        return item
    return None


def _build_manifest(
    service_name: str | None,
    service_version: str | None,
    components: list[WorkerlessComponent],
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "service_name": service_name,
        "service_version": service_version,
        "components": [_manifest_component(component) for component in components],
    }


def _manifest_component(component: WorkerlessComponent) -> dict[str, Any]:
    config = component.config
    payload: dict[str, Any] = {
        "name": component.name,
        "type": component.component_type,
        "component_type": component.component_type,
    }
    if getattr(config, "input_schema", None):
        payload["input_schema"] = config.input_schema
    if getattr(config, "output_schema", None):
        payload["output_schema"] = config.output_schema
    metadata = dict(getattr(config, "metadata", None) or {})
    if component.component_type == "tool":
        metadata.update(
            {
                "description": getattr(config, "description", ""),
                "requires_confirmation": bool(getattr(config, "confirmation", False)),
            }
        )
    elif component.component_type == "agent":
        metadata["model"] = getattr(config, "model_name", getattr(config, "model", ""))
    if metadata:
        payload["metadata"] = metadata
    triggers = getattr(config, "triggers", None)
    if triggers:
        payload["triggers"] = [_trigger_payload(trigger) for trigger in triggers]
    if getattr(config, "retries", None):
        payload.setdefault("flow_control", {})["retry"] = _retry_payload(config.retries)
    return payload


def _trigger_payload(trigger: Any) -> dict[str, Any]:
    return {
        "trigger_id": getattr(trigger, "trigger_id", ""),
        "trigger_type": getattr(trigger, "trigger_type", ""),
        "event_name": getattr(trigger, "event_name", ""),
        "filter_expression": getattr(trigger, "filter_expression", ""),
        "input_mapping": getattr(trigger, "input_mapping", ""),
        "batch_window_ms": getattr(trigger, "batch_window_ms", 0),
        "delay_expression": getattr(trigger, "delay_expression", ""),
    }


def _retry_payload(retry_policy: Any) -> dict[str, Any]:
    return {
        "max_attempts": getattr(retry_policy, "max_attempts", 3),
        "initial_interval_ms": getattr(retry_policy, "initial_interval_ms", 1000),
        "max_interval_ms": getattr(retry_policy, "max_interval_ms", 60000),
    }


async def _invoke_component(component: WorkerlessComponent, ctx: WorkerlessContext, input_value: Any) -> Any:
    config = component.config
    if component.component_type == "function":
        function_ctx = _WorkerlessFunctionContext(ctx)
        return await _call_handler(config.handler, function_ctx, input_value)
    if component.component_type == "tool":
        arguments = input_value if isinstance(input_value, dict) else {}
        return await config.invoke(_WorkerlessFunctionContext(ctx), **arguments)
    if component.component_type == "agent":
        return await _invoke_agent(config, ctx, input_value)
    return await _call_handler(config.handler, ctx, input_value)


async def _invoke_agent(agent: Any, ctx: WorkerlessContext, input_value: Any) -> str:
    session_id = _agent_session_id(input_value, ctx.metadata, ctx.run_id)
    checkpoint_history = _agent_history_from_checkpoint(ctx.checkpoint_snapshot(), session_id)
    history = checkpoint_history if checkpoint_history is not None else _agent_history_from_input(input_value)
    user_message = _agent_user_message(input_value)
    agent_ctx = _WorkerlessAgentContext(ctx, agent.name, session_id)
    output: str | None = None

    async for event in agent.stream(user_message, context=agent_ctx, history=history):
        await ctx.emit(event)
        if getattr(event, "event_type", "") != "agent.completed":
            continue
        output_data = getattr(event, "output_data", None)
        if isinstance(output_data, dict):
            output = str(output_data.get("output", ""))

    if output is None:
        raise RuntimeError(f"Agent '{agent.name}' completed without producing a result")

    updated_messages = _normalize_agent_messages(agent_ctx.saved_messages)
    if not updated_messages:
        updated_messages = [
            *history,
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": output},
        ]
    ctx.set_checkpoint(
        _agent_session_checkpoint_key(session_id),
        {"messages": updated_messages, "updated_at_ms": int(time.time() * 1000)},
    )
    await ctx.emit(
        {
            "event_type": "session.created",
            "data": {
                "session_id": session_id,
                "agent_name": agent.name,
                "turn_count": len(updated_messages),
            },
            "metadata": {
                "session_id": session_id,
                "agent_name": agent.name,
                "session_type": "agent",
            },
        }
    )
    return output


async def _call_handler(handler: Callable[..., Any], ctx: Any, input_value: Any) -> Any:
    wants_ctx = _handler_wants_context(handler)
    if isinstance(input_value, dict):
        result = handler(ctx, **input_value) if wants_ctx else handler(**input_value)
    elif input_value is None:
        result = handler(ctx) if wants_ctx else handler()
    else:
        result = handler(ctx, input_value) if wants_ctx else handler(input_value)
    if inspect.isasyncgen(result):
        items = []
        async for item in result:
            items.append(item)
        return items
    if inspect.isawaitable(result):
        return await result
    return result


def _handler_wants_context(handler: Callable[..., Any]) -> bool:
    try:
        parameters = list(inspect.signature(handler).parameters.values())
    except (TypeError, ValueError):
        return True
    return bool(parameters and parameters[0].name == "ctx")


def _agent_session_id(input_value: Any, metadata: dict[str, str], run_id: str) -> str:
    if isinstance(input_value, dict):
        session_id = _nonempty_string(input_value.get("session_id"))
        if session_id:
            return session_id
    return _nonempty_string(metadata.get("session_id")) or run_id


def _agent_user_message(input_value: Any) -> str:
    if isinstance(input_value, str):
        return input_value.strip()
    if isinstance(input_value, dict):
        for key in ("prompt", "message", "input"):
            message = _nonempty_string(input_value.get(key))
            if message:
                return message
    if input_value is None:
        return ""
    try:
        return serialize(input_value).decode("utf-8")
    except Exception:
        return str(input_value)


def _agent_history_from_input(input_value: Any) -> list[dict[str, str]]:
    if not isinstance(input_value, dict):
        return []
    return _normalize_agent_messages(input_value.get("history"))


def _agent_history_from_checkpoint(
    storage: dict[str, Any],
    session_id: str,
) -> list[dict[str, str]] | None:
    checkpoint = storage.get(_agent_session_checkpoint_key(session_id))
    if not isinstance(checkpoint, dict):
        return None
    return _normalize_agent_messages(checkpoint.get("messages"))


def _agent_session_checkpoint_key(session_id: str) -> str:
    return f"agent_session:{session_id}"


def _normalize_agent_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    messages: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            raw_role = item.get("role")
            raw_content = item.get("content")
        else:
            raw_role = getattr(item, "role", None)
            raw_content = getattr(item, "content", None)
        role_value = getattr(raw_role, "value", raw_role)
        role = _nonempty_string(role_value) or "user"
        content = str(raw_content or "")
        messages.append({"role": role, "content": content})
    return messages


def _nonempty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_component_type(value: Any) -> str | None:
    candidate = str(value or "").strip().lower()
    if candidate in {"function", "workflow", "tool", "agent"}:
        return candidate
    return None


async def _resolve_input(payload: dict[str, Any]) -> tuple[tuple[int, dict[str, Any], dict[str, str]] | None, Any]:
    has_input = "input" in payload
    input_ref = payload.get("input_ref")
    if has_input and input_ref:
        return _failed("WORKERLESS_INPUT_REF_INVALID", "workerless invoke payload must not include both input and input_ref", 400), None
    if not input_ref:
        return None, payload.get("input", {})
    if not isinstance(input_ref, dict):
        return _failed("WORKERLESS_INPUT_REF_INVALID", "workerless input_ref must be an object", 400), None
    if input_ref.get("kind") != INPUT_REF_KIND:
        return _failed("WORKERLESS_INPUT_REF_UNSUPPORTED", "workerless input_ref kind is unsupported", 400), None
    if str(input_ref.get("method") or "GET").upper() != "GET":
        return _failed("WORKERLESS_INPUT_REF_UNSUPPORTED", "workerless input_ref method is unsupported", 400), None
    url = str(input_ref.get("url") or "")
    if not url.startswith(("http://", "https://")):
        return _failed("WORKERLESS_INPUT_REF_INVALID", "workerless input_ref url must be http or https", 400), None
    size_bytes = _optional_int(input_ref.get("size_bytes"))
    if size_bytes is None:
        return _failed("WORKERLESS_INPUT_REF_INVALID", "workerless input_ref size_bytes is required", 400), None
    if size_bytes > MAX_INPUT_REF_BYTES:
        return _failed("WORKERLESS_INPUT_REF_TOO_LARGE", f"workerless input_ref exceeds {MAX_INPUT_REF_BYTES} bytes", 413), None
    sha256 = str(input_ref.get("sha256") or "")
    if not sha256:
        return _failed("WORKERLESS_INPUT_REF_INVALID", "workerless input_ref sha256 is required", 400), None
    expires_at_ms = _optional_int(input_ref.get("expires_at_ms"))
    if expires_at_ms is not None and expires_at_ms < int(time.time() * 1000):
        return _failed("WORKERLESS_INPUT_REF_EXPIRED", "workerless input_ref has expired", 410), None

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
        body = response.content
    except Exception as exc:
        return _failed("WORKERLESS_INPUT_REF_FETCH_FAILED", f"workerless input_ref could not be fetched: {exc}", 502), None
    if len(body) != size_bytes:
        return _failed("WORKERLESS_INPUT_REF_INVALID", "workerless input_ref size_bytes did not match fetched payload", 400), None
    if hashlib.sha256(body).hexdigest() != sha256:
        return _failed("WORKERLESS_INPUT_REF_CHECKSUM_MISMATCH", "workerless input_ref sha256 did not match fetched payload", 400), None
    try:
        return None, deserialize(body)
    except Exception as exc:
        return _failed("WORKERLESS_INPUT_REF_INVALID", f"workerless input_ref payload must be JSON: {exc}", 400), None


async def _completed_response(
    output: Any,
    checkpoint: dict[str, Any] | None,
    upload: Any,
) -> tuple[tuple[int, dict[str, Any], dict[str, str]] | None, dict[str, Any]]:
    inline_body = {"status": "completed", "output": output, "checkpoint": checkpoint}
    if not upload:
        return None, inline_body
    if not isinstance(upload, dict):
        return _failed("WORKERLESS_OUTPUT_UPLOAD_INVALID", "workerless output_upload must be an object", 400), {}
    threshold_bytes = _optional_int(upload.get("threshold_bytes"))
    try:
        output_bytes = serialize(output)
    except Exception as exc:
        return _failed("WORKERLESS_OUTPUT_SERIALIZATION_FAILED", f"workerless output could not be serialized: {exc}", 500), {}
    if threshold_bytes is None or threshold_bytes < 0 or len(output_bytes) <= threshold_bytes:
        return None, inline_body

    validation_error = _validate_output_upload(upload)
    if validation_error:
        return validation_error, {}
    max_bytes = _safe_int(upload.get("max_bytes"), MAX_OUTPUT_REF_BYTES)
    if len(output_bytes) > max_bytes:
        return _failed("WORKERLESS_PAYLOAD_TOO_LARGE", f"workerless output payload exceeds {max_bytes} bytes", 413), {}

    content_type = str(upload.get("content_type") or "application/json")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.put(str(upload["url"]), content=output_bytes, headers={"content-type": content_type})
            response.raise_for_status()
    except Exception as exc:
        return _failed("WORKERLESS_OUTPUT_REF_UPLOAD_FAILED", f"workerless output_ref upload failed: {exc}", 502), {}

    return None, {
        "status": "completed",
        "output_ref": {
            "kind": OUTPUT_REF_KIND,
            "ref": str(upload["ref"]),
            "size_bytes": len(output_bytes),
            "sha256": hashlib.sha256(output_bytes).hexdigest(),
            "content_type": content_type,
        },
        "checkpoint": checkpoint,
    }


def _validate_output_upload(upload: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, str]] | None:
    if upload.get("kind") != OUTPUT_UPLOAD_KIND:
        return _failed("WORKERLESS_OUTPUT_UPLOAD_UNSUPPORTED", "workerless output_upload kind is unsupported", 400)
    if str(upload.get("method") or "PUT").upper() != "PUT":
        return _failed("WORKERLESS_OUTPUT_UPLOAD_UNSUPPORTED", "workerless output_upload method is unsupported", 400)
    url = str(upload.get("url") or "")
    if not url.startswith(("http://", "https://")):
        return _failed("WORKERLESS_OUTPUT_UPLOAD_INVALID", "workerless output_upload url must be http or https", 400)
    if not upload.get("ref"):
        return _failed("WORKERLESS_OUTPUT_UPLOAD_INVALID", "workerless output_upload ref is required", 400)
    expires_at_ms = _optional_int(upload.get("expires_at_ms"))
    if expires_at_ms is not None and expires_at_ms < int(time.time() * 1000):
        return _failed("WORKERLESS_OUTPUT_UPLOAD_EXPIRED", "workerless output_upload has expired", 410)
    return None


async def _verify_signature(
    headers: HeaderMap,
    body: bytes,
    signing_secret: SigningSecretResolver | None,
) -> tuple[int, dict[str, Any], dict[str, str]] | None:
    secret = await _resolve_signing_secret(signing_secret, headers)
    if not secret:
        return None

    signature_version = headers.get("x-agnt5-signature-version")
    signature = headers.get("x-agnt5-signature")
    timestamp = headers.get("x-agnt5-timestamp")
    attempt_id = headers.get("x-agnt5-attempt-id")
    if not signature or not timestamp or not attempt_id:
        return _failed("WORKERLESS_SIGNATURE_MISSING", "workerless invoke signature headers are required", 401)
    if signature_version != SIGNATURE_VERSION:
        return _failed("WORKERLESS_SIGNATURE_VERSION_UNSUPPORTED", "workerless invoke signature version is unsupported", 401)
    try:
        timestamp_ms = int(timestamp)
    except ValueError:
        return _failed("WORKERLESS_SIGNATURE_TIMESTAMP_INVALID", "workerless invoke signature timestamp is invalid", 401)
    if abs(int(time.time() * 1000) - timestamp_ms) > SIGNATURE_MAX_SKEW_MS:
        return _failed("WORKERLESS_SIGNATURE_EXPIRED", "workerless invoke signature timestamp is outside the allowed window", 401)

    digest = hmac.new(
        secret.encode("utf-8"),
        f"{timestamp}.{attempt_id}.".encode("utf-8") + body,
        hashlib.sha256,
    ).hexdigest()
    expected = f"sha256={digest}"
    if not hmac.compare_digest(signature, expected):
        return _failed("WORKERLESS_SIGNATURE_INVALID", "workerless invoke signature is invalid", 401)
    return None


async def _resolve_signing_secret(
    resolver: SigningSecretResolver | None,
    headers: HeaderMap,
) -> str | None:
    if resolver is None:
        return None
    if isinstance(resolver, str):
        return resolver
    resolver_fn: Callable[..., Any] = resolver
    result = resolver_fn(headers) if _callable_accepts_arg(resolver_fn) else resolver_fn()
    if inspect.isawaitable(result):
        result = await result
    return str(result) if result else None


async def _resolve_enabled(enabled: EnabledResolver, headers: HeaderMap) -> bool:
    if isinstance(enabled, bool):
        return enabled
    enabled_fn: Callable[..., Any] = enabled
    result = enabled_fn(headers) if _callable_accepts_arg(enabled_fn) else enabled_fn()
    if inspect.isawaitable(result):
        result = await result
    return bool(result)


def _callable_accepts_arg(fn: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return len(signature.parameters) > 0


def _checkpoint_storage_from_payload(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        return {}
    steps = checkpoint.get("steps")
    storage = dict(steps) if isinstance(steps, dict) else {}
    agent_sessions = checkpoint.get("agent_sessions")
    if isinstance(agent_sessions, dict):
        for session_id, value in agent_sessions.items():
            normalized_id = _nonempty_string(session_id)
            if normalized_id and isinstance(value, dict):
                storage[_agent_session_checkpoint_key(normalized_id)] = {
                    "messages": _normalize_agent_messages(value.get("messages")),
                    "updated_at_ms": _optional_int(value.get("updated_at_ms")),
                }
    return storage


def _checkpoint_payload_from_storage(storage: dict[str, Any]) -> dict[str, Any] | None:
    if not storage:
        return None
    steps: dict[str, Any] = {}
    agent_sessions: dict[str, Any] = {}
    for key, value in storage.items():
        if key.startswith("agent_session:") and isinstance(value, dict):
            session_id = key.removeprefix("agent_session:")
            if session_id:
                checkpoint: dict[str, Any] = {
                    "messages": _normalize_agent_messages(value.get("messages")),
                }
                updated_at_ms = _optional_int(value.get("updated_at_ms"))
                if updated_at_ms is not None:
                    checkpoint["updated_at_ms"] = updated_at_ms
                agent_sessions[session_id] = checkpoint
        else:
            steps[key] = value
    payload: dict[str, Any] = {}
    if steps:
        payload["steps"] = steps
    if agent_sessions:
        payload["agent_sessions"] = agent_sessions
    return payload or None


def _with_events(body: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    if events:
        body = dict(body)
        body["events"] = events
    return body


def _failed(
    code: str,
    message: str,
    status: int,
    *,
    retryable: bool | None = None,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    error: dict[str, Any] = {"code": code, "message": message}
    if retryable is not None:
        error["retryable"] = retryable
    return status, {"status": "failed", "error": error}, {}


async def _read_asgi_body(receive: Callable[[], Any]) -> bytes:
    chunks: list[bytes] = []
    while True:
        message = await receive()
        if message.get("type") != "http.request":
            continue
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            break
    return b"".join(chunks)


async def _send_json(
    send: Callable[[dict[str, Any]], Any],
    status: int,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> None:
    response_headers = [(b"content-type", b"application/json")]
    for key, value in headers.items():
        response_headers.append((key.lower().encode("latin-1"), value.encode("latin-1")))
    await send({"type": "http.response.start", "status": status, "headers": response_headers})
    await send({"type": "http.response.body", "body": serialize(payload), "more_body": False})


def _headers_from_scope(scope: dict[str, Any]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in scope.get("headers", []):
        headers[key.decode("latin-1").lower()] = value.decode("latin-1")
    return headers


def _url_from_scope(scope: dict[str, Any]) -> str:
    scheme = scope.get("scheme", "http")
    headers = _headers_from_scope(scope)
    host = headers.get("host", "127.0.0.1")
    path = scope.get("path", "")
    query = scope.get("query_string", b"")
    query_string = query.decode("latin-1") if isinstance(query, bytes) else urlencode(query)
    return f"{scheme}://{host}{path}" + (f"?{query_string}" if query_string else "")


def _string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any, default: int) -> int:
    parsed = _optional_int(value)
    return parsed if parsed is not None else default


def _user_responses_from_metadata(metadata: dict[str, str]) -> dict[int, str | None]:
    if "user_response" not in metadata:
        return {}
    pause_index = _safe_int(metadata.get("pause_index"), 0)
    return {pause_index: metadata.get("user_response")}


def _signal_responses_from_metadata(metadata: dict[str, str]) -> dict[str, Any]:
    raw = metadata.get("signals")
    if not raw:
        return {}
    try:
        parsed = deserialize(raw)
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _decode_event_output(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return deserialize(value)
        except Exception:
            return value.decode("utf-8", errors="replace")
    return value


__all__ = ["ServerlessApp", "WorkerlessContext", "serve"]
